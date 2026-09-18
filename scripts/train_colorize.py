#!/usr/bin/env python
"""Train the colouriser. Usage: train_colorize.py --cube data/cube/4096 --steps 20000"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from planetcreator.colorize import ColorizeConfig, Colorizer, pick_device
from planetcreator.dataset import CubePatchDataset, PatchSpec
from planetcreator.features import COND_LAYERS, build_cond

LAYERS = (*COND_LAYERS, "rgb")


def to_dev(b: dict, device: str) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in b.items() if k != "meta"}


def sample_grid(model, batch, device, path: Path, n: int = 8) -> None:
    """Rows: height (grey), prediction, target."""
    model.eval()
    with torch.no_grad():
        b = {k: v[:n] for k, v in batch.items()}
        pred = model(build_cond(to_dev(b, device))).cpu()
    model.train()
    h = b["height"][:, 0].clamp(-500, 5500).add(500).div(6000)
    rows = [
        torch.cat([h[i].expand(3, -1, -1) for i in range(len(h))], dim=2),
        torch.cat(list(pred), dim=2),
        torch.cat(list(b["rgb"]), dim=2),
    ]
    img = (torch.cat(rows, dim=1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for b in loader:
        b = to_dev(b, device)
        tot += torch.nn.functional.l1_loss(model(build_cond(b)), b["rgb"]).item() * len(b["rgb"])
        n += len(b["rgb"])
    model.train()
    return tot / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", type=Path, default=Path("data/cube/4096"))
    ap.add_argument("--out", type=Path, default=Path("runs/colorize"))
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--min-land", type=float, default=0.3)
    ap.add_argument("--adv", type=float, default=0.0, help="adversarial weight; >0 trains a PatchGAN too")
    ap.add_argument("--d-lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-patches", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} out={args.out}")

    train_ds = CubePatchDataset(
        args.cube, PatchSpec(args.patch, LAYERS, "train", args.min_land), length=args.steps * args.batch, seed=args.seed
    )
    val_ds = CubePatchDataset(
        args.cube, PatchSpec(args.patch, LAYERS, "val", args.min_land, augment=False), length=args.val_patches, seed=1
    )
    dl_kw = {"num_workers": args.workers, "persistent_workers": args.workers > 0, "pin_memory": device == "cuda"}
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=False, drop_last=True, **dl_kw)
    val_dl = DataLoader(val_ds, batch_size=args.batch, **dl_kw)
    val_batch = next(iter(val_dl))

    step = 0
    if args.resume:
        # Resume weights; the optimiser/schedule restart so a run can e.g. add --adv to an L1 model.
        model, ck = Colorizer.load(args.resume, device)
        model.cfg.adv_weight = args.adv
        model.to(device)
        print(f"resumed weights from {args.resume} (step {ck.get('step')})")
    else:
        model = Colorizer(ColorizeConfig(base=args.base, depth=args.depth, adv_weight=args.adv)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    disc = d_opt = None
    if args.adv > 0:
        disc = model.make_discriminator().to(device)
        d_opt = torch.optim.Adam(disc.parameters(), lr=args.d_lr, betas=(0.0, 0.99))
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
          + (f" + D {sum(p.numel() for p in disc.parameters()) / 1e6:.2f}M" if disc else ""))

    with open(args.out / "log.csv", "a", newline="") as log:
        train(args, model, opt, sched, disc, d_opt, step, train_dl, val_dl, val_batch, device, log)


def train(args, model, opt, sched, disc, d_opt, step, train_dl, val_dl, val_batch, device, log) -> None:
    writer = csv.writer(log)
    if log.tell() == 0:
        writer.writerow(["step", "loss", "l1", "grad", "adv", "d", "val_l1", "lr", "sec"])
    t0 = time.time()
    model.train()
    for b in train_dl:
        if step >= args.steps:
            break
        b = to_dev(b, device)
        cond = build_cond(b)
        pred = model(cond)
        d_val = float("nan")
        if disc is not None:
            d_loss = model.d_loss(disc(cond, b["rgb"]), disc(cond, pred.detach()))
            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            d_opt.step()
            d_val = d_loss.item()
        loss, parts = model.loss(pred, b["rgb"], disc(cond, pred) if disc is not None else None)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            extra = f" adv {parts['adv']:.3f} d {d_val:.3f}" if disc is not None else ""
            print(f"step {step} loss {loss.item():.4f} l1 {parts['l1']:.4f}{extra} {(time.time() - t0) / step:.2f}s/it", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            val_l1 = evaluate(model, val_dl, device)
            writer.writerow([step, loss.item(), parts["l1"], parts["grad"], parts.get("adv", float("nan")), d_val,
                             val_l1, sched.get_last_lr()[0], time.time() - t0])
            log.flush()
            sample_grid(model, val_batch, device, args.out / f"val_{step:06d}.png")
            model.save(args.out / "last.pt", step=step, disc=disc.state_dict() if disc else None)
            print(f"  val_l1 {val_l1:.4f}  saved", flush=True)


if __name__ == "__main__":
    main()
