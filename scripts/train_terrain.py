#!/usr/bin/env python
"""Train the terrain model. Usage: train_terrain.py --mode joint --steps 20000 --out runs/joint"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from planetcreator.dataset import FINE_LAYERS, CubePatchDataset, PatchSpec
from planetcreator.features import HEIGHT_SCALE
from planetcreator.terrain import TerrainConfig, TerrainNet, pick_device


def to_dev(b: dict, device: str) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in b.items() if k != "meta"}


def sample_grid(model, batch, device, path: Path, n: int = 8) -> None:
    """Rows: input height (coarse in joint mode), predicted height (joint), predicted rgb, true rgb, true height."""
    model.eval()
    with torch.no_grad():
        b = to_dev({k: v[:n] for k, v in batch.items()}, device)
        cond, ctx = model.inputs(b)
        out = {k: v.cpu() for k, v in model(cond, ctx).items()}
        hin = cond[:, 0:1].cpu()
    model.train()

    def grey(h):  # normalised height -> 3ch grey
        return ((h * HEIGHT_SCALE).clamp(-500, 5500) + 500) / 6000

    rows = [torch.cat([grey(hin[i]).expand(3, -1, -1) for i in range(n)], dim=2)]
    if "height" in out:
        rows.append(torch.cat([grey(out["height"][i]).expand(3, -1, -1) for i in range(n)], dim=2))
    rows.append(torch.cat(list(out["rgb"]), dim=2))
    rows.append(torch.cat(list(b["rgb"].cpu()), dim=2))
    rows.append(torch.cat([grey(b["height"][i].cpu() / HEIGHT_SCALE).expand(3, -1, -1) for i in range(n)], dim=2))
    img = (torch.cat(rows, dim=1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    tot = {"rgb": 0.0, "height_m": 0.0}
    n = 0
    for b in loader:
        b = to_dev(b, device)
        cond, ctx = model.inputs(b)
        out = model(cond, ctx)
        t = model.target(b)
        k = len(b["rgb"])
        tot["rgb"] += torch.nn.functional.l1_loss(out["rgb"], t["rgb"]).item() * k
        if "height" in out:
            tot["height_m"] += torch.nn.functional.l1_loss(out["height"], t["height"]).item() * HEIGHT_SCALE * k
        n += k
    model.train()
    return {k: v / n for k, v in tot.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", type=Path, default=Path("data/cube/4096"))
    ap.add_argument("--out", type=Path, default=Path("runs/terrain"))
    ap.add_argument("--mode", choices=("colour", "joint"), default="joint")
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
    ap.add_argument("--resume", type=Path, default=None, help="weights to start from (optimiser restarts)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} mode={args.mode} out={args.out}")

    train_ds = CubePatchDataset(
        args.cube, PatchSpec(args.patch, FINE_LAYERS, True, "train", args.min_land), length=args.steps * args.batch, seed=args.seed
    )
    val_ds = CubePatchDataset(
        args.cube, PatchSpec(args.patch, FINE_LAYERS, True, "val", args.min_land, augment=False), length=args.val_patches, seed=1
    )
    dl_kw = {"num_workers": args.workers, "persistent_workers": args.workers > 0, "pin_memory": device == "cuda"}
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=False, drop_last=True, **dl_kw)
    val_dl = DataLoader(val_ds, batch_size=args.batch, **dl_kw)
    val_batch = next(iter(val_dl))

    if args.resume:
        model, ck = TerrainNet.load(args.resume, device)
        model.cfg.adv_weight = args.adv
        print(f"resumed weights from {args.resume} (step {ck.get('step')})")
    else:
        model = TerrainNet(
            TerrainConfig(mode=args.mode, base=args.base, depth=args.depth, adv_weight=args.adv,
                          ctx_pad=train_ds.pad, ctx_factor=train_ds.f)
        )
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    disc = d_opt = None
    if args.adv > 0:
        disc = model.make_discriminator().to(device)
        d_opt = torch.optim.Adam(disc.parameters(), lr=args.d_lr, betas=(0.0, 0.99))
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
          + (f" + D {sum(p.numel() for p in disc.parameters()) / 1e6:.2f}M" if disc else ""))

    with open(args.out / "log.csv", "a", newline="") as log:
        train(args, model, opt, sched, disc, d_opt, train_dl, val_dl, val_batch, device, log)


def train(args, model, opt, sched, disc, d_opt, train_dl, val_dl, val_batch, device, log) -> None:
    writer = csv.writer(log)
    if log.tell() == 0:
        writer.writerow(["step", "loss", "l1", "h_l1_m", "adv", "d", "val_rgb", "val_height_m", "lr", "sec"])
    t0 = time.time()
    model.train()
    for step, b in enumerate(train_dl, start=1):
        if step > args.steps:
            break
        b = to_dev(b, device)
        cond, ctx = model.inputs(b)
        target = model.target(b)
        out = model(cond, ctx)
        d_val = float("nan")
        if disc is not None:
            d_loss = model.d_loss(disc(cond, model.stack_out(target)), disc(cond, model.stack_out(out).detach()))
            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            d_opt.step()
            d_val = d_loss.item()
        loss, parts = model.loss(out, target, disc(cond, model.stack_out(out)) if disc is not None else None)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            extra = f" h {parts['h_l1_m']:.0f}m" if "h_l1_m" in parts else ""
            extra += f" adv {parts['adv']:.3f} d {d_val:.3f}" if disc is not None else ""
            print(f"step {step} loss {loss.item():.4f} l1 {parts['l1']:.4f}{extra} {(time.time() - t0) / step:.2f}s/it", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, val_dl, device)
            writer.writerow([step, loss.item(), parts["l1"], parts.get("h_l1_m", float("nan")), parts.get("adv", float("nan")),
                             d_val, val["rgb"], val["height_m"], sched.get_last_lr()[0], time.time() - t0])
            log.flush()
            sample_grid(model, val_batch, device, args.out / f"val_{step:06d}.png")
            model.save(args.out / "last.pt", step=step, disc=disc.state_dict() if disc else None)
            model.save(args.out / f"step_{step:06d}.pt", step=step)  # GAN runs oscillate; keep every checkpoint
            print(f"  val rgb {val['rgb']:.4f} height {val['height_m']:.0f}m  saved", flush=True)


if __name__ == "__main__":
    main()
