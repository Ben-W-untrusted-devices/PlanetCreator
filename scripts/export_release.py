#!/usr/bin/env python
"""Export a trained checkpoint for distribution.

Writes ``<out>/<name>.pt`` (weights only, fp16, no optimiser/discriminator state: ~17 MB
for the 8.6M-param terrain model), ``<name>.json`` (provenance: step, validation
metrics, git commit, date) and ``<name>_preview.png`` (the last validation grid).

Usage: export_release.py runs/joint2_adv/last.pt --name joint_latest --out models
"""

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

from planetcreator.terrain import TerrainNet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--name", default="joint_latest")
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--fp32", action="store_true", help="keep full precision (twice the size)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model, ck = TerrainNet.load(args.checkpoint)
    state = model.state_dict()
    if not args.fp32:
        state = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}
    dest = args.out / f"{args.name}.pt"
    torch.save({"cfg": ck["cfg"], "state": state, "step": ck.get("step")}, dest)

    run_dir = args.checkpoint.parent
    metrics = None
    log = run_dir / "log.csv"
    if log.exists():
        rows = log.read_text().strip().splitlines()
        if len(rows) > 1:
            head, last = rows[0].split(","), rows[-1].split(",")
            metrics = dict(zip(head, last))
    previews = sorted(run_dir.glob("val_*.png"))
    if previews:
        shutil.copy(previews[-1], args.out / f"{args.name}_preview.png")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        commit = None
    meta = {
        "name": args.name,
        "source_checkpoint": str(args.checkpoint),
        "step": ck.get("step"),
        "cfg": ck["cfg"],
        "params": sum(p.numel() for p in model.parameters()),
        "precision": "fp32" if args.fp32 else "fp16",
        "validation": metrics,
        "git_commit": commit,
        "exported": datetime.now(UTC).isoformat(timespec="seconds"),
        "train_log": (run_dir / "train.log").read_text().splitlines()[0] if (run_dir / "train.log").exists() else None,
    }
    (args.out / f"{args.name}.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB), {args.name}.json, preview")


if __name__ == "__main__":
    main()
