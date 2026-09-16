#!/usr/bin/env python3
"""List the sweep's trained arms as `label=checkpoint` lines, for the shell.

Arm directories are named after the training recipe (`kiwi-mix-frac040-frozen`);
results and plots use the family-prefixed label (`qe-frac040-frozen`). This maps
between them and picks each arm's checkpoint, so the slurm wrappers do not have
to reimplement the Lightning/W&B directory layout.

  python experiments/length_training/list_arms.py                     # best per arm
  python experiments/length_training/list_arms.py --select last
  python experiments/length_training/list_arms.py --arms frac000 frac100
  python experiments/length_training/list_arms.py --newer_than 2026-08-18

`--newer_than` matters after a relaunch: an arm directory can still hold runs
from a superseded sweep, and "best val_kendall" would happily pick one of those.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from fdtem.comet_io import best_checkpoint

# on-disk arm-directory prefix -> results label prefix (longest prefix first)
FAMILIES = (("kiwi-mix-", "qe-"), ("mix-", "da-"))


def label_for(arm_dir: str) -> str:
    """`kiwi-mix-frac040-frozen` -> `qe-frac040-frozen`."""
    for prefix, label in FAMILIES:
        if arm_dir.startswith(prefix):
            return label + arm_dir[len(prefix):]
    return arm_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="~/scratch/checkpoints/retrain-wmt")
    ap.add_argument("--select", choices=("best", "last"), default="best")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="Keep only arm directories containing one of these.")
    ap.add_argument("--newer_than", default=None,
                    help="ISO date or a file path; older checkpoints are skipped.")
    args = ap.parse_args()

    cutoff = None
    if args.newer_than:
        p = Path(args.newer_than).expanduser()
        cutoff = (p.stat().st_mtime if p.exists()
                  else datetime.fromisoformat(args.newer_than).timestamp())

    root = Path(args.root).expanduser()
    if not root.is_dir():
        raise SystemExit(f"no checkpoint root at {root}")

    found = 0
    for arm in sorted(d for d in root.iterdir() if d.is_dir()):
        if args.arms and not any(a in arm.name for a in args.arms):
            continue
        ckpt = best_checkpoint(arm, prefer=args.select)
        if ckpt is None:
            print(f"# {arm.name}: no checkpoint yet", file=sys.stderr)
            continue
        if cutoff is not None and ckpt.stat().st_mtime < cutoff:
            print(f"# {arm.name}: checkpoint predates the cutoff — skipped "
                  f"({ckpt})", file=sys.stderr)
            continue
        print(f"{label_for(arm.name)}={ckpt}")
        found += 1
    if not found:
        raise SystemExit(f"no usable arm checkpoints under {root}")


if __name__ == "__main__":
    main()
