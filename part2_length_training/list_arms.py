"""List the trained arms as `label=checkpoint` lines, for the shell.

Arm directories are named after the recipe (`kiwi-mix-frac040-frozen`); results use
the family-prefixed label (`qe-frac040-frozen`). ArmLabel maps between them.

  python -m part2_length_training.list_arms                      # best per arm
  python -m part2_length_training.list_arms --select last
  python -m part2_length_training.list_arms --arms frac000 frac100
  python -m part2_length_training.list_arms --newer_than 2026-08-18   # skip superseded runs
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from common.comet_models import best_checkpoint
from common.paths import CHECKPOINTS
from part2_length_training.models import ArmLabel


def label_for(dir_name: str) -> str:
    try:
        return ArmLabel.from_dir(dir_name).label
    except ValueError:
        return dir_name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(CHECKPOINTS / "retrain-wmt-v3"))
    ap.add_argument("--select", choices=("best", "last"), default="best")
    ap.add_argument("--arms", nargs="+", default=None, help="keep only arm directories containing one of these")
    ap.add_argument("--newer_than", default=None, help="ISO date or a file path; older checkpoints are skipped")
    args = ap.parse_args()

    cutoff = None
    if args.newer_than:
        p = Path(args.newer_than).expanduser()
        cutoff = p.stat().st_mtime if p.exists() else datetime.fromisoformat(args.newer_than).timestamp()
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
            print(f"# {arm.name}: checkpoint predates the cutoff - skipped ({ckpt})", file=sys.stderr)
            continue
        print(f"{label_for(arm.name)}={ckpt}")
        found += 1
    if not found:
        raise SystemExit(f"no usable arm checkpoints under {root}")


if __name__ == "__main__":
    main()
