"""The published base checkpoint and base config of a metric family (the models to fine-tune).

  python -m part2_length_training.load_model --family da|qe
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common.comet_models import resolve_checkpoint
from part2_length_training import PART_DIR
from part2_length_training.arms import FAMILIES


def base_for(family: str) -> tuple[str, Path]:
    """-> (checkpoint path, downloaded once from the Hub; base config path)."""
    fam = FAMILIES[family]
    return resolve_checkpoint(fam.hub_id), PART_DIR / fam.base_cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(FAMILIES), required=True)
    args = ap.parse_args()
    ckpt, cfg = base_for(args.family)
    print(f"hub id     : {FAMILIES[args.family].hub_id}")
    print(f"checkpoint : {ckpt}")
    print(f"base config: {cfg}")


if __name__ == "__main__":
    main()
