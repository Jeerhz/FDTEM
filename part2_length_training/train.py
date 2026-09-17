"""Train one arm: (family, mix, frozen, preset) -> checkpoints under CHECKPOINTS/<ckpt_root>/<dir_name>.

  python -m part2_length_training.train --family da --mix frac040 [--frozen] [--preset default|wave1|uncontrolled]
      [--resume auto|<ckpt>] [--data_dir ~/scratch/wmt_length_data_v2] [--mix_dir <data_dir>/mixes_pure]

`--resume auto` continues this arm's newest last.ckpt on the same W&B run; a chain of jobs
all carrying the same preset converges on the preset's max_epochs overall.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from common.paths import CHECKPOINTS, SCRATCH
from common.train_comet import train
from part2_length_training import PART_DIR
from part2_length_training.arms import FAMILIES, MIX_SPECS, TRAIN_PRESETS
from part2_length_training.models import ArmLabel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(FAMILIES), required=True)
    ap.add_argument("--mix", choices=sorted(MIX_SPECS), required=True)
    ap.add_argument("--frozen", action="store_true", help="encoder frozen for the whole run")
    ap.add_argument("--preset", choices=sorted(TRAIN_PRESETS), default="default")
    ap.add_argument("--data_dir", default=str(SCRATCH / "wmt_length_data_v2"))
    ap.add_argument("--mix_dir", default=None, help="default: <data_dir>/mixes_pure")
    ap.add_argument("--ckpt_root", default="retrain-wmt-v3")
    ap.add_argument("--resume", default=None, help="'auto' (this arm's newest last.ckpt) or a .ckpt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb_project", default="comet-retrain-wmt")
    ap.add_argument("--run_name", default=None, help="default: <dir_name>-<YYYYmmdd-HHMM>")
    args = ap.parse_args()

    arm = ArmLabel(family=args.family, mix=args.mix, frozen=args.frozen)
    preset = TRAIN_PRESETS[args.preset]
    data_dir = Path(args.data_dir).expanduser()
    mix_dir = (Path(args.mix_dir).expanduser() if args.mix_dir else data_dir / "mixes_pure") / args.mix
    ckpt_dir = CHECKPOINTS / args.ckpt_root / arm.dir_name

    train_file = mix_dir / "all_train.csv"
    if not train_file.exists():
        raise SystemExit(f"missing {train_file} - build it with\n  python -m part2_length_training.make_mixtures "
                         f"--data_dir {data_dir} --out_dir {mix_dir.parent} --total_policy pure --arms {args.mix}")
    tags = ["retrain-wmt", arm.dir_name, f"budget{preset.max_epochs}ep"]
    if (mix_dir / "CONTAMINATED").exists():
        print(f"!!!! CONTAMINATED MIX - {mix_dir} !!!!\n{(mix_dir / 'CONTAMINATED').read_text()}"
              "!!!! correlation results from this arm measure memorisation !!!!")
        tags.append("contaminated")
    val_file = mix_dir / "all_val.csv" if preset.val_files_from_mix else data_dir / "all_val.csv"

    print(f"arm {arm.label} ({arm.dir_name}) preset={args.preset} base={FAMILIES[args.family].hub_id}")
    train(PART_DIR / FAMILIES[args.family].base_cfg, [str(train_file)], [str(val_file)],
          FAMILIES[args.family].hub_id, ckpt_dir,
          run_name=args.run_name or f"{arm.dir_name}-{datetime.now():%Y%m%d-%H%M}",
          wandb_project=args.wandb_project, tags=tags, resume=args.resume, seed=args.seed,
          frozen=args.frozen, max_epochs=preset.max_epochs, patience=preset.patience,
          encoder_lr=preset.encoder_lr, head_lr=preset.head_lr,
          nr_frozen_epochs=preset.nr_frozen_epochs, save_top_k=1)


if __name__ == "__main__":
    main()
