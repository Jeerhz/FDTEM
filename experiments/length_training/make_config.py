#!/usr/bin/env python3
"""
make_config.py — build the per-arm training config for the length-composition
sweep, with an epoch that means the same thing in every arm.

Why this script exists
──────────────────────
COMET picks the training file for an epoch itself:

    # comet/models/base.py :: train_dataloader
    data_path = self.hparams.train_data[self.current_epoch % len(self.hparams.train_data)]

and Lightning only calls `train_dataloader()` again when
`reload_dataloaders_every_n_epochs >= 1` (our trainer config sets 0). So a
multi-file `train_data` does NOT mean "train on all of them":

  * with reload disabled, only `train_data[0]` is ever read — the rest of the
    mix is silently dropped;
  * with reload enabled, one "epoch" is one FILE, so arms whose mix covers more
    language pairs get proportionally smaller epochs and hit early-stopping
    patience after far less data.

The 2026-08-14 sweep passed one CSV per language pair, so every arm trained on
a single language pair's slice of its mix (205–6,940 rows depending on the arm)
instead of the 24,000-row mix. Arm ranking tracked slice size, not composition.

The fix: pass the mix's single `all_train.csv`. One epoch = the whole mix =
the same number of examples in all 36 arms, whatever their language coverage.

This script also pins the training budget, so early stopping compares like with
like: the trainer / early-stopping / checkpoint blocks are INLINED into the
generated file (rather than referenced by path) with the budget overrides
applied, which makes `$CKPT_DIR/config.yaml` a complete, self-contained record
of what the arm actually ran.

Usage (normally called by slurm/train.sh):
    python experiments/length_training/make_config.py \
        --base_cfg experiments/length_training/configs/comet_da.yaml \
        --out      $CKPT_DIR/config.yaml \
        --mix_dir  ~/scratch/wmt_length_data/mixes/frac040 \
        --val_files ~/scratch/wmt_length_data/all_val.csv \
        --max_epochs 6 --patience 3
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import yaml

csv.field_size_limit(10 ** 9)

SUBCONFIG_KEYS = ("trainer", "early_stopping", "model_checkpoint")


def count_rows(path: Path) -> int:
    """Row count of a COMET CSV. Native documents embed newlines, so `wc -l`
    over-counts by ~4x — go through the csv reader."""
    with open(path, newline="", encoding="utf-8") as fh:
        return max(0, sum(1 for _ in csv.reader(fh)) - 1)


def load_subconfig(value, base_dir: Path):
    """Sub-configs are given either inline (dict) or as a path relative to the
    base config. Return the inline dict either way."""
    if isinstance(value, dict):
        return value
    path = (base_dir / os.path.expanduser(str(value))).resolve()
    if not path.exists():
        sys.exit(f"ERROR: sub-config {value!r} does not resolve ({path})")
    return yaml.safe_load(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mix_dir", default=None,
                    help="Mix directory; its all_train.csv becomes the single "
                         "train file. Omit to keep the base config's train_data.")
    ap.add_argument("--train_files", nargs="+", default=None,
                    help="Explicit train CSVs (overrides --mix_dir). More than "
                         "one file is refused unless --allow_multi_train.")
    ap.add_argument("--allow_multi_train", action="store_true",
                    help="Permit a multi-file train_data (COMET rotates files "
                         "across epochs — epochs stop being comparable).")
    ap.add_argument("--val_files", nargs="+", default=None)
    ap.add_argument("--frozen", action="store_true",
                    help="Freeze the encoder for the whole run (head-only).")
    # ── training budget (identical across arms) ───────────────────────────────
    ap.add_argument("--max_epochs", type=int, default=None,
                    help="Hard budget in passes over the mix. With one train "
                         "file an epoch is the whole mix, so this is the same "
                         "number of examples for every arm.")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="Hard budget in optimizer steps (-1 = unbounded).")
    ap.add_argument("--val_check_interval", type=float, default=None,
                    help="Fraction of an epoch between validations.")
    ap.add_argument("--patience", type=int, default=None,
                    help="Early-stopping patience, in validation checks.")
    ap.add_argument("--save_top_k", type=int, default=None)
    args = ap.parse_args()

    base_cfg = Path(args.base_cfg).expanduser().resolve()
    base_dir = base_cfg.parent
    cfg = yaml.safe_load(base_cfg.read_text())

    # inline the sub-configs so the generated file stands alone
    for key in SUBCONFIG_KEYS:
        if key in cfg:
            cfg[key] = load_subconfig(cfg[key], base_dir)

    metric_key = next(k for k, v in cfg.items()
                      if isinstance(v, dict) and "class_path" in v
                      and k not in SUBCONFIG_KEYS)
    init = cfg[metric_key]["init_args"]

    # ── train data: ONE file = one epoch = the whole mix ──────────────────────
    if args.train_files:
        train_files = [str(Path(p).expanduser().resolve()) for p in args.train_files]
    elif args.mix_dir:
        mix_dir = Path(args.mix_dir).expanduser().resolve()
        all_train = mix_dir / "all_train.csv"
        if not all_train.exists():
            sys.exit(f"ERROR: {all_train} not found — rebuild the mixes with "
                     f"experiments/length_training/make_mixtures.py")
        train_files = [str(all_train)]
    else:
        train_files = [str(Path(os.path.expanduser(p)).resolve())
                       for p in init.get("train_data", [])]

    if len(train_files) > 1 and not args.allow_multi_train:
        sys.exit(
            "ERROR: train_data has "
            f"{len(train_files)} files. COMET reads ONE file per epoch "
            "(train_data[epoch % len]), and with "
            "reload_dataloaders_every_n_epochs=0 it only ever reads the first "
            "one — so a multi-file list silently trains on a fraction of the "
            "data and makes epochs incomparable across arms. Pass a single "
            "concatenated CSV (e.g. the mix's all_train.csv), or "
            "--allow_multi_train if you really mean it.")

    init["train_data"] = train_files
    n_train = count_rows(Path(train_files[0])) if len(train_files) == 1 else None

    if args.val_files:
        init["validation_data"] = [str(Path(p).expanduser().resolve())
                                   for p in args.val_files]
    else:
        init["validation_data"] = [str(Path(os.path.expanduser(p)).resolve())
                                   for p in init.get("validation_data", [])]

    if args.frozen:
        # comet unfreezes once epoch_nr >= nr_frozen_epochs — this never fires
        init["nr_frozen_epochs"] = 1000
        init["keep_embeddings_frozen"] = True

    # ── budget: same number of examples and the same stopping rule everywhere ─
    tr = cfg["trainer"]["init_args"]
    if args.max_epochs is not None:
        tr["max_epochs"] = args.max_epochs
    if args.max_steps is not None:
        tr["max_steps"] = args.max_steps
    if args.val_check_interval is not None:
        # 1.0 means "once per epoch"; Lightning wants an int for step cadence
        tr["val_check_interval"] = (1.0 if args.val_check_interval >= 1.0
                                    else args.val_check_interval)

    if args.patience is not None:
        cfg["early_stopping"]["init_args"]["patience"] = args.patience

    # Checkpoint on the validation cadence, not every N train steps: with
    # monitor=val_kendall a step cadence ranks checkpoints on a stale metric,
    # and "best by val_kendall" is what the evaluation scripts select.
    mc = cfg["model_checkpoint"]["init_args"]
    mc["every_n_train_steps"] = None
    mc["every_n_epochs"] = 1
    mc["save_on_train_epoch_end"] = False
    if args.save_top_k is not None:
        mc["save_top_k"] = args.save_top_k

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)

    print(f"train_data      → {len(train_files)} file(s)"
          + (f", {n_train:,} rows (= 1 epoch)" if n_train is not None else ""))
    for p in train_files:
        print(f"                  {p}")
    print(f"validation_data → {len(init['validation_data'])} file(s)")
    print(f"encoder         → {'FROZEN for the whole run' if args.frozen else 'unfrozen (base schedule)'}")
    print(f"budget          → max_epochs={tr.get('max_epochs')} "
          f"max_steps={tr.get('max_steps')} "
          f"val_check_interval={tr.get('val_check_interval')} "
          f"patience={cfg['early_stopping']['init_args'].get('patience')}")
    if n_train is not None:
        me = tr.get("max_epochs", -1)
        if isinstance(me, int) and me > 0:
            print(f"                  ≤ {me * n_train:,} training examples")
    print(f"generated config → {out}")


if __name__ == "__main__":
    main()
