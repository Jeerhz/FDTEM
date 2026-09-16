"""Build the self-contained YAML a COMET training run consumes.

One train file only: COMET picks an epoch's file as `train_data[epoch % len]` and
Lightning rebuilds the loader only when `reload_dataloaders_every_n_epochs >= 1`
(ours is 0), so a multi-file list silently trains on the first file. The
2026-08-14 sweep fell for it; this module refuses it.

    python -m common.train_config --base_cfg part2_length_training/configs/comet_da.yaml \
        --train_files <mix>/all_train.csv --val_files <pools>/all_val.csv --print
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import yaml

from common.paths import ROOT

csv.field_size_limit(10 ** 9)
SUBCONFIG_KEYS = ("trainer", "early_stopping", "model_checkpoint")


def count_rows(path: Path) -> int:
    """Rows of a COMET CSV; native documents embed newlines, so `wc -l` over-counts."""
    with open(path, newline="", encoding="utf-8") as fh:
        return max(0, sum(1 for _ in csv.reader(fh)) - 1)


def load_subconfig(value, base_dir: Path) -> dict:
    """A sub-config is inline (dict) or a path, relative to the repo root or to the base YAML."""
    if isinstance(value, dict):
        return value
    for candidate in (ROOT / str(value), base_dir / os.path.expanduser(str(value))):
        if candidate.exists():
            return yaml.safe_load(candidate.read_text())
    raise SystemExit(f"sub-config {value!r} not found (tried under {ROOT} and {base_dir})")


def build_train_config(base_cfg: Path, train_files: list[str], val_files: list[str], *,
                       frozen: bool = False, encoder_lr: float | None = None,
                       head_lr: float | None = None, nr_frozen_epochs: float | None = None,
                       max_epochs: int | None = None, max_steps: int | None = None,
                       val_check_interval: float | None = None, patience: int | None = None,
                       save_top_k: int | None = None, allow_multi_train: bool = False) -> dict:
    base_cfg = Path(base_cfg).expanduser().resolve()
    cfg = yaml.safe_load(base_cfg.read_text())
    for key in SUBCONFIG_KEYS:
        cfg[key] = load_subconfig(cfg[key], base_cfg.parent)

    metric_key = next(k for k, v in cfg.items()
                      if isinstance(v, dict) and "class_path" in v and k not in SUBCONFIG_KEYS)
    init = cfg[metric_key]["init_args"]

    train_files = [str(Path(p).expanduser().resolve()) for p in train_files]
    if len(train_files) > 1 and not allow_multi_train:
        raise SystemExit(f"train_data has {len(train_files)} files: COMET would read only the "
                         "first one. Pass a single concatenated CSV (the mix's all_train.csv).")
    init["train_data"] = train_files
    init["validation_data"] = [str(Path(p).expanduser().resolve()) for p in val_files]

    if frozen:  # comet unfreezes once epoch_nr >= nr_frozen_epochs: this never fires
        init["nr_frozen_epochs"] = 1000
        init["keep_embeddings_frozen"] = True
    elif nr_frozen_epochs is not None:
        init["nr_frozen_epochs"] = nr_frozen_epochs
    if encoder_lr is not None:
        init["encoder_learning_rate"] = encoder_lr
    if head_lr is not None:
        init["learning_rate"] = head_lr

    trainer = cfg["trainer"]["init_args"]
    if max_epochs is not None:
        trainer["max_epochs"] = max_epochs
    if max_steps is not None:
        trainer["max_steps"] = max_steps
    if val_check_interval is not None:
        trainer["val_check_interval"] = min(1.0, val_check_interval)
    if patience is not None:
        cfg["early_stopping"]["init_args"]["patience"] = patience

    # Checkpoint on the validation cadence: "best by val_kendall" is what evaluation selects.
    ckpt = cfg["model_checkpoint"]["init_args"]
    ckpt.update(every_n_train_steps=None, every_n_epochs=1, save_on_train_epoch_end=False)
    if save_top_k is not None:
        ckpt["save_top_k"] = save_top_k
    return cfg


def write_train_config(cfg: dict, out: Path) -> Path:
    out = Path(out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    return out


def summary(cfg: dict) -> str:
    metric = next(v for k, v in cfg.items() if isinstance(v, dict) and "class_path" in v
                  and k not in SUBCONFIG_KEYS)["init_args"]
    trainer = cfg["trainer"]["init_args"]
    n = count_rows(Path(metric["train_data"][0])) if Path(metric["train_data"][0]).exists() else None
    return "\n".join([
        f"train_data      -> {metric['train_data'][0]}" + (f" ({n:,} rows = 1 epoch)" if n else ""),
        f"validation_data -> {len(metric['validation_data'])} file(s)",
        f"encoder         -> nr_frozen_epochs={metric.get('nr_frozen_epochs')} "
        f"lr={metric.get('encoder_learning_rate')} head_lr={metric.get('learning_rate')}",
        f"budget          -> max_epochs={trainer.get('max_epochs')} max_steps={trainer.get('max_steps')} "
        f"patience={cfg['early_stopping']['init_args'].get('patience')}",
    ])


def add_override_args(ap: argparse.ArgumentParser) -> None:
    """The overrides, shared with the trainer CLI."""
    ap.add_argument("--frozen", action="store_true", help="freeze the encoder for the whole run")
    ap.add_argument("--encoder_lr", type=float)
    ap.add_argument("--head_lr", type=float)
    ap.add_argument("--nr_frozen_epochs", type=float)
    ap.add_argument("--max_epochs", type=int)
    ap.add_argument("--max_steps", type=int)
    ap.add_argument("--val_check_interval", type=float)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--save_top_k", type=int)


def overrides(args: argparse.Namespace) -> dict:
    keys = ("frozen", "encoder_lr", "head_lr", "nr_frozen_epochs", "max_epochs", "max_steps",
            "val_check_interval", "patience", "save_top_k")
    return {k: getattr(args, k) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_cfg", required=True)
    ap.add_argument("--train_files", nargs="+", required=True)
    ap.add_argument("--val_files", nargs="+", required=True)
    ap.add_argument("--out", help="write the generated YAML here")
    ap.add_argument("--print", action="store_true", help="print the generated YAML")
    add_override_args(ap)
    args = ap.parse_args()
    cfg = build_train_config(args.base_cfg, args.train_files, args.val_files, **overrides(args))
    if args.out:
        print(f"generated config -> {write_train_config(cfg, args.out)}")
    if args.print or not args.out:
        print(yaml.safe_dump(cfg, sort_keys=False))
    print(summary(cfg))


if __name__ == "__main__":
    main()
