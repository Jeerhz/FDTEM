"""Train a COMET metric with a W&B logger — the trainer shared by parts 2 and 3.

`comet-train` cannot attach a WandbLogger (jsonargparse 3.13.1 cannot build
Union[Logger, bool, None] from YAML), so the Trainer is assembled here from the
generated config. Everything a training job needs around it lives here too, once:
the base checkpoint, resuming a chained job, keeping superseded runs out of the
way, and the hparams.yaml that evaluation needs.

    python -m common.train_comet --base_cfg <base.yaml> --train_files <mix>/all_train.csv \
        --val_files <pools>/all_val.csv --base_model Unbabel/wmt22-comet-da \
        --ckpt_dir ~/scratch/checkpoints/<root>/<arm> --run_name <name> --wandb_project <project> \
        [--resume auto|<ckpt>] [--max_epochs 60 --patience 10] [--frozen] ...

`--resume auto` picks the arm's newest last.ckpt and continues the same W&B run,
which is what lets a long budget survive the wall clock as a chain of jobs.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from common.auth import wandb_logger
from common.comet_models import resolve_checkpoint, write_missing_hparams
from common.train_config import add_override_args, build_train_config, overrides, summary, write_train_config

logger = logging.getLogger(__name__)


def find_last_checkpoint(ckpt_dir: Path) -> Path | None:
    """Newest `<ckpt_dir>/<project>/<run-id>/checkpoints/last.ckpt`, if any."""
    found = list(Path(ckpt_dir).glob("*/*/checkpoints/last.ckpt"))
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def move_previous_runs_aside(ckpt_dir: Path) -> Path | None:
    """A fresh run must not leave older runs where checkpoint selection could pick them."""
    ckpt_dir = Path(ckpt_dir)
    if not any(ckpt_dir.glob("*/*/checkpoints")):
        return None
    aside = ckpt_dir.with_name(f"{ckpt_dir.name}.superseded-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.move(str(ckpt_dir), str(aside))
    ckpt_dir.mkdir(parents=True)
    return aside


def _reset_early_stopping(ckpt: str, patience: int) -> str:
    """Patch a resume checkpoint whose saved patience would stop training immediately."""
    import torch

    fd, patched = tempfile.mkstemp(suffix=".ckpt")
    os.close(fd)
    shutil.copy2(ckpt, patched)
    state = torch.load(patched, map_location="cpu", weights_only=False)
    for key in list(state.get("callbacks", {})):
        if "EarlyStopping" in str(key):
            state["callbacks"][key].update(wait_count=0, stopped_epoch=0, patience=patience)
    torch.save(state, patched)
    return patched


def fit(cfg_path: Path, load_from: str, *, resume_from: str | None = None, seed: int = 42,
        reset_early_stopping: bool = False, wandb_project: str, run_name: str, save_dir: Path,
        tags: list[str] | None = None, run_id: str | None = None) -> None:
    import torch
    from comet.cli.train import initialize_model, read_arguments
    from jsonargparse import namespace_to_dict
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.trainer.trainer import Trainer

    torch.set_float32_matmul_precision("high")
    cfg = read_arguments().parse_args(
        ["--cfg", str(cfg_path), "--load_from_checkpoint", load_from, "--seed_everything", str(seed)])
    seed_everything(seed)

    early_stop = EarlyStopping(**namespace_to_dict(cfg.early_stopping.init_args))
    checkpoint = ModelCheckpoint(**namespace_to_dict(cfg.model_checkpoint.init_args))
    if resume_from and reset_early_stopping:
        resume_from = _reset_early_stopping(resume_from, early_stop.patience)

    trainer_args = namespace_to_dict(cfg.trainer.init_args)
    trainer_args["callbacks"] = [early_stop, checkpoint, LearningRateMonitor(logging_interval="step")]
    trainer_args["logger"] = wandb_logger(wandb_project, run_name, save_dir, tags, run_id)

    model = initialize_model(cfg)
    try:
        Trainer(**trainer_args).fit(model, ckpt_path=resume_from)
    except KeyError as exc:
        if not (resume_from and "save_weights_only" in str(exc)):
            raise
        # weights-only checkpoint: no optimizer state to restore, restart from epoch 0
        logger.warning("%s — resuming weights only, optimizer starts fresh", exc)
        Trainer(**trainer_args).fit(model)


def train(base_cfg: Path, train_files: list[str], val_files: list[str], base_model: str,
          ckpt_dir: Path, run_name: str, wandb_project: str, tags: list[str] | None = None,
          resume: str | None = None, keep_previous_runs: bool = False,
          reset_early_stopping: bool = False, seed: int = 42, **config_overrides) -> Path:
    """One training run: config, base or resume checkpoint, fit, hparams repair.

    `resume`: None (fresh run), "auto" (this arm's newest last.ckpt) or a .ckpt path.
    `config_overrides` are `build_train_config`'s keyword arguments. Returns `ckpt_dir`.
    """
    ckpt_dir = Path(ckpt_dir).expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_id, resume_from = None, None
    if resume == "auto":
        resume_from = find_last_checkpoint(ckpt_dir)
        if resume_from is None:
            print("resume=auto: no checkpoint yet, starting fresh")
    elif resume:
        resume_from = Path(resume).expanduser()
    if resume_from:
        run_id = resume_from.parents[1].name  # <project>/<run-id>/checkpoints/last.ckpt
        name_file = ckpt_dir / "run_name"
        run_name = name_file.read_text().strip() if name_file.is_file() else run_name
        print(f"resuming {resume_from} (W&B run {run_id}, name {run_name})")
    elif not keep_previous_runs:
        aside = move_previous_runs_aside(ckpt_dir)
        if aside:
            print(f"moved previous runs aside: {aside}")
    (ckpt_dir / "run_name").write_text(run_name + "\n")

    cfg = build_train_config(base_cfg, train_files, val_files, **config_overrides)
    cfg_path = write_train_config(cfg, ckpt_dir / "config.yaml")
    print(summary(cfg))

    load_from = str(resume_from) if resume_from else resolve_checkpoint(base_model)
    fit(cfg_path, load_from, resume_from=str(resume_from) if resume_from else None, seed=seed,
        reset_early_stopping=reset_early_stopping, wandb_project=wandb_project,
        run_name=run_name, save_dir=ckpt_dir, tags=tags, run_id=run_id)

    for path in write_missing_hparams(ckpt_dir):
        print(f"wrote {path}")
    print(f"done — checkpoints under {ckpt_dir}")
    return ckpt_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_cfg", required=True)
    ap.add_argument("--train_files", nargs="+", required=True)
    ap.add_argument("--val_files", nargs="+", required=True)
    ap.add_argument("--base_model", required=True, help="Hub id or .ckpt to start from")
    ap.add_argument("--ckpt_dir", required=True, help="the arm's directory; runs nest inside it")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--wandb_project", required=True)
    ap.add_argument("--tags", default="", help="comma-separated W&B tags")
    ap.add_argument("--resume", default=None, help="'auto' (newest last.ckpt of this arm) or a .ckpt")
    ap.add_argument("--keep_previous_runs", action="store_true")
    ap.add_argument("--reset_early_stopping", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    add_override_args(ap)
    args = ap.parse_args()
    train(args.base_cfg, args.train_files, args.val_files, args.base_model, args.ckpt_dir,
          args.run_name, args.wandb_project, tags=[t for t in args.tags.split(",") if t],
          resume=args.resume, keep_previous_runs=args.keep_previous_runs,
          reset_early_stopping=args.reset_early_stopping, seed=args.seed, **overrides(args))


if __name__ == "__main__":
    main()
