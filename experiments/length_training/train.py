#!/usr/bin/env python3
"""
train_wandb.py — COMET finetuning with WandbLogger.

Replicates comet-train but injects WandbLogger directly, working around the
jsonargparse 3.13.1 bug that cannot instantiate Union[Logger, bool, None] from
a class_path/init_args structure in YAML or via --trainer.logger CLI flag.

Usage (fresh run):
    python experiments/length_training/train.py \
        --cfg configs/models/bio_mqm_finetune.yaml \
        --load_from_checkpoint /path/to/base.ckpt

Usage (resume after job timeout):
    python experiments/length_training/train.py \
        --cfg configs/models/bio_mqm_finetune.yaml \
        --load_from_checkpoint /path/to/best.ckpt \
        --resume_from_checkpoint /path/to/best.ckpt

Set WANDB_RUN_ID to resume the same W&B run (metrics continue on the same chart).
"""
from __future__ import annotations

import json
import logging
import os
import warnings

import torch
from jsonargparse import namespace_to_dict
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer

from comet.cli.train import initialize_model, read_arguments

torch.set_float32_matmul_precision("high")

_log = logging.getLogger(__name__)


def main() -> None:
    parser = read_arguments()
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Lightning checkpoint to resume from (restores epoch, optimizer, and scheduler state).",
    )
    parser.add_argument(
        "--reset_early_stopping",
        action="store_true",
        default=False,
        help="Reset EarlyStopping wait_count/stopped_epoch in the resume checkpoint before training. "
             "Use when the saved patience/wait_count mismatch would trigger early stopping immediately.",
    )
    cfg = parser.parse_args()
    seed_everything(cfg.seed_everything)

    resume_ckpt: str | None = cfg.resume_from_checkpoint

    # When WANDB_RUN_ID is set, metrics continue on the existing W&B run chart.
    wandb_run_id = os.environ.get("WANDB_RUN_ID", None)
    wandb_logger = WandbLogger(
        project=os.environ.get("WANDB_PROJECT", "comet-bio-mqm"),
        name=os.environ.get("WANDB_RUN_NAME", None),
        save_dir=os.environ.get("WANDB_SAVE_DIR", "./wandb_logs"),
        id=wandb_run_id,
        resume="allow" if wandb_run_id else None,
        log_model=False,
        tags=os.environ.get("WANDB_TAGS", "").split(",") if os.environ.get("WANDB_TAGS") else None,
    )

    checkpoint_callback = ModelCheckpoint(
        **namespace_to_dict(cfg.model_checkpoint.init_args)
    )
    early_stop_callback = EarlyStopping(
        **namespace_to_dict(cfg.early_stopping.init_args)
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    if resume_ckpt and cfg.reset_early_stopping:
        import shutil
        import tempfile
        fd, patched = tempfile.mkstemp(suffix=".ckpt")
        os.close(fd)
        shutil.copy2(resume_ckpt, patched)
        state = torch.load(patched, map_location="cpu", weights_only=False)
        for k in list(state.get("callbacks", {}).keys()):
            if "EarlyStopping" in str(k):
                state["callbacks"][k].update(
                    {"wait_count": 0, "stopped_epoch": 0, "patience": early_stop_callback.patience}
                )
                print(f"Reset EarlyStopping state in checkpoint (patience={early_stop_callback.patience})")
        torch.save(state, patched)
        resume_ckpt = patched

    trainer_args = namespace_to_dict(cfg.trainer.init_args)
    trainer_args["callbacks"] = [early_stop_callback, checkpoint_callback, lr_monitor]
    trainer_args["logger"] = wandb_logger

    print("TRAINER ARGUMENTS:")
    print(json.dumps(
        {k: v for k, v in trainer_args.items() if k != "logger"},
        indent=4, default=lambda x: x.__dict__,
    ))

    trainer = Trainer(**trainer_args)
    model = initialize_model(cfg)

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*Consider increasing the value of the `num_workers` argument.*",
    )
    try:
        trainer.fit(model, ckpt_path=resume_ckpt)
    except KeyError as exc:
        if resume_ckpt and "save_weights_only" in str(exc):
            # Checkpoint was saved with save_weights_only=True — no optimizer state.
            # The Trainer's checkpoint connector is now dirty; a fresh Trainer is
            # required. Training restarts from epoch 0 with the finetuned weights.
            print(
                f"\nWARNING: {exc}\n"
                "Falling back to weights-only resume (epoch counter resets, "
                "optimizer starts fresh). Future checkpoints will include full state.\n"
            )
            fresh_trainer = Trainer(**trainer_args)
            fresh_trainer.fit(model)
        else:
            raise


if __name__ == "__main__":
    main()
