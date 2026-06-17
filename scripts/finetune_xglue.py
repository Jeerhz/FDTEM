#!/usr/bin/env python3
"""
finetune_xglue.py — FULL FINE-TUNING evaluation on XGlue (the published protocol).

Procedure
---------
A classification head is placed on top of the encoder and the **whole model**
(encoder + head) is trained end-to-end on the English training split, then
evaluated zero-shot on every other language. This is the XGLUE / XTREME recipe
and reproduces the published XLM-R baselines (e.g. XNLI ~79% for xlm-roberta-large).

This is the upper-bound counterpart to probe_xglue.py: it measures how much task
signal the encoder can express once it is allowed to adapt. Comparing the two
procedures — and XLM-R vs the COMET encoder under each — is how you characterise
the representation change induced by COMET's translation-quality training.

One task per invocation (fine-tuning trains a separate model per task).

Both scripts share xglue_common.py so HF and COMET encoders are handled identically.

Usage
-----
  # Reproduce XLM-R large on XNLI
  python scripts/finetune_xglue.py \\
      --encoder hf:xlm-roberta-large --task xnli \\
      --output results/ft_xlmr_large_xnli.json \\
      --wandb_project comet-xglue

  # Same protocol on the COMET encoder
  python scripts/finetune_xglue.py \\
      --encoder comet:Unbabel/wmt22-comet-da --task xnli \\
      --output results/ft_comet_xnli.json \\
      --wandb_project comet-xglue

  # News classification (needs the tar)
  python scripts/finetune_xglue.py \\
      --encoder hf:xlm-roberta-large --task nc \\
      --xglue_tar data/xglue_full_dataset.tar.gz \\
      --output results/ft_xlmr_large_nc.json
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from xglue_common import (
    TASK_INFO,
    EncoderAdapter,
    amp_context,
    average_over_langs,
    build_data,
    compute_metrics,
    load_encoder,
    make_collate,
    plot_task,
    pool,
    resolve_device_and_dtype,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ── classification model (identical head for every encoder) ────────────────────

class SequenceClassifier(nn.Module):
    """Encoder + RoBERTa-style classification head, mirroring transformers'
    RobertaClassificationHead so results match standard fine-tuning baselines."""

    def __init__(self, encoder: EncoderAdapter, hidden_size: int, num_labels: int,
                 pool_mode: str = "cls", dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.pool_mode = pool_mode
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids, attention_mask)
        x = pool(hidden, attention_mask, self.pool_mode)
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


# ── optimizer ───────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay, nodecay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (nodecay if any(nd in name for nd in no_decay) else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


# ── eval ─────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype) -> Dict[str, float]:
    model.eval()
    preds_all, labels_all = [], []
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with amp_context(device, autocast_dtype):
            logits = model(input_ids, attention_mask)
        preds_all.append(logits.argmax(dim=-1).cpu().numpy())
        labels_all.append(labels.numpy())
    return compute_metrics(np.concatenate(labels_all), np.concatenate(preds_all))


# ── train + zero-shot eval ───────────────────────────────────────────────────────

def train_and_eval(args) -> Dict:
    device, autocast_dtype = resolve_device_and_dtype(args.device, args.precision)
    logger.info(f"Device: {device}  precision: {autocast_dtype}")

    info = TASK_INFO[args.task]
    max_len = args.max_len or info["max_len"]
    epochs = args.epochs or info["epochs"]
    languages = args.languages or info["langs"]
    num_labels = info["num_labels"]

    logger.info(f"Loading data for task: {args.task}")
    train_ex, test_by_lang, label_names = build_data(
        args.task, languages, args.max_train_samples, args.xglue_tar
    )
    logger.info(f"  Train: {len(train_ex)} (en)   Test langs: {list(test_by_lang)}")

    encoder, tokenizer, hidden, tag = load_encoder(args.encoder)
    model = SequenceClassifier(encoder, hidden, num_labels,
                               pool_mode=args.pool, dropout=args.dropout).to(device)
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        logger.info("  Encoder FROZEN (head-only training — use probe_xglue.py for the standard probe)")

    collate = make_collate(tokenizer, max_len)
    train_loader = DataLoader(
        train_ex, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.num_workers, drop_last=True,
    )

    from transformers import get_linear_schedule_with_warmup
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = max(1, steps_per_epoch * epochs)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=autocast_dtype == torch.float16)
    loss_fn = nn.CrossEntropyLoss()

    run_name = args.run_name or f"{tag}_{args.task}_ft"
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project, name=run_name,
                config={
                    "encoder": args.encoder, "task": args.task, "max_len": max_len,
                    "epochs": epochs, "batch_size": args.batch_size, "lr": args.lr,
                    "grad_accum": args.grad_accum, "warmup_ratio": args.warmup_ratio,
                    "weight_decay": args.weight_decay, "pool": args.pool,
                    "freeze_encoder": args.freeze_encoder, "procedure": "finetune",
                },
            )
        except Exception as exc:
            logger.warning(f"W&B init failed: {exc}")

    logger.info(f"Training {epochs} epoch(s) — {total_steps} optimizer steps")
    global_step = 0
    per_lang: Dict[str, Dict[str, float]] = {}
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}")
        for i, (input_ids, attention_mask, labels) in enumerate(pbar):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            with amp_context(device, autocast_dtype):
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels) / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * args.grad_accum

            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.log_every == 0:
                    avg_loss = running / (args.log_every * args.grad_accum)
                    pbar.set_postfix(loss=f"{avg_loss:.4f}")
                    if wandb_run is not None:
                        wandb_run.log({"train/loss": avg_loss,
                                       "train/lr": scheduler.get_last_lr()[0]}, step=global_step)
                    running = 0.0

        logger.info(f"── eval after epoch {epoch+1} ──")
        per_lang = {}
        for lang, examples in test_by_lang.items():
            loader = DataLoader(examples, batch_size=args.eval_batch_size, shuffle=False,
                                collate_fn=collate, num_workers=args.num_workers)
            m = evaluate(model, loader, device, autocast_dtype)
            per_lang[lang] = m
            logger.info(f"  [{lang}] acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}")
            if wandb_run is not None:
                wandb_run.log({f"{args.task}/{lang}/accuracy": m["accuracy"],
                               f"{args.task}/{lang}/f1_macro": m["f1_macro"]}, step=global_step)
        per_lang["avg"] = average_over_langs(per_lang)
        avg = per_lang["avg"]
        logger.info(f"  [avg] acc={avg['accuracy']:.4f}  f1={avg['f1_macro']:.4f}")
        if wandb_run is not None:
            wandb_run.log({f"{args.task}/avg/accuracy": avg["accuracy"],
                           f"{args.task}/avg/f1_macro": avg["f1_macro"]}, step=global_step)

    if wandb_run is not None:
        wandb_run.finish()

    return {
        "run_name": run_name,
        "encoder": args.encoder,
        "task": args.task,
        "procedure": "finetune" if not args.freeze_encoder else "head_only",
        "pool": args.pool,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "config": {
            "max_len": max_len, "epochs": epochs, "batch_size": args.batch_size,
            "lr": args.lr, "grad_accum": args.grad_accum, "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay, "max_train_samples": args.max_train_samples,
        },
        "label_names": label_names,
        "results": per_lang,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    import os
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", required=True, help="hf:<model_id>  or  comet:<id-or-ckpt>")
    p.add_argument("--task", required=True, choices=list(TASK_INFO.keys()))
    p.add_argument("--languages", nargs="+", default=None, help="Override per-task default langs.")
    p.add_argument("--xglue_tar", default=None, help="Path to xglue_full_dataset.tar.gz (required for nc).")

    # training hyperparameters (defaults follow the XGLUE/XTREME recipe)
    p.add_argument("--pool", choices=["cls", "mean"], default="cls",
                   help="Sentence pooling for the head (default cls — standard for fine-tuning).")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=None, help="Override per-task default.")
    p.add_argument("--max_len", type=int, default=None, help="Override per-task default.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Head-only training (encoder frozen). For the standard probe use probe_xglue.py.")

    # infra
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=None, help="Output JSON (default results/<run_name>.json).")
    p.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT"))
    p.add_argument("--run_name", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_and_eval(args)

    out_path = Path(args.output) if args.output else Path("results") / f"{summary['run_name']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved → {out_path}")

    plot_task(summary["task"], summary["results"], out_path.parent / "plots",
              summary["run_name"], subtitle="fine-tuned")

    avg = summary["results"].get("avg")
    if avg:
        logger.info("\n── Summary (fine-tuned) ─────────────────────────────────")
        logger.info(f"  {summary['task']:<10} acc={avg['accuracy']:.4f}  f1={avg['f1_macro']:.4f}")
        logger.info("─────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
