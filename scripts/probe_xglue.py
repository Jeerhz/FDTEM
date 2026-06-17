#!/usr/bin/env python3
"""
probe_xglue.py — FROZEN-ENCODER evaluation on XGlue.

Procedure (representation analysis)
-----------------------------------
The encoder is **frozen**. Every input is encoded once, reduced to a sentence
embedding by a fixed pooling, then a linear classifier (logistic regression on
standardised features) is trained on the English training split and evaluated
zero-shot on every other language.

This measures what is *linearly accessible* in the representations as they are —
so comparing a plain XLM-R encoder against the COMET encoder tells you how
COMET's translation-quality training reshaped the representation space, with the
encoder itself left untouched.

The companion script finetune_xglue.py runs the same tasks but lets the encoder
adapt (the published XGlue protocol). Run both to see the gap.

Both scripts share xglue_common.py, so HF and COMET encoders go through the
identical loader / tokenizer / pooling — the only difference is the weights.

Usage
-----
  # XLM-R large, frozen probe, all tasks
  python scripts/probe_xglue.py \\
      --encoder hf:xlm-roberta-large \\
      --tasks xnli paws-x nc \\
      --xglue_tar data/xglue_full_dataset.tar.gz \\
      --output results/probe_xlmr_large.json \\
      --wandb_project comet-xglue

  # COMET encoder, identical frozen probe
  python scripts/probe_xglue.py \\
      --encoder comet:Unbabel/wmt22-comet-da \\
      --tasks xnli paws-x nc \\
      --xglue_tar data/xglue_full_dataset.tar.gz \\
      --output results/probe_comet.json \\
      --wandb_project comet-xglue
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from xglue_common import (
    TASK_INFO,
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


# ── frozen encoding ─────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_examples(
    encoder,
    examples: List[dict],
    collate,
    pool_mode: str,
    batch_size: int,
    device: str,
    autocast_dtype,
    desc: str,
) -> np.ndarray:
    """Encode a list of examples into a (N, hidden) feature matrix with the frozen
    encoder + fixed pooling. Labels are read separately by the caller."""
    encoder.eval()
    loader = DataLoader(examples, batch_size=batch_size, shuffle=False, collate_fn=collate)
    feats: List[np.ndarray] = []
    for input_ids, attention_mask, _ in tqdm(loader, desc=f"  {desc}", leave=False):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with amp_context(device, autocast_dtype):
            hidden = encoder(input_ids, attention_mask)
            sent = pool(hidden, attention_mask, pool_mode)
        feats.append(sent.cpu().float().numpy())
    return np.concatenate(feats, axis=0)


def labels_of(examples: List[dict]) -> np.ndarray:
    return np.array([ex["label"] for ex in examples], dtype=np.int64)


def fit_probe(X_train, y_train, max_iter: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=max_iter, n_jobs=-1),
    )
    clf.fit(X_train, y_train)
    return clf


# ── per-task evaluation ─────────────────────────────────────────────────────────

def run_task(
    task: str,
    encoder,
    tokenizer,
    args,
    device: str,
    autocast_dtype,
    wandb_run=None,
) -> Dict[str, Dict[str, float]]:
    info = TASK_INFO[task]
    max_len = args.max_len or info["max_len"]
    languages = args.languages or info["langs"]

    logger.info(f"── {task.upper()} (frozen probe) ──────────────────────────")
    train_ex, test_by_lang, _ = build_data(task, languages, args.max_train_samples, args.xglue_tar)
    logger.info(f"  Train: {len(train_ex)} (en)   Test langs: {list(test_by_lang)}")

    collate = make_collate(tokenizer, max_len)

    X_train = encode_examples(encoder, train_ex, collate, args.pool,
                              args.batch_size, device, autocast_dtype, "encode train")
    y_train = labels_of(train_ex)

    logger.info("  Fitting linear probe …")
    clf = fit_probe(X_train, y_train, args.probe_max_iter)

    per_lang: Dict[str, Dict[str, float]] = {}
    for lang, examples in test_by_lang.items():
        X_test = encode_examples(encoder, examples, collate, args.pool,
                                 args.batch_size, device, autocast_dtype, f"encode {lang}")
        y_test = labels_of(examples)
        preds = clf.predict(X_test)
        m = compute_metrics(y_test, preds)
        per_lang[lang] = m
        logger.info(f"  [{lang}] acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}")
        if wandb_run is not None:
            wandb_run.log({f"{task}/{lang}/accuracy": m["accuracy"],
                           f"{task}/{lang}/f1_macro": m["f1_macro"]})

    if per_lang:
        per_lang["avg"] = average_over_langs(per_lang)
        avg = per_lang["avg"]
        logger.info(f"  [avg] acc={avg['accuracy']:.4f}  f1={avg['f1_macro']:.4f}")
        if wandb_run is not None:
            wandb_run.log({f"{task}/avg/accuracy": avg["accuracy"],
                           f"{task}/avg/f1_macro": avg["f1_macro"]})
    return per_lang


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", required=True, help="hf:<model_id>  or  comet:<id-or-ckpt>")
    p.add_argument("--tasks", nargs="+", default=["xnli", "paws-x", "nc"],
                   choices=list(TASK_INFO.keys()))
    p.add_argument("--languages", nargs="+", default=None, help="Override per-task default langs.")
    p.add_argument("--xglue_tar", default=None, help="Path to xglue_full_dataset.tar.gz (required for nc).")

    p.add_argument("--pool", choices=["mean", "cls"], default="mean",
                   help="Sentence pooling (default mean — best for a frozen encoder).")
    p.add_argument("--max_len", type=int, default=None, help="Override per-task default.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Cap English training examples (probe fit can be slow on the full 392k XNLI).")
    p.add_argument("--probe_max_iter", type=int, default=1000)

    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--device", default=None)
    p.add_argument("--output", default="results/probe_xglue.json")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--run_name", default=None)
    return p.parse_args()


def main() -> None:
    import os
    args = parse_args()
    if args.wandb_project is None:
        args.wandb_project = os.environ.get("WANDB_PROJECT")

    device, autocast_dtype = resolve_device_and_dtype(args.device, args.precision)
    logger.info(f"Device: {device}  precision: {autocast_dtype}")

    encoder, tokenizer, hidden, tag = load_encoder(args.encoder)
    encoder = encoder.to(device)
    for ppar in encoder.parameters():
        ppar.requires_grad_(False)

    run_name = args.run_name or f"{tag}_probe"

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project, name=run_name,
                config={"encoder": args.encoder, "tasks": args.tasks, "pool": args.pool,
                        "procedure": "frozen_probe", "max_train_samples": args.max_train_samples},
            )
        except Exception as exc:
            logger.warning(f"W&B init failed: {exc}")

    summary: Dict = {
        "run_name": run_name,
        "encoder": args.encoder,
        "procedure": "frozen_probe",
        "pool": args.pool,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "max_train_samples": args.max_train_samples,
        "tasks": {},
    }

    out_path = Path(args.output)
    plot_dir = out_path.parent / "plots"

    for task in args.tasks:
        per_lang = run_task(task, encoder, tokenizer, args, device, autocast_dtype, wandb_run)
        summary["tasks"][task] = per_lang
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        plot_task(task, per_lang, plot_dir, run_name, subtitle="frozen probe")

    if wandb_run is not None:
        wandb_run.finish()

    logger.info(f"\nResults saved → {out_path}")
    logger.info("\n── Summary (frozen probe) ───────────────────────────────")
    for task, per_lang in summary["tasks"].items():
        if "avg" in per_lang:
            m = per_lang["avg"]
            logger.info(f"  {task:<10} acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}")
    logger.info("─────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
