#!/usr/bin/env python3
"""
eval_comet_encoder_xglue.py

Extract the encoder from a COMET checkpoint and evaluate its multilingual
sentence representations on XGlue benchmark tasks via a frozen linear probe.

Evaluated tasks (all loaded from HuggingFace datasets "xglue")
--------------------------------------------------------------
  xnli   : Cross-lingual NLI — train on English, 0-shot transfer to 15 languages
  paws-x : Paraphrase detection — train on English, 0-shot transfer to 6 languages
  nc     : Multilingual news classification — per-language train/test

Methodology
-----------
For each task the encoder is frozen. Sentence pairs are represented as
  [u ; v ; |u−v|]  (Conneau et al. 2017 InferSent)
Single sentences use the raw embedding. A logistic-regression linear probe is
trained on the training split and evaluated on the test split.

Outputs
-------
  - JSON file (always):   results/<run_name>.json
  - W&B run (optional):   set --wandb_project or $WANDB_PROJECT

Usage
-----
  python scripts/eval_comet_encoder_xglue.py \\
      --checkpoint path/to/best_model.ckpt \\
      --tasks xnli paws-x nc \\
      --output results/xglue_eval.json

  # Or with a HuggingFace model id (will download automatically):
  python scripts/eval_comet_encoder_xglue.py \\
      --checkpoint Unbabel/wmt22-comet-da \\
      --wandb_project comet-eval
"""
import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ── default language lists ────────────────────────────────────────────────────

XNLI_LANGS  = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
PAWS_LANGS   = ["de", "en", "es", "fr", "ja", "ko", "zh"]
NC_LANGS     = ["de", "en", "es", "fr", "ru"]  # "tr" sometimes absent from xglue nc


# ── COMET loading ─────────────────────────────────────────────────────────────

def load_encoder(checkpoint: str, device: str):
    """Load a COMET model and return its frozen encoder on *device*.

    Accepts a local .ckpt path or a HuggingFace model id such as
    "Unbabel/wmt22-comet-da" (downloaded automatically if not found locally).
    """
    from comet import download_model, load_from_checkpoint

    ckpt_path = checkpoint
    if not os.path.isfile(checkpoint):
        logger.info(f"Checkpoint not found locally — downloading from HuggingFace: {checkpoint}")
        ckpt_path = download_model(checkpoint)

    logger.info(f"Loading COMET model from: {ckpt_path}")
    model = load_from_checkpoint(ckpt_path)
    model.eval()

    encoder = model.encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    logger.info(f"Encoder loaded  ({n_params:.1f}M parameters, device={device})")
    return encoder


# ── encoding ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_texts(
    encoder,
    texts: List[str],
    batch_size: int,
    device: str,
    desc: str = "encoding",
) -> np.ndarray:
    """Encode a list of strings → numpy array of shape (N, d)."""
    embeddings: List[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"  {desc}", leave=False):
        batch = texts[start : start + batch_size]
        enc_input = encoder.prepare_sample(batch)
        enc_input = {k: v.to(device) for k, v in enc_input.items()}
        out = encoder(**enc_input)
        embeddings.append(out["sentemb"].cpu().float().numpy())
    return np.concatenate(embeddings, axis=0)


def pair_features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """[u ; v ; |u−v|] — standard InferSent-style feature for sentence pairs."""
    return np.concatenate([u, v, np.abs(u - v)], axis=1)


# ── linear probe ─────────────────────────────────────────────────────────────

def linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    clf = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_macro": float(f1_score(y_test, preds, average="macro")),
    }


def _avg_results(per_lang: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = next(iter(per_lang.values())).keys()
    return {k: float(np.mean([v[k] for v in per_lang.values()])) for k in keys}


# ── task evaluators ───────────────────────────────────────────────────────────

def eval_xnli(
    encoder,
    batch_size: int,
    device: str,
    languages: List[str],
    max_train: Optional[int],
) -> Dict[str, Dict]:
    from datasets import load_dataset

    logger.info("── XNLI ─────────────────────────────────────────────")
    train_ds = load_dataset("xglue", "xnli", split="train")
    if max_train:
        train_ds = train_ds.select(range(min(max_train, len(train_ds))))

    logger.info(f"  Training set: {len(train_ds)} English examples")
    u_tr = encode_texts(encoder, train_ds["premise"],    batch_size, device, "premise train")
    v_tr = encode_texts(encoder, train_ds["hypothesis"], batch_size, device, "hypo  train")
    X_train = pair_features(u_tr, v_tr)
    y_train = np.array(train_ds["label"])

    results: Dict[str, Dict] = {}
    for lang in languages:
        logger.info(f"  [{lang}]")
        try:
            test_ds = load_dataset("xglue", "xnli", split=f"test.{lang}")
        except Exception as exc:
            logger.warning(f"    Skipping — could not load test.{lang}: {exc}")
            continue
        u_te = encode_texts(encoder, test_ds["premise"],    batch_size, device, "premise test")
        v_te = encode_texts(encoder, test_ds["hypothesis"], batch_size, device, "hypo  test")
        X_test = pair_features(u_te, v_te)
        y_test = np.array(test_ds["label"])
        m = linear_probe(X_train, y_train, X_test, y_test)
        results[lang] = m
        logger.info(f"    acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}")

    if results:
        results["avg"] = _avg_results(results)
        logger.info(f"  [avg] acc={results['avg']['accuracy']:.4f}  f1={results['avg']['f1_macro']:.4f}")
    return results


def eval_paws_x(
    encoder,
    batch_size: int,
    device: str,
    languages: List[str],
    max_train: Optional[int],
) -> Dict[str, Dict]:
    from datasets import load_dataset

    logger.info("── PAWS-X ────────────────────────────────────────────")
    train_ds = load_dataset("xglue", "paws-x", split="train")
    if max_train:
        train_ds = train_ds.select(range(min(max_train, len(train_ds))))

    logger.info(f"  Training set: {len(train_ds)} English examples")
    u_tr = encode_texts(encoder, train_ds["sentence1"], batch_size, device, "s1 train")
    v_tr = encode_texts(encoder, train_ds["sentence2"], batch_size, device, "s2 train")
    X_train = pair_features(u_tr, v_tr)
    y_train = np.array(train_ds["label"])

    results: Dict[str, Dict] = {}
    for lang in languages:
        logger.info(f"  [{lang}]")
        try:
            test_ds = load_dataset("xglue", "paws-x", split=f"test.{lang}")
        except Exception as exc:
            logger.warning(f"    Skipping — could not load test.{lang}: {exc}")
            continue
        u_te = encode_texts(encoder, test_ds["sentence1"], batch_size, device, "s1 test")
        v_te = encode_texts(encoder, test_ds["sentence2"], batch_size, device, "s2 test")
        X_test = pair_features(u_te, v_te)
        y_test = np.array(test_ds["label"])
        m = linear_probe(X_train, y_train, X_test, y_test)
        results[lang] = m
        logger.info(f"    acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}")

    if results:
        results["avg"] = _avg_results(results)
        logger.info(f"  [avg] acc={results['avg']['accuracy']:.4f}  f1={results['avg']['f1_macro']:.4f}")
    return results


def eval_nc(
    encoder,
    batch_size: int,
    device: str,
    languages: List[str],
    max_train: Optional[int],
) -> Dict[str, Dict]:
    from datasets import load_dataset

    logger.info("── NC (News Classification) ─────────────────────────")
    results: Dict[str, Dict] = {}
    for lang in languages:
        logger.info(f"  [{lang}]")
        try:
            train_ds = load_dataset("xglue", "nc", split=f"train.{lang}")
            test_ds  = load_dataset("xglue", "nc", split=f"test.{lang}")
        except Exception as exc:
            logger.warning(f"    Skipping — could not load {lang}: {exc}")
            continue
        if max_train:
            train_ds = train_ds.select(range(min(max_train, len(train_ds))))
        # Title + body gives richer signal; encoder handles truncation internally.
        train_texts = [f"{t} {b}" for t, b in zip(train_ds["news_title"], train_ds["news_body"])]
        test_texts  = [f"{t} {b}" for t, b in zip(test_ds["news_title"],  test_ds["news_body"])]
        X_train = encode_texts(encoder, train_texts, batch_size, device, "train")
        X_test  = encode_texts(encoder, test_texts,  batch_size, device, "test")
        y_train = np.array(train_ds["news_category"])
        y_test  = np.array(test_ds["news_category"])
        m = linear_probe(X_train, y_train, X_test, y_test)
        results[lang] = m
        logger.info(f"    acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}")

    if results:
        results["avg"] = _avg_results(results)
        logger.info(f"  [avg] acc={results['avg']['accuracy']:.4f}  f1={results['avg']['f1_macro']:.4f}")
    return results


TASK_REGISTRY = {
    "xnli":   (eval_xnli,   XNLI_LANGS),
    "paws-x": (eval_paws_x, PAWS_LANGS),
    "nc":     (eval_nc,     NC_LANGS),
}


# ── W&B logging ───────────────────────────────────────────────────────────────

def log_to_wandb(project: str, run_name: str, config: Dict, results: Dict) -> None:
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed — skipping W&B logging.")
        return

    try:
        run = wandb.init(project=project, name=run_name, config=config)
        flat: Dict[str, float] = {}
        for task, task_res in results["tasks"].items():
            for lang, metrics in task_res.items():
                for metric, value in metrics.items():
                    flat[f"{task}/{lang}/{metric}"] = value
        wandb.log(flat)
        run.finish()
        logger.info(f"W&B run: {run.url}")
    except Exception as exc:
        logger.warning(f"W&B logging failed: {exc}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Local .ckpt path or HuggingFace model id (e.g. Unbabel/wmt22-comet-da).",
    )
    p.add_argument(
        "--tasks", nargs="+", default=list(TASK_REGISTRY.keys()),
        choices=list(TASK_REGISTRY.keys()),
        help="XGlue tasks to evaluate. Default: all.",
    )
    p.add_argument(
        "--languages", nargs="+", default=None,
        help="Override default language list (applied to all selected tasks).",
    )
    p.add_argument(
        "--batch_size", type=int, default=64,
        help="Encoding batch size (default: 64).",
    )
    p.add_argument(
        "--max_train_samples", type=int, default=None,
        help="Cap the training set size for the linear probe (useful for quick runs).",
    )
    p.add_argument(
        "--device", default=None,
        help="Force device: cuda / mps / cpu (auto-detected if omitted).",
    )
    p.add_argument(
        "--output", default="results/xglue_eval.json",
        help="Output JSON path (default: results/xglue_eval.json).",
    )
    p.add_argument(
        "--wandb_project", default=os.environ.get("WANDB_PROJECT"),
        help="W&B project name. Skipped if not provided.",
    )
    p.add_argument(
        "--run_name", default=None,
        help="Run name for W&B and the output file stem (defaults to checkpoint basename).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── device ──
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    # ── load encoder ──
    encoder = load_encoder(args.checkpoint, device)

    run_name = args.run_name or Path(args.checkpoint).stem
    timestamp = datetime.now(timezone.utc).isoformat()

    results: Dict = {
        "run_name": run_name,
        "checkpoint": args.checkpoint,
        "timestamp": timestamp,
        "device": device,
        "batch_size": args.batch_size,
        "max_train_samples": args.max_train_samples,
        "tasks": {},
    }

    for task in args.tasks:
        eval_fn, default_langs = TASK_REGISTRY[task]
        langs = args.languages or default_langs
        results["tasks"][task] = eval_fn(
            encoder=encoder,
            batch_size=args.batch_size,
            device=device,
            languages=langs,
            max_train=args.max_train_samples,
        )

    # ── save JSON ──
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved → {out_path}")

    # ── W&B ──
    if args.wandb_project:
        wandb_config = {
            "checkpoint": args.checkpoint,
            "tasks": args.tasks,
            "batch_size": args.batch_size,
            "max_train_samples": args.max_train_samples,
        }
        log_to_wandb(args.wandb_project, run_name, wandb_config, results)

    # ── summary table ──
    logger.info("\n── Summary ──────────────────────────────────────────────")
    for task, task_res in results["tasks"].items():
        if "avg" in task_res:
            m = task_res["avg"]
            logger.info(f"  {task:<10}  acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}")
    logger.info("─────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
