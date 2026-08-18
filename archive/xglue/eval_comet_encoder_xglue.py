#!/usr/bin/env python3
"""
eval_comet_encoder_xglue.py

Extract the encoder from a COMET checkpoint and evaluate its multilingual
sentence representations on XGlue benchmark tasks via a frozen linear probe.

Evaluated tasks
---------------
  xnli   : Cross-lingual NLI — train on English, 0-shot transfer to 15 languages
  paws-x : Paraphrase detection — train on English, 0-shot transfer to 6 languages
  nc     : News Classification — train on English, 0-shot transfer to 4 languages
             (requires --xglue_data_dir or the tar.gz set via --xglue_tar)

Methodology
-----------
Sentence embeddings are obtained via the FULL COMET pipeline:
  model.get_sentence_embedding(input_ids, attention_mask)
which applies learned layerwise attention across all transformer layers and
then average-pools over non-pad/sep tokens — exactly what COMET uses during
scoring. NOT the raw CLS token from encoder.forward().

For sentence pairs (XNLI, PAWS-X) we use InferSent-style features:
  [u ; v ; |u−v|]
For single sentences (NC) we use the raw embedding.

Outputs
-------
  - JSON file (always):   <output>
  - W&B run (optional):   set --wandb_project or $WANDB_PROJECT

Usage
-----
  python scripts/eval_comet_encoder_xglue.py \\
      --checkpoint Unbabel/wmt22-comet-da \\
      --xglue_tar data/xglue_full_dataset.tar.gz \\
      --tasks xnli paws-x nc \\
      --wandb_project comet-xglue
"""
import argparse
import io
import json
import logging
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ── default language lists ────────────────────────────────────────────────────

XNLI_LANGS = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
PAWS_LANGS  = ["de", "en", "es", "fr", "ja", "ko", "zh"]
NC_LANGS    = ["de", "en", "es", "fr", "ru"]


# ── COMET loading ─────────────────────────────────────────────────────────────

def load_comet_model(checkpoint: str, device: str):
    """Load a COMET model and freeze its encoder. Returns the full CometModel.

    Accepts a local .ckpt path or a HuggingFace model id such as
    "Unbabel/wmt22-comet-da".

    We return the FULL model (not just model.encoder) because the meaningful
    sentence embedding requires the layerwise attention + pooling that lives in
    model.compute_sentence_embedding(), not the raw CLS from encoder.forward().
    """
    from comet import download_model, load_from_checkpoint

    ckpt_path = checkpoint
    if not os.path.isfile(checkpoint):
        logger.info(f"Downloading from HuggingFace: {checkpoint}")
        ckpt_path = download_model(checkpoint)

    logger.info(f"Loading COMET model from: {ckpt_path}")
    model = load_from_checkpoint(ckpt_path)
    model.eval()
    model = model.to(device)

    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    pool = getattr(model.hparams, "pool", "?")
    layer = getattr(model.hparams, "layer", "?")
    logger.info(f"Model loaded  ({n_params:.1f}M params, pool={pool}, layer={layer}, device={device})")
    return model


# ── encoding (via the proper COMET pipeline) ──────────────────────────────────

@torch.no_grad()
def encode_texts(
    model,
    texts: List[str],
    batch_size: int,
    device: str,
    desc: str = "encoding",
) -> np.ndarray:
    """Encode texts using model.get_sentence_embedding() — the full COMET
    pipeline (layerwise attention + pooling), NOT the raw encoder CLS token.
    """
    embeddings: List[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"  {desc}", leave=False):
        batch = texts[start : start + batch_size]
        enc_input = model.encoder.prepare_sample(batch)
        input_ids = enc_input["input_ids"].to(device)
        attention_mask = enc_input["attention_mask"].to(device)
        token_type_ids = enc_input.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        # This applies layerwise attention + average pooling — the correct path
        sentemb = model.get_sentence_embedding(input_ids, attention_mask, token_type_ids)
        embeddings.append(sentemb.cpu().float().numpy())
    return np.concatenate(embeddings, axis=0)


def pair_features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """InferSent-style [u ; v ; |u−v|] for sentence pairs."""
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
    model,
    batch_size: int,
    device: str,
    languages: List[str],
    max_train: Optional[int],
    wandb_run=None,
) -> Dict[str, Dict]:
    from datasets import load_dataset

    logger.info("── XNLI ─────────────────────────────────────────────")
    train_ds = load_dataset("xnli", "en", split="train")
    if max_train:
        train_ds = train_ds.select(range(min(max_train, len(train_ds))))

    logger.info(f"  Training set: {len(train_ds)} English examples")
    u_tr = encode_texts(model, list(train_ds["premise"]),    batch_size, device, "premise train")
    v_tr = encode_texts(model, list(train_ds["hypothesis"]), batch_size, device, "hypo  train")
    X_train = pair_features(u_tr, v_tr)
    y_train = np.array(train_ds["label"])

    results: Dict[str, Dict] = {}
    for lang in languages:
        logger.info(f"  [{lang}]")
        try:
            test_ds = load_dataset("xnli", lang, split="test")
        except Exception as exc:
            logger.warning(f"    Skipping — could not load {lang}: {exc}")
            continue
        u_te = encode_texts(model, list(test_ds["premise"]),    batch_size, device, "premise test")
        v_te = encode_texts(model, list(test_ds["hypothesis"]), batch_size, device, "hypo  test")
        X_test = pair_features(u_te, v_te)
        y_test = np.array(test_ds["label"])
        m = linear_probe(X_train, y_train, X_test, y_test)
        results[lang] = m
        logger.info(f"    acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}")
        if wandb_run is not None:
            wandb_run.log({f"xnli/{lang}/accuracy": m["accuracy"], f"xnli/{lang}/f1_macro": m["f1_macro"]})

    if results:
        results["avg"] = _avg_results(results)
        logger.info(f"  [avg] acc={results['avg']['accuracy']:.4f}  f1={results['avg']['f1_macro']:.4f}")
        if wandb_run is not None:
            wandb_run.log({"xnli/avg/accuracy": results["avg"]["accuracy"], "xnli/avg/f1_macro": results["avg"]["f1_macro"]})
    return results


def eval_paws_x(
    model,
    batch_size: int,
    device: str,
    languages: List[str],
    max_train: Optional[int],
    wandb_run=None,
) -> Dict[str, Dict]:
    from datasets import load_dataset

    logger.info("── PAWS-X ────────────────────────────────────────────")
    train_ds = load_dataset("google-research-datasets/paws-x", "en", split="train")
    if max_train:
        train_ds = train_ds.select(range(min(max_train, len(train_ds))))

    logger.info(f"  Training set: {len(train_ds)} English examples")
    u_tr = encode_texts(model, list(train_ds["sentence1"]), batch_size, device, "s1 train")
    v_tr = encode_texts(model, list(train_ds["sentence2"]), batch_size, device, "s2 train")
    X_train = pair_features(u_tr, v_tr)
    y_train = np.array(train_ds["label"])

    results: Dict[str, Dict] = {}
    for lang in languages:
        logger.info(f"  [{lang}]")
        try:
            test_ds = load_dataset("google-research-datasets/paws-x", lang, split="test")
        except Exception as exc:
            logger.warning(f"    Skipping — could not load {lang}: {exc}")
            continue
        u_te = encode_texts(model, list(test_ds["sentence1"]), batch_size, device, "s1 test")
        v_te = encode_texts(model, list(test_ds["sentence2"]), batch_size, device, "s2 test")
        X_test = pair_features(u_te, v_te)
        y_test = np.array(test_ds["label"])
        m = linear_probe(X_train, y_train, X_test, y_test)
        results[lang] = m
        logger.info(f"    acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}")
        if wandb_run is not None:
            wandb_run.log({f"paws-x/{lang}/accuracy": m["accuracy"], f"paws-x/{lang}/f1_macro": m["f1_macro"]})

    if results:
        results["avg"] = _avg_results(results)
        logger.info(f"  [avg] acc={results['avg']['accuracy']:.4f}  f1={results['avg']['f1_macro']:.4f}")
        if wandb_run is not None:
            wandb_run.log({"paws-x/avg/accuracy": results["avg"]["accuracy"], "paws-x/avg/f1_macro": results["avg"]["f1_macro"]})
    return results


def _load_nc_split(tar: tarfile.TarFile, lang: str, split: str):
    """Load NC data from a tarfile. Returns (texts, labels, label2id)."""
    member_name = f"xglue_full_dataset/NC/xglue.nc.{lang}.{split}"
    try:
        f = tar.extractfile(member_name)
    except KeyError:
        return None, None, None
    if f is None:
        return None, None, None

    texts, raw_labels = [], []
    for line in io.TextIOWrapper(f, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        title, body, category = parts[0], parts[1], parts[2]
        texts.append(title + " " + body)
        raw_labels.append(category)
    return texts, raw_labels


def eval_nc(
    model,
    batch_size: int,
    device: str,
    languages: List[str],
    max_train: Optional[int],
    xglue_tar: Optional[str],
    wandb_run=None,
) -> Dict[str, Dict]:
    logger.info("── NC (News Classification) ─────────────────────────")

    if not xglue_tar or not os.path.isfile(xglue_tar):
        logger.warning(
            "  NC skipped: pass --xglue_tar path/to/xglue_full_dataset.tar.gz"
        )
        return {}

    with tarfile.open(xglue_tar, "r:gz") as tar:
        train_texts, train_raw = _load_nc_split(tar, "en", "train")
        if train_texts is None:
            logger.warning("  NC skipped: could not find English training data in the tar.")
            return {}

        # Build a stable label mapping from the training set
        label2id = {lbl: i for i, lbl in enumerate(sorted(set(train_raw)))}
        logger.info(f"  {len(label2id)} categories: {sorted(label2id)}")

        if max_train:
            train_texts = train_texts[:max_train]
            train_raw   = train_raw[:max_train]

        logger.info(f"  Training set: {len(train_texts)} English examples")
        X_train = encode_texts(model, train_texts, batch_size, device, "NC train")
        y_train = np.array([label2id[lbl] for lbl in train_raw])

        results: Dict[str, Dict] = {}
        for lang in languages:
            logger.info(f"  [{lang}]")
            test_texts, test_raw = _load_nc_split(tar, lang, "test")
            if test_texts is None:
                logger.warning(f"    Skipping — no test file for {lang}")
                continue
            # Filter out labels unseen during training
            valid = [(t, lbl) for t, lbl in zip(test_texts, test_raw) if lbl in label2id]
            if not valid:
                logger.warning(f"    Skipping — no valid labels for {lang}")
                continue
            test_texts_filtered, test_raw_filtered = zip(*valid)
            X_test = encode_texts(model, list(test_texts_filtered), batch_size, device, f"NC test {lang}")
            y_test = np.array([label2id[lbl] for lbl in test_raw_filtered])
            m = linear_probe(X_train, y_train, X_test, y_test)
            results[lang] = m
            logger.info(f"    acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}")
            if wandb_run is not None:
                wandb_run.log({f"nc/{lang}/accuracy": m["accuracy"], f"nc/{lang}/f1_macro": m["f1_macro"]})

    if results:
        results["avg"] = _avg_results(results)
        logger.info(f"  [avg] acc={results['avg']['accuracy']:.4f}  f1={results['avg']['f1_macro']:.4f}")
        if wandb_run is not None:
            wandb_run.log({"nc/avg/accuracy": results["avg"]["accuracy"], "nc/avg/f1_macro": results["avg"]["f1_macro"]})
    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_task(task: str, task_res: Dict, out_dir: Path, run_name: str) -> None:
    if not task_res:
        return

    langs = [l for l in task_res if l != "avg"]
    accs  = [task_res[l]["accuracy"] for l in langs]
    f1s   = [task_res[l]["f1_macro"] for l in langs]

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(langs) * 0.55 + 1.5)))
    fig.suptitle(f"{run_name}  —  {task.upper()}", fontsize=13, fontweight="bold")

    for ax, values, label, color in zip(
        axes, [accs, f1s], ["Accuracy", "F1-macro"], ["#4C72B0", "#DD8452"]
    ):
        bars = ax.barh(langs, values, color=color, alpha=0.85)
        if "avg" in task_res:
            key = "accuracy" if label == "Accuracy" else "f1_macro"
            avg_val = task_res["avg"][key]
            ax.axvline(avg_val, color="black", linestyle="--", linewidth=1.2, label=f"avg={avg_val:.3f}")
            ax.legend(fontsize=8)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_xlabel(label)
        ax.set_xlim(0, 1.05)
        ax.set_title(label)
        ax.invert_yaxis()

    plt.tight_layout()
    out_path = out_dir / f"{run_name}_{task}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [plot] {out_path}")


def plot_summary(results: Dict, out_dir: Path) -> None:
    tasks, accs, f1s = [], [], []
    for task, task_res in results["tasks"].items():
        if "avg" in task_res:
            tasks.append(task)
            accs.append(task_res["avg"]["accuracy"])
            f1s.append(task_res["avg"]["f1_macro"])

    if not tasks:
        return

    run_name = results["run_name"]
    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(tasks) * 2.2), 5))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width / 2, f1s,  width, label="F1-macro",  color="#DD8452", alpha=0.85)
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"XGlue summary — {run_name}", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    out_path = out_dir / f"{run_name}_summary.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [plot] {out_path}")


def save_intermediate(results: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    logger.info(f"  [checkpoint] → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Local .ckpt path or HuggingFace model id (e.g. Unbabel/wmt22-comet-da).")
    p.add_argument("--tasks", nargs="+", default=["xnli", "paws-x", "nc"],
                   choices=["xnli", "paws-x", "nc"])
    p.add_argument("--languages", nargs="+", default=None,
                   help="Override default language list (applied to all tasks).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default="results/xglue_eval.json")
    p.add_argument("--xglue_tar", default=None,
                   help="Path to xglue_full_dataset.tar.gz (required for NC task).")
    p.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT"))
    p.add_argument("--run_name", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    model = load_comet_model(args.checkpoint, device)

    run_name = args.run_name or Path(args.checkpoint).stem
    timestamp = datetime.now(timezone.utc).isoformat()

    results: Dict = {
        "run_name": run_name,
        "checkpoint": args.checkpoint,
        "timestamp": timestamp,
        "device": device,
        "batch_size": args.batch_size,
        "max_train_samples": args.max_train_samples,
        "embedding": "layerwise_attention+avg_pool",
        "tasks": {},
    }

    out_path = Path(args.output)
    plot_dir = out_path.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # W&B run (initialise early so we can log per-language as we go)
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "checkpoint": args.checkpoint,
                    "tasks": args.tasks,
                    "batch_size": args.batch_size,
                    "max_train_samples": args.max_train_samples,
                    "embedding": "layerwise_attention+avg_pool",
                },
            )
        except Exception as exc:
            logger.warning(f"W&B init failed: {exc}")

    task_fns = {
        "xnli":   lambda: eval_xnli(model, args.batch_size, device,
                                     args.languages or XNLI_LANGS, args.max_train_samples, wandb_run),
        "paws-x": lambda: eval_paws_x(model, args.batch_size, device,
                                       args.languages or PAWS_LANGS, args.max_train_samples, wandb_run),
        "nc":     lambda: eval_nc(model, args.batch_size, device,
                                   args.languages or NC_LANGS, args.max_train_samples,
                                   args.xglue_tar, wandb_run),
    }

    for task in args.tasks:
        task_res = task_fns[task]()
        results["tasks"][task] = task_res
        save_intermediate(results, out_path)
        plot_task(task, task_res, plot_dir, run_name)

    plot_summary(results, plot_dir)

    if wandb_run is not None:
        wandb_run.finish()

    logger.info(f"\nResults saved → {out_path}")
    logger.info(f"Plots saved  → {plot_dir}/")
    logger.info("\n── Summary ──────────────────────────────────────────────")
    for task, task_res in results["tasks"].items():
        if "avg" in task_res:
            m = task_res["avg"]
            logger.info(f"  {task:<10}  acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}")
    logger.info("─────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
