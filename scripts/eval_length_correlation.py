#!/usr/bin/env python3
"""
eval_length_correlation.py — does a COMET model correlate better with human
quality on LONGER text?

For each model and each (language pair, window size k) cell of the paragraph
validation set built by scripts/prepare_paragraph_data.py, predict scores and
report Pearson / Spearman / Kendall against the gold (MQM-derived) scores.
The headline plot is Kendall τ vs k, one line per model: a model retrained on
paragraph-level data should degrade less (or improve) as k grows.

Usage:
    python scripts/eval_length_correlation.py \
        --models wmt22=Unbabel/wmt22-comet-da \
                 bio=$HOME/scratch/checkpoints/bio_mqm/comet-bio-mqm/4yeqp7cn/checkpoints/last.ckpt \
                 paragraph=$HOME/scratch/checkpoints/retrain/paragraph/.../last.ckpt \
        --data_dir ~/scratch/paragraph_mqm \
        --output results/retrain/length_correlation.json \
        [--wandb_project comet-retrain]

Each --models entry is `label=<hf-id-or-.ckpt-path>`. Predictions are cached in
--cache_dir keyed by (model label, data hash), so re-running with an extra model
only scores the new one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_comet(ref: str, device_count: int):
    from comet import download_model, load_from_checkpoint
    import os
    ckpt = ref if os.path.isfile(os.path.expanduser(ref)) else download_model(ref)
    model = load_from_checkpoint(os.path.expanduser(ckpt))
    return model


def predict_cached(model, label: str, df: pd.DataFrame, cache_dir: Path,
                   batch_size: int, gpus: int) -> np.ndarray:
    h = hashlib.md5("\n".join(df["src"] + "|" + df["mt"] + "|" + df["ref"])
                    .encode("utf-8")).hexdigest()[:10]
    cache = cache_dir / f"{label}__{h}__n{len(df)}.npy"
    if cache.exists():
        return np.load(cache)
    data = df[["src", "mt", "ref"]].to_dict("records")
    out = model.predict(data, batch_size=batch_size, gpus=gpus,
                        progress_bar=True)
    scores = np.asarray(out["scores"], dtype=float)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, scores)
    return scores


def correlations(gold: np.ndarray, pred: np.ndarray) -> dict:
    from scipy.stats import kendalltau, pearsonr, spearmanr
    if len(gold) < 3 or np.std(gold) == 0 or np.std(pred) == 0:
        return {"n": int(len(gold)), "pearson": None, "spearman": None, "kendall": None}
    return {"n": int(len(gold)),
            "pearson": float(pearsonr(gold, pred)[0]),
            "spearman": float(spearmanr(gold, pred)[0]),
            "kendall": float(kendalltau(gold, pred)[0])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="label=<hf-id-or-ckpt> entries.")
    ap.add_argument("--data_dir", default="~/scratch/paragraph_mqm")
    ap.add_argument("--lang_pairs", nargs="+", default=None,
                    help="Default: every *_val.csv in --data_dir.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gpus", type=int, default=None,
                    help="Default: 1 if CUDA is available else 0.")
    ap.add_argument("--cache_dir", default="results/retrain/pred_cache")
    ap.add_argument("--output", default="results/retrain/length_correlation.json")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--run_name", default="length_correlation")
    args = ap.parse_args()

    import torch
    gpus = args.gpus if args.gpus is not None else (1 if torch.cuda.is_available() else 0)

    data_dir = Path(args.data_dir).expanduser()
    if args.lang_pairs:
        val_files = [data_dir / f"{lp}_val.csv" for lp in args.lang_pairs]
    else:
        val_files = sorted(data_dir.glob("*_val.csv"))
        val_files = [f for f in val_files if not f.name.startswith("all_")]
    if not val_files:
        raise SystemExit(f"No *_val.csv found in {data_dir}")

    models = {}
    for entry in args.models:
        label, _, ref = entry.partition("=")
        if not ref:
            raise SystemExit(f"--models entry {entry!r} must be label=<ref>")
        models[label] = ref

    cache_dir = Path(args.cache_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, name=args.run_name,
                                   job_type="length_correlation",
                                   config={"models": models, "data_dir": str(data_dir)})
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"W&B init failed: {exc}")

    results: dict = {"experiment": "length_correlation",
                     "timestamp": datetime.now(timezone.utc).isoformat(),
                     "data_dir": str(data_dir), "models": {}}

    for label, ref in models.items():
        logger.info(f"\n══ {label} ({ref}) ══")
        model = load_comet(ref, gpus)
        per_lp: dict = {}
        for f in val_files:
            lp = f.name.replace("_val.csv", "")
            df = pd.read_csv(f)
            pred = predict_cached(model, label, df, cache_dir, args.batch_size, gpus)
            df = df.assign(pred=pred)
            per_k = {}
            for k, g in df.groupby("k"):
                per_k[str(int(k))] = correlations(g["score"].to_numpy(),
                                                  g["pred"].to_numpy())
            per_lp[lp] = per_k
            logger.info(f"  {lp}: " + "  ".join(
                f"k={k}:τ={v['kendall']:.3f}" for k, v in sorted(per_k.items(), key=lambda x: int(x[0]))
                if v["kendall"] is not None))
        results["models"][label] = per_lp

        # mean over language pairs, per k
        ks = sorted({k for lp in per_lp.values() for k in lp}, key=int)
        mean_by_k = {}
        for k in ks:
            vals = {m: [lp[k][m] for lp in per_lp.values()
                        if k in lp and lp[k][m] is not None]
                    for m in ("pearson", "spearman", "kendall")}
            mean_by_k[k] = {m: (float(np.mean(v)) if v else None) for m, v in vals.items()}
            if wandb_run is not None and mean_by_k[k]["kendall"] is not None:
                wandb_run.log({f"{label}/k{k}/kendall": mean_by_k[k]["kendall"],
                               f"{label}/k{k}/pearson": mean_by_k[k]["pearson"],
                               f"{label}/k{k}/spearman": mean_by_k[k]["spearman"]})
        results["models"][label]["_mean_by_k"] = mean_by_k

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plot_path = _plot(results, out_path.parent / "plots")
    if wandb_run is not None:
        import wandb
        wandb_run.log({"plots/kendall_vs_length": wandb.Image(str(plot_path))})
        wandb_run.finish()
    logger.info(f"\nResults → {out_path}")


def _plot(results: dict, plot_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, per_lp in results["models"].items():
        mean_by_k = per_lp.get("_mean_by_k", {})
        ks = sorted((k for k, v in mean_by_k.items() if v["kendall"] is not None), key=int)
        ax.plot([int(k) for k in ks], [mean_by_k[k]["kendall"] for k in ks],
                marker="o", label=label)
    ax.set_xlabel("window size k (consecutive segments)")
    ax.set_ylabel("Kendall τ vs gold MQM score (mean over LPs)")
    ax.set_title("Segment-level correlation vs text length")
    ax.grid(alpha=0.3)
    ax.legend()
    out = plot_dir / "kendall_vs_length.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    logger.info(f"  [plot] {out}")
    return out


if __name__ == "__main__":
    main()
