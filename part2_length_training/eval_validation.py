#!/usr/bin/env python3
"""Agreement with human judgement, per input length.

Scores every `*_val.csv` in --data_dir with each model and reports Kendall /
Spearman / Pearson per (file, k), plus the mean over files for each k.
k=1 is a single sentence, k=2..6 an aggregated window, k=0 a native document.

  python experiments/length_training/eval_correlation.py \
      --models base=Unbabel/wmt22-comet-da frac040=<ckpt> \
      --data_dir ~/scratch/wmt_eval_portion \
      --output results/length_training/correlation.json
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fdtem.comet_io import load_comet, score
from fdtem.stats import correlations

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_models(entries: list[str]) -> dict[str, str]:
    models = {}
    for e in entries:
        label, _, ref = e.partition("=")
        if not ref:
            raise SystemExit(f"--models entry {e!r} must be label=<ckpt-or-hub-id>")
        models[label] = ref
    return models


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="label=<ckpt-or-hub-id>")
    ap.add_argument("--data_dir", default="~/scratch/wmt_eval_portion")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--cache_dir", default="results/length_training/pred_cache")
    ap.add_argument("--output", default="results/length_training/correlation.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    files = [f for f in sorted(data_dir.glob("*_val.csv")) if not f.name.startswith("all_")]
    if not files:
        raise SystemExit(f"no *_val.csv in {data_dir}")
    models = parse_models(args.models)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"experiment": "length_correlation",
               "timestamp": datetime.now(timezone.utc).isoformat(),
               "data_dir": str(data_dir), "models": {}}

    for label, ref in models.items():
        logger.info(f"\n══ {label} ══")
        model = load_comet(ref)
        per_file = {}
        for f in files:
            df = pd.read_csv(f)
            df = df.assign(pred=score(model, label, df, Path(args.cache_dir),
                                      args.batch_size, args.gpus))
            df["k"] = pd.to_numeric(df["k"], errors="coerce")
            per_k = {str(int(k)): correlations(g["score"].to_numpy(float),
                                               g["pred"].to_numpy())
                     for k, g in df.dropna(subset=["k"]).groupby("k")}
            per_file[f.name.replace("_val.csv", "")] = per_k
            shown = "  ".join(f"k={k}:{v['kendall']:.3f}" for k, v in sorted(
                per_k.items(), key=lambda x: int(x[0])) if v["kendall"] is not None)
            logger.info(f"  {f.stem}: {shown}")

        ks = sorted({k for v in per_file.values() for k in v}, key=int)
        per_file["_mean_by_k"] = {
            k: {m: (float(np.mean(vals)) if (vals := [v[k][m] for v in per_file.values()
                                                      if k in v and v[k][m] is not None]) else None)
                for m in ("pearson", "spearman", "kendall")}
            for k in ks}
        results["models"][label] = per_file
        json.dump(results, open(out_path, "w"), indent=2)   # incremental: safe to interrupt
        del model

    logger.info(f"\n{len(models)} models → {out_path}")


if __name__ == "__main__":
    main()
