"""Correlation lens: agreement with human judgement per (file, k), plus the mean over files per k.

k=1 single sentence, k=2..6 aggregated window, k=0 native document. Two lenses:
  val      the pools' validation split (what early stopping monitors: a development signal)
  heldout  the WMT22/23/24/25 portions never trained on (the reportable numbers)

  python -m part2_length_training.eval_validation --lens heldout \
      --models da-base=Unbabel/wmt22-comet-da da-frac000=<ckpt>
Writes CorrelationResults to results/correlation_<lens>.json (saved after every model);
predictions are cached under results/cache/pred_<lens>, shared with eval_length_profile.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common.comet_models import load_comet, score
from common.paths import SCRATCH
from common.stats import correlations
from part2_length_training import CACHE_DIR, RESULTS_DIR
from part2_length_training.models import CorrelationResults, MeanCorrelation, ModelCorrelation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

LENS_DATA = {"val": SCRATCH / "wmt_length_data_v2", "heldout": SCRATCH / "wmt_eval_portion"}


def parse_models(entries: list[str]) -> dict[str, str]:
    models = {}
    for e in entries:
        label, _, ref = e.partition("=")
        if not ref:
            raise SystemExit(f"--models entry {e!r} must be label=<ckpt-or-hub-id>")
        models[label] = ref
    return models


def eval_files(data_dir: Path) -> list[Path]:
    files = [f for f in sorted(data_dir.glob("*_val.csv")) if not f.name.startswith("all_")]
    if not files:
        raise SystemExit(f"no *_val.csv in {data_dir}")
    return files


def mean_by_k(by_file: dict) -> dict[str, MeanCorrelation]:
    ks = sorted({k for v in by_file.values() for k in v}, key=int)
    out = {}
    for k in ks:
        vals = {}
        for m in ("pearson", "spearman", "kendall"):
            xs = [getattr(v[k], m) for v in by_file.values() if k in v and getattr(v[k], m) is not None]
            vals[m] = float(np.mean(xs)) if xs else None
        out[k] = MeanCorrelation(**vals)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens", choices=sorted(LENS_DATA), default="heldout")
    ap.add_argument("--models", nargs="+", required=True, help="label=<ckpt-or-hub-id>")
    ap.add_argument("--data_dir", default=None, help="default: the lens's directory")
    ap.add_argument("--cache_dir", default=None, help="default: results/cache/pred_<lens>")
    ap.add_argument("--output", default=None, help="default: results/correlation_<lens>.json")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=1)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser() if args.data_dir else LENS_DATA[args.lens]
    cache_dir = Path(args.cache_dir) if args.cache_dir else CACHE_DIR / f"pred_{args.lens}"
    out_path = Path(args.output) if args.output else RESULTS_DIR / f"correlation_{args.lens}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    files = eval_files(data_dir)
    models = parse_models(args.models)
    results = CorrelationResults(experiment="length_correlation",
                                 timestamp=datetime.now(timezone.utc).isoformat(),
                                 data_dir=str(data_dir), models={})

    for label, ref in models.items():
        logger.info(f"\n== {label} ==")
        model = load_comet(ref)
        by_file = {}
        for f in files:
            df = pd.read_csv(f)
            df = df.assign(pred=score(model, label, df, cache_dir, args.batch_size, args.gpus))
            df["k"] = pd.to_numeric(df["k"], errors="coerce")
            per_k = {str(int(k)): correlations(g["score"].to_numpy(float), g["pred"].to_numpy())
                     for k, g in df.dropna(subset=["k"]).groupby("k")}
            by_file[f.name.replace("_val.csv", "")] = per_k
            shown = "  ".join(f"k={k}:{v.kendall:.3f}" for k, v in sorted(per_k.items(), key=lambda x: int(x[0]))
                              if v.kendall is not None)
            logger.info(f"  {f.stem}: {shown}")
        results.models[label] = ModelCorrelation(by_file=by_file, mean_by_k=mean_by_k(by_file))
        out_path.write_text(results.model_dump_json(indent=2))  # incremental: safe to interrupt
        del model

    logger.info(f"\n{len(models)} models -> {out_path}")


if __name__ == "__main__":
    main()
