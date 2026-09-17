"""Length-profile lens: what each metric does to a text as a function of its length.

Two things the correlation lens hides: the DISTRIBUTION of the scores per k (a metric
whose long-text scores pile up in a narrow band keeps its Kendall tau and loses its
resolution; `spread_ratio_vs_k1` = this k's IQR over the k=1 IQR) and length in TOKENS
rather than sentences (fixed XLM-R token bins shared by every model and file). Two token
conventions: n_mt (the DA arms encode each side separately) and n_concat = src+mt (the
CometKiwi single 512-token sequence).

Predictions come from eval_validation's cache for the same lens; missing cells are scored.

  python -m part2_length_training.eval_length_profile --lens heldout \
      --models da-base=Unbabel/wmt22-comet-da da-frac000nat=<ckpt>
Writes LengthProfileResults to results/length_profile_<lens>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common.comet_models import load_comet, score
from common.stats import correlations
from part2_length_training import CACHE_DIR, RESULTS_DIR
from part2_length_training.eval_validation import LENS_DATA, eval_files, parse_models
from part2_length_training.models import (FileProfile, KProfileCell, LengthProfileResults, ProfileCell,
                                          ScoreDistribution)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# The last edge is the CometKiwi concatenated budget: rows beyond it are the ones the QE arms cannot see whole.
TOKEN_EDGES = (0, 32, 64, 96, 128, 192, 256, 320, 384, 448, 512, 10 ** 6)
SCORE_EDGES = np.linspace(-0.2, 1.4, 65)  # wide enough for heads that overshoot (0, 1)
QUANTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


class Tok:
    """XLM-R token counts; CometKiwi's infoxlm-large shares the sentencepiece model."""

    def __init__(self, model: str = "xlm-roberta-large"):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(model)

    def counts(self, texts: list[str], batch: int = 512) -> np.ndarray:
        out = np.empty(len(texts), dtype=np.int32)
        for i in range(0, len(texts), batch):
            enc = self.t([str(t) for t in texts[i:i + batch]], add_special_tokens=False)["input_ids"]
            out[i:i + batch] = [len(x) for x in enc]
        return out


def token_counts(df: pd.DataFrame, path: Path, cache_dir: Path, tok_factory) -> pd.DataFrame:
    """n_src / n_mt / n_ref / n_concat for a file, cached on (path, contents): same for every model."""
    key = hashlib.md5(("|".join(df["src"].astype(str)) + "\n" + "|".join(df["mt"].astype(str)))
                      .encode("utf-8")).hexdigest()[:10]
    cache = Path(cache_dir) / f"tokens__{path.stem}__{key}__n{len(df)}.npz"
    if cache.exists():
        z = np.load(cache)
        return pd.DataFrame({k: z[k] for k in z.files})
    tok = tok_factory()
    cols = {"n_src": tok.counts(df["src"].tolist()), "n_mt": tok.counts(df["mt"].tolist())}
    cols["n_ref"] = tok.counts(df["ref"].tolist()) if "ref" in df.columns else np.zeros(len(df), dtype=np.int32)
    cols["n_concat"] = cols["n_src"] + cols["n_mt"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **cols)
    logger.info(f"    tokenised {path.name}: n_mt median {int(np.median(cols['n_mt']))}, "
                f"n_concat median {int(np.median(cols['n_concat']))}")
    return pd.DataFrame(cols)


def describe(values: np.ndarray) -> ScoreDistribution:
    """Everything needed to draw a violin or a ridge line, without the raw array."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return ScoreDistribution(n=0, mean=None, std=None, min=None, max=None, quantiles={},
                                 iqr=None, hist=[], below_grid=0, above_grid=0)
    qs = np.percentile(v, QUANTILES)
    hist, _ = np.histogram(v, bins=SCORE_EDGES)
    return ScoreDistribution(
        n=int(len(v)), mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        min=float(v.min()), max=float(v.max()),
        quantiles={str(q): float(x) for q, x in zip(QUANTILES, qs)},
        iqr=float(qs[QUANTILES.index(75)] - qs[QUANTILES.index(25)]),
        hist=[int(x) for x in hist],
        below_grid=int((v < SCORE_EDGES[0]).sum()), above_grid=int((v > SCORE_EDGES[-1]).sum()))


def cell_fields(gold: np.ndarray, pred: np.ndarray, n_tok: np.ndarray) -> dict:
    """One (file, bin) cell: how the metric scores it, and how well it ranks it."""
    return dict(pred=describe(pred), gold=describe(gold),
                corr=correlations(np.asarray(gold, float), np.asarray(pred, float)),
                tokens={"mean": float(np.mean(n_tok)) if len(n_tok) else None,
                        "median": float(np.median(n_tok)) if len(n_tok) else None})


def spread_ratio(by_k: dict[str, dict], k: str) -> float | None:
    """IQR of this k's predictions relative to k=1, the regime the published metrics were trained on."""
    ref = by_k["1"]["pred"].iqr if "1" in by_k else None
    iqr = by_k[k]["pred"].iqr
    return float(iqr / ref) if ref and iqr is not None and ref > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens", choices=sorted(LENS_DATA), default="heldout")
    ap.add_argument("--models", nargs="+", required=True, help="label=<ckpt-or-hub-id>")
    ap.add_argument("--data_dir", default=None, help="default: the lens's directory")
    ap.add_argument("--cache_dir", default=None, help="default: results/cache/pred_<lens> (shared with eval_validation)")
    ap.add_argument("--output", default=None, help="default: results/length_profile_<lens>.json")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--token_edges", nargs="+", type=int, default=list(TOKEN_EDGES))
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser() if args.data_dir else LENS_DATA[args.lens]
    cache_dir = Path(args.cache_dir) if args.cache_dir else CACHE_DIR / f"pred_{args.lens}"
    out_path = Path(args.output) if args.output else RESULTS_DIR / f"length_profile_{args.lens}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    files = eval_files(data_dir)
    models = parse_models(args.models)
    edges = np.asarray(args.token_edges, dtype=float)
    bin_labels = [f"{int(edges[i])}-{int(edges[i + 1])}" if edges[i + 1] < 10 ** 6 else f"{int(edges[i])}+"
                  for i in range(len(edges) - 1)]

    results = LengthProfileResults(
        experiment="length_profile",
        question="how the score distribution and the rank agreement move with input length, "
                 "in sentences (k) and in XLM-R tokens",
        timestamp=datetime.now(timezone.utc).isoformat(), data_dir=str(data_dir),
        token_edges=[int(e) for e in edges], token_bins=bin_labels,
        score_grid=[float(x) for x in SCORE_EDGES], quantiles=list(QUANTILES), models={})

    logger.info("Token counts")  # every file once, before any model is loaded
    tokens = {f.name: token_counts(pd.read_csv(f), f, cache_dir, Tok) for f in files}

    for label, ref in models.items():
        logger.info(f"\n== {label} ==")
        model = load_comet(ref)
        per_file = {}
        for f in files:
            df = pd.read_csv(f)
            df = df.assign(pred=score(model, label, df, cache_dir, args.batch_size, args.gpus))
            tk = tokens[f.name]
            df["n_mt"], df["n_concat"] = tk["n_mt"].to_numpy(), tk["n_concat"].to_numpy()
            df["k"] = pd.to_numeric(df["k"], errors="coerce")
            gold, pred = df["score"].to_numpy(float), df["pred"].to_numpy(float)

            by_k_fields = {str(int(k)): cell_fields(g["score"].to_numpy(float), g["pred"].to_numpy(), g["n_mt"].to_numpy())
                           for k, g in df.dropna(subset=["k"]).groupby("k")}
            by_k = {k: KProfileCell(**c, spread_ratio_vs_k1=spread_ratio(by_k_fields, k))
                    for k, c in by_k_fields.items()}
            by_tokens = {}
            for field in ("n_mt", "n_concat"):
                idx = np.digitize(df[field].to_numpy(float), edges[1:-1], right=False)
                by_bin = {}
                for b in range(len(bin_labels)):
                    g = df[idx == b]
                    if len(g):
                        by_bin[bin_labels[b]] = ProfileCell(**cell_fields(
                            g["score"].to_numpy(float), g["pred"].to_numpy(), g[field].to_numpy()))
                by_tokens[field] = by_bin
            per_file[f.name.replace("_val.csv", "")] = FileProfile(
                all=ProfileCell(**cell_fields(gold, pred, df["n_mt"].to_numpy())), by_k=by_k, by_tokens=by_tokens)
            logger.info("  " + f.stem + ": " + "  ".join(
                f"k={k} mean={by_k[k].pred.mean:.3f} iqr={by_k[k].pred.iqr:.3f}" for k in sorted(by_k, key=int)))

        results.models[label] = per_file
        out_path.write_text(results.model_dump_json(indent=2))  # incremental
        del model

    logger.info(f"\n{len(models)} model(s) -> {out_path}")


if __name__ == "__main__":
    main()
