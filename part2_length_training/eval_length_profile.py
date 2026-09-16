#!/usr/bin/env python3
"""
eval_length_profile.py — what each metric DOES to a text, as a function of how
long that text is.

`eval_correlation.py` answers "does the score still rank translations the way a
human would, per window size k". That is one number per cell, and it hides two
things this script reports instead:

  1. **The distribution of the scores themselves.** A metric whose k=6 scores
     all pile up at 0.85 can still have a respectable Kendall tau — ranking is
     invariant to any monotone squashing — while being useless in practice,
     because the differences it reports are no longer resolvable. Score mean,
     spread, quantiles and a histogram per k say directly what happens to the
     scale as text gets longer; `spread_ratio_vs_k1` (this k's IQR over the k=1
     IQR) is the one-number version.

  2. **Length in tokens, not in sentences.** k is a proxy: a k=6 window of short
     segments and a native document can differ by a factor of ten in actual
     input length, and the pools have different sentence-length distributions,
     so a curve against k confounds "more sentences" with "more tokens". Every
     quantity here is therefore ALSO reported against the XLM-R token count of
     the model's real input, in fixed bins shared by every model and file so the
     curves are directly comparable.

Two token conventions, both reported, because the two families do not read the
same thing:
    n_mt      tokens of the translation alone — the DA arms encode src, mt and
              ref separately, so this is the length of one encoder input.
    n_concat  tokens of src+mt — CometKiwi packs them into ONE 512-token
              sequence, so this is the length that its 512-token budget applies
              to, and the length at which it starts truncating.

Predictions come from `eval_correlation.py`'s cache when it is there (keyed by
model label + exact input rows + checkpoint fingerprint), so running this after
an evaluation costs no GPU. Missing cells are scored and cached like any other.

Usage:
  python experiments/length_training/eval_length_profile.py \
      --models da-base=Unbabel/wmt22-comet-da da-frac000nat=<ckpt> \
      --data_dir ~/scratch/wmt_eval_portion \
      --cache_dir results/length_training/pred_cache_heldout \
      --output results/length_training/length_profile_heldout.json
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

from fdtem.comet_io import load_comet, score
from fdtem.stats import correlations

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Shared by every model, file and lens, so any two cells are comparable. The
# last edge is the CometKiwi concatenated budget: rows beyond it are the ones
# the QE arms cannot see whole.
TOKEN_EDGES = (0, 32, 64, 96, 128, 192, 256, 320, 384, 448, 512, 10 ** 6)
# Fixed score grid, wide enough for heads that overshoot (0, 1).
SCORE_EDGES = np.linspace(-0.2, 1.4, 65)
QUANTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def parse_models(entries: list[str]) -> dict[str, str]:
    models = {}
    for e in entries:
        label, _, ref = e.partition("=")
        if not ref:
            raise SystemExit(f"--models entry {e!r} must be label=<ckpt-or-hub-id>")
        models[label] = ref
    return models


class Tok:
    """XLM-R token counts. Both families share this vocabulary — CometKiwi's
    infoxlm-large uses the XLM-R sentencepiece model — so one tokenizer gives
    the input length for the DA and the QE arms alike."""

    def __init__(self, model: str = "xlm-roberta-large"):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(model)

    def counts(self, texts: list[str], batch: int = 512) -> np.ndarray:
        out = np.empty(len(texts), dtype=np.int32)
        for i in range(0, len(texts), batch):
            chunk = [str(t) for t in texts[i:i + batch]]
            enc = self.t(chunk, add_special_tokens=False)["input_ids"]
            out[i:i + batch] = [len(x) for x in enc]
        return out


def token_counts(df: pd.DataFrame, path: Path, cache_dir: Path, tok_factory
                 ) -> pd.DataFrame:
    """n_src / n_mt / n_ref / n_concat for a file, cached on (path, contents).

    Tokenising the WMT25 documents is minutes of CPU, and it is the same answer
    for every model, so it is computed once per file rather than once per cell.
    """
    key = hashlib.md5(
        ("|".join(df["src"].astype(str)) + "\n" +
         "|".join(df["mt"].astype(str))).encode("utf-8")).hexdigest()[:10]
    cache = Path(cache_dir) / f"tokens__{path.stem}__{key}__n{len(df)}.npz"
    if cache.exists():
        z = np.load(cache)
        return pd.DataFrame({k: z[k] for k in z.files})
    tok = tok_factory()
    cols = {"n_src": tok.counts(df["src"].tolist()),
            "n_mt": tok.counts(df["mt"].tolist())}
    cols["n_ref"] = (tok.counts(df["ref"].tolist()) if "ref" in df.columns
                     else np.zeros(len(df), dtype=np.int32))
    cols["n_concat"] = cols["n_src"] + cols["n_mt"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **cols)
    logger.info(f"    tokenised {path.name}: n_mt median "
                f"{int(np.median(cols['n_mt']))}, n_concat median "
                f"{int(np.median(cols['n_concat']))}")
    return pd.DataFrame(cols)


def describe(values: np.ndarray) -> dict:
    """Everything needed to draw a violin or a ridge line, without the raw array."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0}
    qs = np.percentile(v, QUANTILES)
    hist, _ = np.histogram(v, bins=SCORE_EDGES)
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "min": float(v.min()), "max": float(v.max()),
        "quantiles": {str(q): float(x) for q, x in zip(QUANTILES, qs)},
        "iqr": float(qs[QUANTILES.index(75)] - qs[QUANTILES.index(25)]),
        "hist": [int(x) for x in hist],
        "below_grid": int((v < SCORE_EDGES[0]).sum()),
        "above_grid": int((v > SCORE_EDGES[-1]).sum()),
    }


def cell(gold: np.ndarray, pred: np.ndarray, n_tok: np.ndarray) -> dict:
    """One (file, bin) cell: how the metric scores it, and how well it ranks it."""
    return {
        "pred": describe(pred),
        "gold": describe(gold),
        "corr": correlations(np.asarray(gold, float), np.asarray(pred, float)),
        "tokens": {"mean": float(np.mean(n_tok)) if len(n_tok) else None,
                   "median": float(np.median(n_tok)) if len(n_tok) else None},
    }


def add_spread_ratio(by_k: dict) -> None:
    """IQR of this k's predictions relative to k=1 — the compression number.

    k=1 is the reference because it is the regime the published metrics were
    trained on. Below 1 means the metric resolves smaller quality differences on
    longer text than it does on sentences, whatever its rank correlation says.
    """
    ref = by_k.get("1", {}).get("pred", {}).get("iqr")
    for k, c in by_k.items():
        iqr = c.get("pred", {}).get("iqr")
        c["spread_ratio_vs_k1"] = (float(iqr / ref) if ref and iqr is not None
                                   and ref > 0 else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="label=<ckpt-or-hub-id>")
    ap.add_argument("--data_dir", default="~/scratch/wmt_eval_portion")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--cache_dir", default="results/length_training/pred_cache_heldout",
                    help="Shared with eval_correlation.py — point it at the same "
                         "directory and nothing is rescored.")
    ap.add_argument("--token_edges", nargs="+", type=int, default=list(TOKEN_EDGES))
    ap.add_argument("--output",
                    default="results/length_training/length_profile.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    files = [f for f in sorted(data_dir.glob("*_val.csv")) if not f.name.startswith("all_")]
    if not files:
        raise SystemExit(f"no *_val.csv in {data_dir}")
    models = parse_models(args.models)
    edges = np.asarray(args.token_edges, dtype=float)
    bin_labels = [f"{int(edges[i])}-{int(edges[i + 1])}" if edges[i + 1] < 10 ** 6
                  else f"{int(edges[i])}+" for i in range(len(edges) - 1)]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "length_profile",
        "question": "how the score distribution and the rank agreement move "
                    "with input length, in sentences (k) and in XLM-R tokens",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "token_edges": [int(e) for e in edges],
        "token_bins": bin_labels,
        "score_grid": [float(x) for x in SCORE_EDGES],
        "quantiles": list(QUANTILES),
        "models": {},
    }

    # tokenise every file once, before any model is loaded
    logger.info("Token counts")
    tokens = {}
    for f in files:
        df = pd.read_csv(f)
        tokens[f.name] = token_counts(df, f, Path(args.cache_dir), Tok)

    for label, ref in models.items():
        logger.info(f"\n══ {label} ══")
        model = load_comet(ref)
        per_file = {}
        for f in files:
            df = pd.read_csv(f)
            df = df.assign(pred=score(model, label, df, Path(args.cache_dir),
                                      args.batch_size, args.gpus))
            tk = tokens[f.name]
            df["n_mt"], df["n_concat"] = tk["n_mt"].to_numpy(), tk["n_concat"].to_numpy()
            df["k"] = pd.to_numeric(df["k"], errors="coerce")

            gold, pred = df["score"].to_numpy(float), df["pred"].to_numpy(float)
            entry = {
                "all": cell(gold, pred, df["n_mt"].to_numpy()),
                "by_k": {}, "by_tokens": {},
            }
            for k, g in df.dropna(subset=["k"]).groupby("k"):
                entry["by_k"][str(int(k))] = cell(
                    g["score"].to_numpy(float), g["pred"].to_numpy(),
                    g["n_mt"].to_numpy())
            add_spread_ratio(entry["by_k"])

            for field in ("n_mt", "n_concat"):
                idx = np.digitize(df[field].to_numpy(float), edges[1:-1], right=False)
                by_bin = {}
                for b in range(len(bin_labels)):
                    g = df[idx == b]
                    if len(g) == 0:
                        continue
                    by_bin[bin_labels[b]] = cell(
                        g["score"].to_numpy(float), g["pred"].to_numpy(),
                        g[field].to_numpy())
                entry["by_tokens"][field] = by_bin

            per_file[f.name.replace("_val.csv", "")] = entry
            ks = sorted(entry["by_k"], key=int)
            logger.info("  " + f.stem + ": " + "  ".join(
                f"k={k} mean={entry['by_k'][k]['pred']['mean']:.3f}"
                f" iqr={entry['by_k'][k]['pred']['iqr']:.3f}" for k in ks))

        results["models"][label] = per_file
        json.dump(results, open(out_path, "w"), indent=2)   # incremental
        del model

    logger.info(f"\n{len(models)} model(s) → {out_path}")


if __name__ == "__main__":
    main()
