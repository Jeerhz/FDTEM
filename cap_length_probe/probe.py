#!/usr/bin/env python3
"""
probe.py — train/eval the linear probe on frozen features.

Classifier: multinomial logistic regression (the primary probe, for
reproducibility), class_weight="balanced", on z-scored features (train stats
only). A 1-hidden-layer MLP is available behind `probe.classifier: mlp`.

Two evaluation regimes
----------------------
  regime1 (probe_per_length)   : for each (encoder, L) train AND test at length L.
  regime2 (train_short)        : train once at the shortest L (probe.train_short_L),
                                 then evaluate at every L — does the representation
                                 stay usable as inputs lengthen?

Seeds: each probe is trained over >= 5 seeds. The seed draws a fixed-fraction
sub-sample of the training set (probe.train_subsample_frac), so for the otherwise
deterministic LR the seed spread reflects sensitivity to the training subset; for
the MLP it also varies initialisation. We report mean +/- std across seeds, and
analyze.py adds bootstrap CIs over test items.

Writes
  results/results.csv          tidy: encoder, lang, L, seed, acc, macro_f1 (+ regime, meta)
  results/predictions.csv.gz    seed-0 per-item predictions (for bootstrap CIs)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from common import N_CLASSES, ensure_dirs, load_config, banner

logger = logging.getLogger(__name__)


def _make_clf(cfg: dict, seed: int):
    p = cfg["probe"]
    if p["classifier"] == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(int(p["mlp_hidden"]),),
                             max_iter=int(p["max_iter"]), random_state=seed,
                             early_stopping=True)
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=float(p["C"]), class_weight="balanced",
                              max_iter=int(p["max_iter"]), random_state=seed,
                              n_jobs=-1)


def _metrics(y_true, y_pred) -> Tuple[float, float]:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=list(range(N_CLASSES)),
                  average="macro", zero_division=0)
    return float(acc), float(f1)


def _resample_train(n: int, seed: int, frac: float) -> np.ndarray:
    if frac >= 1.0:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    k = max(N_CLASSES, int(round(n * frac)))
    return rng.choice(n, size=min(k, n), replace=False)


def _fit_predict(Xtr, ytr, Xte, cfg, seed):
    """z-score on train stats, fit, predict test. Returns (y_pred, scaler)."""
    if cfg["probe"]["standardize"]:
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    else:
        sc = None
    clf = _make_clf(cfg, seed)
    clf.fit(Xtr, ytr)
    return clf, sc


def _eval_rows(encoder, regime, L, seed, y_true, y_pred, langs, cfg) -> List[dict]:
    rows = []
    acc, f1 = _metrics(y_true, y_pred)
    rows.append(dict(encoder=encoder, lang="all", L=L, seed=seed, acc=acc, macro_f1=f1,
                     regime=regime, n_test=len(y_true)))
    for lg in sorted(set(langs)):
        m = langs == lg
        if m.sum() == 0:
            continue
        a, f = _metrics(y_true[m], y_pred[m])
        rows.append(dict(encoder=encoder, lang=lg, L=int(L), seed=seed, acc=a, macro_f1=f,
                         regime=regime, n_test=int(m.sum())))
    return rows


def run_probes(cfg: dict, df: pd.DataFrame,
               feats: Dict[str, Dict[int, np.ndarray]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    buckets = list(cfg["length_buckets"])
    seeds = list(cfg["probe"]["seeds"])
    frac = float(cfg["probe"].get("train_subsample_frac", 0.9))
    L0 = int(cfg["probe"]["train_short_L"])
    is_train = (df["split"].values == "train")
    is_test = (df["split"].values == "test")
    y = df["label"].values
    langs = df["lang"].values
    y_te, lang_te = y[is_test], langs[is_test]

    results: List[dict] = []
    preds: List[dict] = []
    doc_te = df["doc_id"].values[is_test]

    for enc_tag, per_L in feats.items():
        banner(f"probe: {enc_tag}  classifier={cfg['probe']['classifier']}")
        # ── regime 1: probe-per-length ──
        for L in buckets:
            Xtr_full, Xte = per_L[L][is_train], per_L[L][is_test]
            ytr_full = y[is_train]
            for seed in seeds:
                sub = _resample_train(len(Xtr_full), seed, frac)
                clf, sc = _fit_predict(Xtr_full[sub], ytr_full[sub], Xte, cfg, seed)
                Xte_t = sc.transform(Xte) if sc is not None else Xte
                yp = clf.predict(Xte_t)
                results += _eval_rows(enc_tag, "probe_per_length", L, seed, y_te, yp, lang_te, cfg)
                if seed == seeds[0]:
                    preds += [dict(encoder=enc_tag, regime="probe_per_length", L=int(L),
                                   seed=int(seed), doc_id=d, lang=lg, y_true=int(t), y_pred=int(p))
                              for d, lg, t, p in zip(doc_te, lang_te, y_te, yp)]
            acc = np.mean([r["acc"] for r in results
                           if r["encoder"] == enc_tag and r["L"] == L and r["lang"] == "all"
                           and r["regime"] == "probe_per_length"])
            print(f"  [r1] L={L:<4} mean acc={acc:.4f}")

        # ── regime 2: train-short, test-across-length ──
        Xtr0_full = per_L[L0][is_train]
        ytr0_full = y[is_train]
        for seed in seeds:
            sub = _resample_train(len(Xtr0_full), seed, frac)
            clf, sc = _fit_predict(Xtr0_full[sub], ytr0_full[sub], Xtr0_full[sub], cfg, seed)
            for L in buckets:
                Xte = per_L[L][is_test]
                Xte_t = sc.transform(Xte) if sc is not None else Xte
                yp = clf.predict(Xte_t)
                results += _eval_rows(enc_tag, "train_short", L, seed, y_te, yp, lang_te, cfg)
                if seed == seeds[0]:
                    preds += [dict(encoder=enc_tag, regime="train_short", L=int(L),
                                   seed=int(seed), doc_id=d, lang=lg, y_true=int(t), y_pred=int(p))
                              for d, lg, t, p in zip(doc_te, lang_te, y_te, yp)]

    res_df = pd.DataFrame(results)
    pred_df = pd.DataFrame(preds)
    # static metadata columns (same for all rows)
    for k, v in dict(pooling=cfg["features"]["pooling"],
                     normalize_l2=cfg["features"]["normalize_l2"],
                     classifier=cfg["probe"]["classifier"],
                     backend=cfg["features"]["backend"]).items():
        res_df[k] = v
    return res_df, pred_df


def save(cfg: dict, res_df: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    ensure_dirs(cfg)
    rd = Path(cfg["paths"]["results_dir"])
    res_path = rd / "results.csv"
    res_df.to_csv(res_path, index=False)
    pred_path = rd / "predictions.csv.gz"
    pred_df.to_csv(pred_path, index=False, compression="gzip")
    print(f"\n  results → {res_path}  ({len(res_df)} rows)")
    print(f"  preds   → {pred_path}  ({len(pred_df)} rows)")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.dataset:
        df = pd.read_csv(args.dataset)
    else:
        from data import build_dataset
        df = build_dataset(cfg, save=True)
    from features import extract_all
    feats = extract_all(cfg, df, cfg["length_buckets"])
    res_df, pred_df = run_probes(cfg, df, feats)
    save(cfg, res_df, pred_df)


if __name__ == "__main__":
    main()
