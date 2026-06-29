#!/usr/bin/env python3
"""
analyze.py — metrics, slopes, CIs, and the COMET-vs-XLM-R gap.

From results/results.csv (+ results/predictions.csv.gz) it produces:
  analysis/summary.csv        per (encoder, regime, L, lang): acc/macro-F1 mean+/-std
                              across seeds, plus bootstrap 95% CIs over test items.
  analysis/slopes.csv         per encoder: slope of acc & macro-F1 on log2(L)
                              (regime probe_per_length, lang=all) — the length-sensitivity
                              summary. Flatter/steeper slope answers the hypothesis.
  analysis/gap_by_L.csv       COMET - XLM-R accuracy/F1 gap as a function of L.
  analysis/per_language.csv   per-language acc/F1 vs L.
  analysis/regime2.csv        train-short / test-across-length stability.
  analysis/geometry.csv       (optional) mean norm, anisotropy, and L2 / Wasserstein
                              distance of length-L features to length-32, per encoder.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from common import N_CLASSES, ensure_dirs, load_config, banner

logger = logging.getLogger(__name__)
PRIMARY = "probe_per_length"


def role_of(encoder_tag: str) -> str:
    return "comet" if encoder_tag.startswith("comet") else "raw"


# ════════════════════════════════════════════════════════════════════════════
# Aggregation across seeds
# ════════════════════════════════════════════════════════════════════════════
def aggregate(res: pd.DataFrame) -> pd.DataFrame:
    g = (res.groupby(["encoder", "regime", "L", "lang"], as_index=False)
            .agg(acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                 f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
                 n_seeds=("seed", "nunique"), n_test=("n_test", "max")))
    g[["acc_std", "f1_std"]] = g[["acc_std", "f1_std"]].fillna(0.0)
    g["role"] = g["encoder"].map(role_of)
    return g.sort_values(["regime", "encoder", "lang", "L"]).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# Bootstrap 95% CIs over test items (seed-0 predictions)
# ════════════════════════════════════════════════════════════════════════════
def _boot(y_true: np.ndarray, y_pred: np.ndarray, n: int, ci: float, seed: int):
    """Bootstrap acc / macro-F1 CIs. Vectorised via a per-resample confusion matrix
    (bincount), so it scales to large test sets without 1000s of sklearn calls.
    macro-F1 averages over all N_CLASSES (absent classes contribute 0; matches
    sklearn f1_score(labels=range(N), average='macro', zero_division=0))."""
    m = len(y_true)
    if m == 0:
        return (np.nan,) * 4
    K = N_CLASSES
    cell = (y_true.astype(np.int64) * K + y_pred.astype(np.int64))
    correct = (y_true == y_pred).astype(np.float64)
    rng = np.random.RandomState(seed)
    accs, f1s = np.empty(n), np.empty(n)
    for i in range(n):
        idx = rng.randint(0, m, m)
        cm = np.bincount(cell[idx], minlength=K * K).reshape(K, K).astype(np.float64)
        tp = np.diag(cm)
        denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)  # 2TP + FP + FN
        f1c = np.where(denom > 0, 2 * tp / denom, 0.0)
        f1s[i] = f1c.mean()
        accs[i] = correct[idx].mean()
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return (np.percentile(accs, lo), np.percentile(accs, hi),
            np.percentile(f1s, lo), np.percentile(f1s, hi))


def bootstrap_cis(preds: Optional[pd.DataFrame], cfg: dict) -> pd.DataFrame:
    if preds is None or len(preds) == 0:
        return pd.DataFrame()
    b = cfg["bootstrap"]
    n, ci, seed = int(b["n_resamples"]), float(b["ci"]), int(b["seed"])
    rows = []
    for (enc, regime, L), grp in preds.groupby(["encoder", "regime", "L"]):
        for lang in ["all"] + sorted(grp["lang"].unique().tolist()):
            sub = grp if lang == "all" else grp[grp["lang"] == lang]
            a_lo, a_hi, f_lo, f_hi = _boot(sub["y_true"].values, sub["y_pred"].values, n, ci, seed)
            rows.append(dict(encoder=enc, regime=regime, L=int(L), lang=lang,
                             acc_lo=a_lo, acc_hi=a_hi, f1_lo=f_lo, f1_hi=f_hi))
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Length-sensitivity slopes + gap
# ════════════════════════════════════════════════════════════════════════════
def length_slopes(summary: pd.DataFrame) -> pd.DataFrame:
    s = summary[(summary["regime"] == PRIMARY) & (summary["lang"] == "all")]
    rows = []
    for enc, g in s.groupby("encoder"):
        g = g.sort_values("L")
        x = np.log2(g["L"].values.astype(float))
        for metric, col in [("acc", "acc_mean"), ("macro_f1", "f1_mean")]:
            y = g[col].values
            if len(x) >= 2:
                slope, intercept = np.polyfit(x, y, 1)
                yhat = slope * x + intercept
                ss_res = np.sum((y - yhat) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            else:
                slope = intercept = r2 = np.nan
            rows.append(dict(encoder=enc, role=role_of(enc), metric=metric,
                             slope_per_log2L=slope, intercept=intercept, r2=r2,
                             value_at_minL=g[col].iloc[0], value_at_maxL=g[col].iloc[-1]))
    return pd.DataFrame(rows)


def gap_by_L(summary: pd.DataFrame) -> pd.DataFrame:
    s = summary[(summary["regime"] == PRIMARY) & (summary["lang"] == "all")]
    comet = s[s["role"] == "comet"].set_index("L")
    raw = s[s["role"] == "raw"].set_index("L")
    Ls = sorted(set(comet.index) & set(raw.index))
    rows = []
    for L in Ls:
        rows.append(dict(L=int(L),
                         comet_acc=comet.loc[L, "acc_mean"], raw_acc=raw.loc[L, "acc_mean"],
                         gap_acc=comet.loc[L, "acc_mean"] - raw.loc[L, "acc_mean"],
                         comet_f1=comet.loc[L, "f1_mean"], raw_f1=raw.loc[L, "f1_mean"],
                         gap_f1=comet.loc[L, "f1_mean"] - raw.loc[L, "f1_mean"]))
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Optional geometry diagnostics (needs in-memory features)
# ════════════════════════════════════════════════════════════════════════════
def geometry_diagnostics(feats: Dict[str, Dict[int, np.ndarray]], cfg: dict,
                         sample: int = 800) -> pd.DataFrame:
    from scipy.stats import wasserstein_distance
    rng = np.random.RandomState(0)
    L_ref = int(cfg["probe"]["train_short_L"])
    rows = []
    for enc, per_L in feats.items():
        ref = per_L.get(L_ref)
        for L, X in sorted(per_L.items()):
            norms = np.linalg.norm(X, axis=1)
            k = min(sample, len(X))
            idx = rng.choice(len(X), k, replace=False)
            Xs = X[idx]
            Xn = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-9)
            cos = Xn @ Xn.T
            aniso = (cos.sum() - np.trace(cos)) / (k * (k - 1)) if k > 1 else np.nan
            paired_l2 = float(np.linalg.norm(X - ref, axis=1).mean()) if ref is not None else np.nan
            wass = float(wasserstein_distance(norms, np.linalg.norm(ref, axis=1))) if ref is not None else np.nan
            rows.append(dict(encoder=enc, role=role_of(enc), L=int(L),
                             mean_norm=float(norms.mean()), anisotropy=float(aniso),
                             paired_l2_to_Lref=paired_l2, wasserstein_norm_to_Lref=wass,
                             L_ref=L_ref))
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════════════════════════════════
def analyze(cfg: dict, feats: Optional[Dict[str, Dict[int, np.ndarray]]] = None) -> Dict[str, pd.DataFrame]:
    ensure_dirs(cfg)
    rd = Path(cfg["paths"]["results_dir"])
    ad = Path(cfg["paths"]["analysis_dir"])
    res = pd.read_csv(rd / "results.csv")
    preds_path = rd / "predictions.csv.gz"
    preds = pd.read_csv(preds_path) if preds_path.exists() else None

    summary = aggregate(res)
    cis = bootstrap_cis(preds, cfg)
    if len(cis):
        summary = summary.merge(cis, on=["encoder", "regime", "L", "lang"], how="left")
    slopes = length_slopes(summary)
    gap = gap_by_L(summary)
    per_lang = summary[(summary["regime"] == PRIMARY) & (summary["lang"] != "all")].copy()
    regime2 = summary[summary["regime"] == "train_short"].copy()

    out = {"summary": summary, "slopes": slopes, "gap_by_L": gap,
           "per_language": per_lang, "regime2": regime2}
    if feats is not None and cfg.get("geometry", {}).get("enabled", False):
        out["geometry"] = geometry_diagnostics(feats, cfg)

    for name, dfo in out.items():
        dfo.to_csv(ad / f"{name}.csv", index=False)

    banner("Analysis — length sensitivity (regime: probe_per_length, lang=all)")
    print(slopes.to_string(index=False))
    print("\nCOMET - XLM-R gap by length:")
    print(gap.to_string(index=False))
    print(f"\n  analysis tables → {ad}/")
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    analyze(cfg)


if __name__ == "__main__":
    main()
