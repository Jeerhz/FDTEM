#!/usr/bin/env python3
"""
analyze.py — run every fit/control on the tidy dataframe and write CSVs.

Outputs (analysis_dir)
----------------------
  candidate_fits.csv        all candidates × (metric, k): R², CV-R², AIC/BIC, p̂+CI
  candidate_fits_pooled.csv same, pooled across k (separates sum from mean)
  p_by_k.csv                power-mean exponent p̂ (95% bootstrap CI) per (metric, k)
  p_by_condition.csv        p̂ split by severity and by error family
  position_test.csv         symmetry control: residual ~ position of the m=1 error
  residual_by_k.csv         S − f̂(parts) with f frozen at the smallest k
  dilution.csv              Δ(m, k, severity, family) — the null-model curve
  mixed_effects_<metric>.txt  optional MixedLM on residuals
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from fit import (FitResult, best_by_cv, bootstrap_param, fit_cell, fit_candidate,
                 frozen_residuals, mixed_effects_residuals, position_test)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def parts_of(df: pd.DataFrame, kind: str, eps: float) -> List[np.ndarray]:
    out = []
    for _, row in df.iterrows():
        k = int(row["k"])
        s = np.array([row[f"s_{i + 1}"] for i in range(k)], dtype=float)
        if kind == "quality":
            s = np.clip(s, eps, None)          # power mean needs positivity
        else:
            s = s + eps                        # penalties are ≥ 0
        out.append(s)
    return out


def _res_row(metric: str, scope: str, r: FitResult) -> dict:
    return {"metric": metric, "scope": scope, "candidate": r.candidate, "n": r.n,
            "n_params": r.n_params, "r2": r.r2, "rmse": r.rmse, "cv_r2": r.cv_r2,
            "cv_rmse": r.cv_rmse, "aic": r.aic, "bic": r.bic, "param": r.param,
            "param_lo": r.param_lo, "param_hi": r.param_hi, "a": r.a, "b": r.b}


def _doc_clustered_ci(resid: np.ndarray, docs: np.ndarray):
    """Mean residual ± 1.96·SEM over document-level means (clusters)."""
    s = pd.Series(resid).groupby(pd.Series(docs)).mean()
    m = float(s.mean())
    sem = float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else float("nan")
    return m, m - 1.96 * sem, m + 1.96 * sem, len(s)


# ════════════════════════════════════════════════════════════════════════════
# Main entry
# ════════════════════════════════════════════════════════════════════════════
def analyze(cfg: dict, df: pd.DataFrame) -> Dict[str, dict]:
    fit_cfg = cfg["fit"]
    eps = float(fit_cfg["eps"])
    seed = int(cfg["seed"])
    ad = Path(cfg["paths"]["analysis_dir"])
    ad.mkdir(parents=True, exist_ok=True)
    kind_of = {m["name"]: m["kind"] for m in cfg["metrics"]}

    fits_rows, pooled_rows, pbyk_rows, pcond_rows = [], [], [], []
    pos_rows, frozen_rows, summaries = [], [], {}

    for metric in sorted(df["metric"].unique()):
        dm = df[df["metric"] == metric].reset_index(drop=True)
        kind = kind_of[metric]
        ks = sorted(dm["k"].unique())
        best_cells: Dict[int, FitResult] = {}
        cell_data: Dict[int, tuple] = {}

        # ── per-k fits, all candidates ─────────────────────────────────────
        for k in ks:
            dk = dm[dm["k"] == k].reset_index(drop=True)
            if len(dk) < fit_cfg.get("min_rows_per_fit", 40):
                logger.info(f"  [{metric}, k={k}] only {len(dk)} rows — skipped")
                continue
            parts = parts_of(dk, kind, eps)
            y = dk["S"].to_numpy(float)
            groups = dk["doc_id"].to_numpy()
            results = fit_cell(parts, y, groups, kind, fit_cfg, seed)
            for r in results.values():
                fits_rows.append(_res_row(metric, f"k={k}", r))
            best = best_by_cv(results)
            best_cells[k] = best
            cell_data[k] = (dk, parts, y, groups)

            pm = results["power_mean"]
            pbyk_rows.append({"metric": metric, "k": k, "p_hat": pm.param,
                              "p_lo": pm.param_lo, "p_hi": pm.param_hi,
                              "cv_r2": pm.cv_r2, "n": pm.n})

            # severity / family interaction (power mean; m=0 rows as anchors)
            for col in ("severity", "family"):
                for level in sorted(dk[col].dropna().unique()):
                    sub = dk[(dk[col] == level) | (dk["m"] == 0)].reset_index(drop=True)
                    if len(sub) < fit_cfg.get("min_rows_per_fit", 40):
                        continue
                    r = fit_candidate("power_mean", parts_of(sub, kind, eps),
                                      sub["S"].to_numpy(float),
                                      sub["doc_id"].to_numpy(), fit_cfg, seed)
                    pcond_rows.append({"metric": metric, "k": k, "split": col,
                                       "level": level, "p_hat": r.param,
                                       "cv_r2": r.cv_r2, "n": r.n})

            # position (symmetry) control on m=1 items, k>1
            if k > 1:
                m1 = dk["m"] == 1
                if m1.sum() >= 10:
                    pos = dk.loc[m1, "positions"].astype(str).str.split(",").str[0]
                    pos_norm = pos.astype(int).to_numpy() / (k - 1)
                    resid = (y - best.predictions)[m1.to_numpy()]
                    pt = position_test(pos_norm, resid, y[m1.to_numpy()])
                    pt.update({"metric": metric, "k": k, "candidate": best.candidate})
                    pos_rows.append(pt)

        if not best_cells:
            continue

        # ── pooled fit across k (mean vs sum separate here) ────────────────
        parts_all = parts_of(dm, kind, eps)
        y_all = dm["S"].to_numpy(float)
        g_all = dm["doc_id"].to_numpy()
        pooled = fit_cell(parts_all, y_all, g_all, kind, fit_cfg, seed,
                          bootstrap=False)
        for r in pooled.values():
            pooled_rows.append(_res_row(metric, "pooled", r))

        # ── frozen-f residuals: bridge to the length experiment ────────────
        k_min = min(best_cells)
        fb = best_cells[k_min]
        for k in ks:
            if k not in cell_data:
                continue
            dk, parts, y, groups = cell_data[k]
            resid = frozen_residuals(fb.candidate, fb.param, fb.a, fb.b, parts, y)
            m, lo, hi, ndoc = _doc_clustered_ci(resid, groups)
            frozen_rows.append({"metric": metric, "k": k, "frozen_at_k": k_min,
                                "candidate": fb.candidate, "mean_resid": m,
                                "ci_lo": lo, "ci_hi": hi, "n_docs": ndoc,
                                "n": len(y)})
            if k == max(ks) and cfg["fit"].get("mixed_effects", False):
                txt = mixed_effects_residuals(resid, dk["lang"].to_numpy(),
                                              dk["doc_id"].to_numpy())
                if txt:
                    (ad / f"mixed_effects_{metric}.txt").write_text(txt,
                                                                    encoding="utf-8")

        # ── summary verdict for this metric ─────────────────────────────────
        ps = [r["p_hat"] for r in pbyk_rows if r["metric"] == metric
              and r["p_hat"] is not None]
        # "position-sensitive" needs both significance and a material effect
        # size (a lone p≈0.05 with ΔR²~1e-4 is the expected false-positive rate).
        pos_sig = any(r["metric"] == metric and not np.isnan(r["pvalue"])
                      and r["pvalue"] < 0.01 and r["delta_r2"] >= 0.005
                      for r in pos_rows)
        drift = (max(ps) - min(ps) > 2.0) if len(ps) > 1 else False
        k_ref = max(best_cells)
        summaries[metric] = {
            "kind": kind,
            "best_candidate": best_cells[k_ref].candidate,
            "best_cv_r2": best_cells[k_ref].cv_r2,
            "p_by_k": {int(r["k"]): (r["p_hat"], r["p_lo"], r["p_hi"])
                       for r in pbyk_rows if r["metric"] == metric},
            "position_sensitive": bool(pos_sig),
            "p_drifts_across_k": bool(drift),
        }

    # ── dilution curve Δ(m, k, severity, family): the null model ────────────
    base = (df[df["m"] == 0].set_index(["metric", "block_id"])["S"]
            .rename("S0"))
    dd = df[df["m"] > 0].join(base, on=["metric", "block_id"])
    dd = dd.assign(dS=dd["S"] - dd["S0"])
    dil = (dd.groupby(["metric", "k", "severity", "family", "m"])["dS"]
           .agg(mean_dS="mean", sd="std", n="count").reset_index())
    dil["sem"] = dil["sd"] / np.sqrt(dil["n"].clip(lower=1))

    pd.DataFrame(fits_rows).to_csv(ad / "candidate_fits.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(ad / "candidate_fits_pooled.csv", index=False)
    pd.DataFrame(pbyk_rows).to_csv(ad / "p_by_k.csv", index=False)
    pd.DataFrame(pcond_rows).to_csv(ad / "p_by_condition.csv", index=False)
    pd.DataFrame(pos_rows).to_csv(ad / "position_test.csv", index=False)
    pd.DataFrame(frozen_rows).to_csv(ad / "residual_by_k.csv", index=False)
    dil.to_csv(ad / "dilution.csv", index=False)
    logger.info(f"  analysis CSVs → {ad}")
    return summaries
