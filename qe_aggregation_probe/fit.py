#!/usr/bin/env python3
"""
fit.py — estimate the aggregation function f in S ≈ f(s_1, ..., s_k).

Candidates (each optionally wrapped in an affine calibration S ≈ a + b·g(s), so
every candidate gets the same 2 free calibration parameters and AIC/BIC compare
shapes, not offsets):

  quality scale (COMET / CometKiwi; higher = better)
    mean · min · max · median · power_mean(p) · softmin(T)
  error/penalty scale (MetricX; higher = worse)
    mean · sum · min · max · power_mean(p) · softmax(T)

The generalized power mean  M_p(s) = (1/k Σ s_i^p)^(1/p)  interpolates
continuously: p=1 arithmetic mean, p→0 geometric, p→−∞ min, p→+∞ max. The
fitted p IS the answer. softmin/softmax are the log-sum-exp analogue with a
temperature. NOTE: within a fixed k, `sum` is affine-equivalent to `mean`; they
only separate in pooled fits across k.

Model selection: cross-validated R²/RMSE with GroupKFold BY DOCUMENT, plus
AIC/BIC on the full fit. Free parameters get bootstrap CIs resampled BY DOCUMENT.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import linregress

logger = logging.getLogger(__name__)

FREE_PARAM_CANDIDATES = {"power_mean", "softmin", "softmax"}


# ════════════════════════════════════════════════════════════════════════════
# Aggregation features (rows may have heterogeneous k: list of 1-d arrays)
# ════════════════════════════════════════════════════════════════════════════
def power_mean(S: np.ndarray, p: float) -> np.ndarray:
    """Generalized power mean over axis 1 of a positive (n,k) matrix. Stable."""
    logS = np.log(S)
    if abs(p) < 1e-4:                       # geometric-mean limit
        return np.exp(logS.mean(axis=1))
    k = S.shape[1]
    return np.exp((logsumexp(p * logS, axis=1) - np.log(k)) / p)


def _softmin(S: np.ndarray, T: float) -> np.ndarray:
    k = S.shape[1]
    return -T * (logsumexp(-S / T, axis=1) - np.log(k))


def _softmax_agg(S: np.ndarray, T: float) -> np.ndarray:
    k = S.shape[1]
    return T * (logsumexp(S / T, axis=1) - np.log(k))


def feature(name: str, parts: List[np.ndarray], param: Optional[float] = None) -> np.ndarray:
    """Compute the candidate aggregation g(s) row-wise; handles ragged k by
    grouping rows with equal k."""
    out = np.empty(len(parts))
    ks = np.array([len(x) for x in parts])
    for k in np.unique(ks):
        idx = np.where(ks == k)[0]
        S = np.stack([parts[i] for i in idx])
        if name == "mean":
            v = S.mean(1)
        elif name == "sum":
            v = S.sum(1)
        elif name == "min":
            v = S.min(1)
        elif name == "max":
            v = S.max(1)
        elif name == "median":
            v = np.median(S, 1)
        elif name == "power_mean":
            v = power_mean(S, float(param))
        elif name == "softmin":
            v = _softmin(S, float(param))
        elif name == "softmax":
            v = _softmax_agg(S, float(param))
        else:
            raise ValueError(f"unknown candidate {name!r}")
        out[idx] = v
    return out


def candidates_for(kind: str) -> List[str]:
    if kind == "quality":
        return ["mean", "min", "max", "median", "power_mean", "softmin"]
    return ["mean", "sum", "min", "max", "power_mean", "softmax"]


# ════════════════════════════════════════════════════════════════════════════
# Affine calibration + least-squares machinery
# ════════════════════════════════════════════════════════════════════════════
def _affine_fit(f: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    X = np.column_stack([np.ones_like(f), f])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1]), X @ coef


def _rss_for(name: str, parts, y, param, affine: bool) -> Tuple[float, Tuple[float, float]]:
    f = feature(name, parts, param)
    if affine:
        a, b, pred = _affine_fit(f, y)
    else:
        a, b, pred = 0.0, 1.0, f
    return float(((y - pred) ** 2).sum()), (a, b)


def _grid(name: str, fit_cfg: dict) -> np.ndarray:
    if name == "power_mean":
        g = fit_cfg["p_grid"]
        return np.linspace(g["lo"], g["hi"], int(g["n"]))
    g = fit_cfg["softmin_logT_grid"]
    return 10.0 ** np.linspace(g["lo"], g["hi"], int(g["n"]))


def _fit_free_param(name: str, parts, y, fit_cfg: dict, refine: bool = True,
                    grid: Optional[np.ndarray] = None
                    ) -> Tuple[float, float, Tuple[float, float]]:
    """Grid + local refine of the inner parameter. Returns (param, rss, (a,b))."""
    if grid is None:
        grid = _grid(name, fit_cfg)
    affine = fit_cfg.get("affine", True)
    rss = np.array([_rss_for(name, parts, y, p, affine)[0] for p in grid])
    i = int(np.argmin(rss))
    best_p = float(grid[i])
    if refine and 0 < i < len(grid) - 1:
        lo, hi = float(grid[i - 1]), float(grid[i + 1])
        res = minimize_scalar(lambda p: _rss_for(name, parts, y, p, affine)[0],
                              bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-3 * max(1.0, abs(hi - lo))})
        if res.fun <= rss[i]:
            best_p = float(res.x)
    best_rss, ab = _rss_for(name, parts, y, best_p, affine)
    return best_p, best_rss, ab


# ════════════════════════════════════════════════════════════════════════════
# Group K-fold (by document) — no sklearn dependency
# ════════════════════════════════════════════════════════════════════════════
def group_kfold(groups: Sequence[str], n_folds: int, seed: int):
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, min(n_folds, len(uniq)))
    garr = np.asarray(groups)
    for fold in folds:
        test = np.isin(garr, fold)
        yield ~test, test


# ════════════════════════════════════════════════════════════════════════════
# Fitting one candidate on one cell
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class FitResult:
    candidate: str
    n: int
    n_params: int
    r2: float
    rmse: float
    cv_r2: float
    cv_rmse: float
    aic: float
    bic: float
    param: Optional[float] = None
    param_lo: Optional[float] = None
    param_hi: Optional[float] = None
    a: float = 0.0
    b: float = 1.0
    predictions: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]


def _predict(name, parts, param, a, b):
    return a + b * feature(name, parts, param)


def fit_candidate(name: str, parts, y, groups, fit_cfg: dict, seed: int) -> FitResult:
    affine = fit_cfg.get("affine", True)
    n = len(y)
    free = name in FREE_PARAM_CANDIDATES

    if free:
        param, rss, (a, b) = _fit_free_param(name, parts, y, fit_cfg)
    else:
        param = None
        rss, (a, b) = _rss_for(name, parts, y, None, affine)

    pred = _predict(name, parts, param, a, b)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / sst if sst > 0 else float("nan")
    rmse = float(np.sqrt(rss / n))
    n_params = (2 if affine else 0) + (1 if free else 0)
    aic = n * np.log(max(rss, 1e-12) / n) + 2 * n_params
    bic = n * np.log(max(rss, 1e-12) / n) + n_params * np.log(n)

    # ── grouped CV (pooled out-of-fold R²) ────────────────────────────────
    oof = np.full(n, np.nan)
    for tr, te in group_kfold(groups, fit_cfg.get("cv_folds", 5), seed):
        parts_tr = [parts[i] for i in np.where(tr)[0]]
        parts_te = [parts[i] for i in np.where(te)[0]]
        if free:
            p_tr, _, (a_tr, b_tr) = _fit_free_param(name, parts_tr, y[tr], fit_cfg,
                                                    refine=False)
        else:
            p_tr = None
            _, (a_tr, b_tr) = _rss_for(name, parts_tr, y[tr], None, affine)
        oof[te] = _predict(name, parts_te, p_tr, a_tr, b_tr)
    cv_rss = float(((y - oof) ** 2).sum())
    cv_r2 = 1.0 - cv_rss / sst if sst > 0 else float("nan")
    cv_rmse = float(np.sqrt(cv_rss / n))

    return FitResult(candidate=name, n=n, n_params=n_params, r2=r2, rmse=rmse,
                     cv_r2=cv_r2, cv_rmse=cv_rmse, aic=float(aic), bic=float(bic),
                     param=param, a=a, b=b, predictions=pred)


def bootstrap_param(name: str, parts, y, groups, fit_cfg: dict,
                    center: Optional[float] = None) -> Tuple[float, float]:
    """Percentile CI for the free parameter, resampling DOCUMENTS.

    Uses a fine grid centered on the full-fit estimate (when given) so the CI
    is not quantised by the coarse global search grid."""
    bs = fit_cfg["bootstrap"]
    rng = np.random.default_rng(bs.get("seed", 12345))
    grid = None
    if center is not None:
        if name == "power_mean":
            grid = np.linspace(center - 5.0, center + 5.0, 101)
        else:                                   # temperatures are positive
            grid = center * (10.0 ** np.linspace(-0.7, 0.7, 57))
    uniq = sorted(set(groups))
    garr = np.asarray(groups)
    idx_of = {g: np.where(garr == g)[0] for g in uniq}
    est = []
    for _ in range(int(bs["n_resamples"])):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_of[g] for g in take])
        p, _, _ = _fit_free_param(name, [parts[i] for i in idx], y[idx], fit_cfg,
                                  refine=False, grid=grid)
        est.append(p)
    alpha = (1.0 - bs.get("ci", 0.95)) / 2.0
    return (float(np.quantile(est, alpha)), float(np.quantile(est, 1 - alpha)))


# ════════════════════════════════════════════════════════════════════════════
# The full cell fit: all candidates on one (metric, k) slice
# ════════════════════════════════════════════════════════════════════════════
def fit_cell(parts, y, groups, kind: str, fit_cfg: dict, seed: int,
             bootstrap: bool = True) -> Dict[str, FitResult]:
    results: Dict[str, FitResult] = {}
    for name in candidates_for(kind):
        res = fit_candidate(name, parts, y, groups, fit_cfg, seed)
        if bootstrap and name in FREE_PARAM_CANDIDATES:
            res.param_lo, res.param_hi = bootstrap_param(name, parts, y, groups,
                                                         fit_cfg, center=res.param)
        results[name] = res
    return results


def best_by_cv(results: Dict[str, FitResult]) -> FitResult:
    return max(results.values(), key=lambda r: (np.nan_to_num(r.cv_r2, nan=-np.inf)))


# ════════════════════════════════════════════════════════════════════════════
# Controls
# ════════════════════════════════════════════════════════════════════════════
def position_test(pos_norm: np.ndarray, resid: np.ndarray, y: np.ndarray) -> dict:
    """Is f symmetric? Regress residuals of the best fit on the (normalized)
    position of the single perturbed sentence (m=1 items only)."""
    if len(pos_norm) < 10 or np.allclose(pos_norm, pos_norm[0]):
        return {"slope": np.nan, "pvalue": np.nan, "delta_r2": np.nan,
                "n": len(pos_norm)}
    lr = linregress(pos_norm, resid)
    sst_y = float(((y - y.mean()) ** 2).sum())
    explained = float((lr.slope ** 2) * ((pos_norm - pos_norm.mean()) ** 2).sum())
    return {"slope": float(lr.slope), "pvalue": float(lr.pvalue),
            "delta_r2": explained / sst_y if sst_y > 0 else np.nan,
            "n": int(len(pos_norm))}


def frozen_residuals(name: str, param, a, b, parts, y) -> np.ndarray:
    """Residual S − f̂(parts) under a FROZEN f (fitted at the smallest k).
    The systematic part of this residual as k grows is the candidate
    length/encoder effect that aggregation alone cannot explain."""
    return y - _predict(name, parts, param, a, b)


def mixed_effects_residuals(resid: np.ndarray, lang: np.ndarray,
                            doc: np.ndarray) -> Optional[str]:
    """Optional statsmodels MixedLM: resid ~ C(lang) + (1|doc)."""
    try:
        import pandas as pd  # noqa: PLC0415
        import statsmodels.formula.api as smf  # noqa: PLC0415
        df = pd.DataFrame({"resid": resid, "lang": lang, "doc": doc})
        md = smf.mixedlm("resid ~ C(lang)", df, groups=df["doc"])
        fit = md.fit(reml=True, method="lbfgs")
        return str(fit.summary())
    except Exception as exc:  # noqa: BLE001
        logger.info(f"  mixed-effects model skipped: {exc}")
        return None
