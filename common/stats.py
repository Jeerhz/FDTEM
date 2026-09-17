"""Correlations and document-level bootstrap.

Windows of the same document share errors and text, so resampling examples would
understate uncertainty. Everything here resamples documents.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel


class CorrelationCell(BaseModel):
    n: int
    pearson: float | None
    spearman: float | None
    kendall: float | None


def correlations(gold: np.ndarray, pred: np.ndarray) -> CorrelationCell:
    from scipy.stats import kendalltau, pearsonr, spearmanr

    if len(gold) < 3 or np.std(gold) == 0 or np.std(pred) == 0:
        return CorrelationCell(n=int(len(gold)), pearson=None, spearman=None, kendall=None)
    return CorrelationCell(n=int(len(gold)),
                           pearson=float(pearsonr(gold, pred)[0]),
                           spearman=float(spearmanr(gold, pred)[0]),
                           kendall=float(kendalltau(gold, pred)[0]))


def doc_units(df: pd.DataFrame, gold: str = "score", pred: str = "pred",
              doc: str = "doc_id", group: str | None = None) -> list[tuple]:
    """Split a scored frame into resamplable (group, gold, pred) units, one per document."""
    units = []
    for keys, g in df.groupby([c for c in (group, doc) if c], sort=False):
        gid = keys[0] if group else 0
        units.append((gid, g[gold].to_numpy(float), g[pred].to_numpy(float)))
    return units


def tau(units: Sequence[tuple], index: Sequence[int] | None = None) -> float:
    """Mean Kendall tau, computed per group then averaged (groups = files / language pairs)."""
    from scipy.stats import kendalltau

    per_group: dict[object, tuple[list, list]] = {}
    for j in (range(len(units)) if index is None else index):
        gid, gold, pred = units[j]
        per_group.setdefault(gid, ([], []))
        per_group[gid][0].append(gold)
        per_group[gid][1].append(pred)
    taus = []
    for golds, preds in per_group.values():
        g, p = np.concatenate(golds), np.concatenate(preds)
        if len(g) > 2 and g.std() > 0 and p.std() > 0:
            taus.append(kendalltau(g, p)[0])
    return float(np.mean(taus)) if taus else float("nan")


def bootstrap_delta(units_a: Sequence[tuple], units_b: Sequence[tuple],
                    n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap of tau(a) - tau(b) over documents -> (delta, lo, hi)."""
    if len(units_a) != len(units_b):
        raise ValueError("paired bootstrap needs the same documents on both sides")
    rng = np.random.RandomState(seed)
    n = len(units_a)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        deltas[i] = tau(units_a, idx) - tau(units_b, idx)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return tau(units_a) - tau(units_b), float(lo), float(hi)
