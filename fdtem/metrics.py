"""Retrieval metrics (xsim) over L2-normalised embedding matrices."""
from __future__ import annotations

from typing import Tuple

import numpy as np

def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A,B assumed L2-normalised → cosine = dot product."""
    return A @ B.T


def xsim_error(src: np.ndarray, tgt: np.ndarray, margin: str = "ratio",
               k: int = 4) -> Tuple[float, np.ndarray]:
    """LASER-style xsim error rate for aligned src/tgt (row i ↔ row i).

    margin="cosine"  : nearest neighbour by plain cosine.
    margin="ratio"   : margin-based score cos(x,y) / mean_k of the two sides'
                       top-k neighbour cosines (robust to hubness; LASER default).
    Returns (error_rate, predicted_indices).
    """
    sim = cosine_matrix(src, tgt)  # (N, N)
    if margin == "cosine":
        pred = sim.argmax(1)
    elif margin == "ratio":
        n = sim.shape[0]
        kk = min(k, n - 1) if n > 1 else 1
        # average top-k similarity from each side (exclude self via partition)
        fwd = np.sort(sim, axis=1)[:, -kk:].mean(1, keepdims=True)        # (N,1)
        bwd = np.sort(sim, axis=0)[-kk:, :].mean(0, keepdims=True)        # (1,N)
        denom = (fwd + bwd) / 2.0
        pred = (sim / np.clip(denom, 1e-9, None)).argmax(1)
    else:
        raise ValueError(margin)
    err = float((pred != np.arange(len(pred))).mean())
    return err, pred
