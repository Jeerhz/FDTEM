#!/usr/bin/env python3
"""
plots.py — figures from the analysis CSVs + tidy dataframe.

  obs_vs_pred_<metric>.png    observed S vs f̂(parts), best candidate, per k
  p_by_k_<metric>.png         power-mean exponent p̂ (CI) vs k
  candidate_cv_<metric>.png   CV-R² per candidate per k
  dilution_<metric>.png       Δ(m) per k, faceted severity × family (null model)
  residual_by_k_<metric>.png  frozen-f residual vs k (bridge to length effect)
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze import parts_of  # noqa: E402
from fit import feature  # noqa: E402

logger = logging.getLogger(__name__)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [plot] {path}")


def make_plots(cfg: dict, df: pd.DataFrame) -> None:
    ad = Path(cfg["paths"]["analysis_dir"])
    fd = Path(cfg["paths"]["figures_dir"])
    fits = pd.read_csv(ad / "candidate_fits.csv")
    pbyk = pd.read_csv(ad / "p_by_k.csv")
    dil = pd.read_csv(ad / "dilution.csv")
    frozen = pd.read_csv(ad / "residual_by_k.csv")
    kind_of = {m["name"]: m["kind"] for m in cfg["metrics"]}
    eps = float(cfg["fit"]["eps"])

    for metric in sorted(df["metric"].unique()):
        dm = df[df["metric"] == metric]
        fm = fits[fits["metric"] == metric]
        ks = sorted(fm["scope"].str.replace("k=", "").astype(int).unique())

        # ── observed vs predicted (best candidate per k) ───────────────────
        fig, axes = plt.subplots(1, len(ks), figsize=(4.2 * len(ks), 4),
                                 squeeze=False)
        for ax, k in zip(axes[0], ks):
            sub = fm[fm["scope"] == f"k={k}"].sort_values("cv_r2", ascending=False)
            best = sub.iloc[0]
            dk = dm[dm["k"] == k].reset_index(drop=True)
            parts = parts_of(dk, kind_of[metric], eps)
            pred = best["a"] + best["b"] * feature(
                best["candidate"], parts,
                best["param"] if pd.notna(best["param"]) else None)
            y = dk["S"].to_numpy(float)
            ax.scatter(pred, y, s=8, alpha=0.35, linewidths=0)
            lims = [min(pred.min(), y.min()), max(pred.max(), y.max())]
            ax.plot(lims, lims, color="grey", ls=":", lw=1)
            lbl = best["candidate"]
            if pd.notna(best["param"]):
                lbl += f" (p={best['param']:.2f})"
            ax.set_title(f"k={k} — {lbl}\nCV R²={best['cv_r2']:.3f}", fontsize=9)
            ax.set_xlabel("predicted f̂(parts)")
            ax.set_ylabel("observed S")
        fig.suptitle(f"{metric}: observed vs best-candidate prediction", y=1.02)
        _save(fig, fd / f"obs_vs_pred_{metric}.png")

        # ── p by k ─────────────────────────────────────────────────────────
        pk = pbyk[pbyk["metric"] == metric].sort_values("k")
        if len(pk):
            fig, ax = plt.subplots(figsize=(5.5, 4))
            yerr = np.array([pk["p_hat"] - pk["p_lo"], pk["p_hi"] - pk["p_hat"]])
            ax.errorbar(pk["k"], pk["p_hat"], yerr=np.nan_to_num(yerr),
                        marker="o", capsize=4)
            ax.axhline(1.0, color="tab:green", ls="--", lw=1, label="p=1 (mean)")
            ax.axhline(0.0, color="tab:orange", ls=":", lw=1, label="p=0 (geometric)")
            ax.set_xlabel("k (sentences per paragraph)")
            ax.set_ylabel("fitted power-mean exponent p̂  (bootstrap 95% CI)")
            ax.set_title(f"{metric}: aggregation exponent vs k\n"
                         "(p→−∞ = min / worst-sentence-dominates)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            _save(fig, fd / f"p_by_k_{metric}.png")

        # ── candidate comparison (CV R²) ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(7.5, 4))
        cands = sorted(fm["candidate"].unique())
        x = np.arange(len(cands))
        w = 0.8 / max(1, len(ks))
        for i, k in enumerate(ks):
            sub = fm[fm["scope"] == f"k={k}"].set_index("candidate")
            ys = [sub.loc[c, "cv_r2"] if c in sub.index else np.nan for c in cands]
            ax.bar(x + i * w, ys, w, label=f"k={k}")
        ax.set_xticks(x + w * (len(ks) - 1) / 2)
        ax.set_xticklabels(cands, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("cross-validated R² (GroupKFold by doc)")
        ax.set_title(f"{metric}: which aggregation shape fits best?")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        _save(fig, fd / f"candidate_cv_{metric}.png")

        # ── dilution curves: the null model Δ(m, k, severity, family) ──────
        dmm = dil[dil["metric"] == metric]
        if len(dmm):
            sevs = sorted(dmm["severity"].dropna().unique())
            fams = sorted(dmm["family"].dropna().unique())
            fig, axes = plt.subplots(len(sevs), len(fams),
                                     figsize=(4.5 * len(fams), 3.6 * len(sevs)),
                                     squeeze=False, sharey=True)
            for r, sev in enumerate(sevs):
                for c, fam in enumerate(fams):
                    ax = axes[r][c]
                    for k in ks:
                        sub = dmm[(dmm["severity"] == sev) & (dmm["family"] == fam)
                                  & (dmm["k"] == k)].sort_values("m")
                        if not len(sub):
                            continue
                        ax.errorbar(sub["m"], sub["mean_dS"], yerr=1.96 * sub["sem"],
                                    marker="o", capsize=3, label=f"k={k}")
                    ax.axhline(0, color="grey", ls=":", lw=1)
                    ax.set_title(f"{sev} {fam}", fontsize=10)
                    ax.set_xlabel("m (perturbed sentences)")
                    if c == 0:
                        ax.set_ylabel("Δ score  S(m) − S(0)")
                    ax.grid(alpha=0.3)
            axes[0][0].legend(fontsize=8)
            fig.suptitle(f"{metric}: dilution curve Δ(m, k, severity, family)",
                         y=1.01)
            _save(fig, fd / f"dilution_{metric}.png")

        # ── frozen-f residual vs k ─────────────────────────────────────────
        fz = frozen[frozen["metric"] == metric].sort_values("k")
        if len(fz):
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.errorbar(fz["k"], fz["mean_resid"],
                        yerr=[fz["mean_resid"] - fz["ci_lo"],
                              fz["ci_hi"] - fz["mean_resid"]],
                        marker="s", capsize=4, color="tab:red")
            ax.axhline(0, color="grey", ls=":", lw=1)
            ax.set_xlabel("k (sentences per paragraph)")
            ax.set_ylabel("mean residual  S − f̂(parts)")
            k0 = int(fz["frozen_at_k"].iloc[0])
            ax.set_title(f"{metric}: residual under f frozen at k={k0}\n"
                         "(non-zero growth = length effect beyond aggregation)")
            ax.grid(alpha=0.3)
            _save(fig, fd / f"residual_by_k_{metric}.png")
