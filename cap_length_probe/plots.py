#!/usr/bin/env python3
"""
plots.py — figures for the length-probe.

Reads the analysis/*.csv tables and writes to results/figures/:
  (a) accuracy_vs_length.png   acc & macro-F1 vs L, one line per encoder, CI bands
  (b) gap_vs_length.png        COMET - XLM-R accuracy/F1 gap vs L
  (c) per_language_accuracy.png  small multiples, one panel per language
  (d) regime2_stability.png    train-short / test-across-length curves
  (e) geometry_vs_length.png   (optional) norm / anisotropy / drift vs L
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_config

logger = logging.getLogger(__name__)
PRIMARY = "probe_per_length"
STYLE = {"comet": dict(color="#c0392b", marker="o", label="COMET encoder"),
         "raw": dict(color="#2c6fbb", marker="s", label="raw XLM-R")}


def _role(tag: str) -> str:
    return "comet" if str(tag).startswith("comet") else "raw"


def _band(ax, x, mean, lo, hi, std, color):
    if lo is not None and np.isfinite(lo).all():
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
    else:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)


def _logx(ax, Ls):
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ls)
    ax.set_xticklabels([str(int(l)) for l in Ls])
    ax.set_xlabel("input length L (tokens)")
    ax.grid(alpha=0.3, which="both")


def fig_accuracy_vs_length(summary: pd.DataFrame, out: Path):
    s = summary[(summary["regime"] == PRIMARY) & (summary["lang"] == "all")].copy()
    s["role"] = s["encoder"].map(_role)
    Ls = sorted(s["L"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (col_m, col_s, lo_c, hi_c, title) in zip(axes, [
            ("acc_mean", "acc_std", "acc_lo", "acc_hi", "Accuracy"),
            ("f1_mean", "f1_std", "f1_lo", "f1_hi", "Macro-F1")]):
        for role, g in s.groupby("role"):
            g = g.sort_values("L")
            x = g["L"].values
            lo = g[lo_c].values if lo_c in g else None
            hi = g[hi_c].values if hi_c in g else None
            _band(ax, x, g[col_m].values, lo, hi, g[col_s].values, STYLE[role]["color"])
            ax.plot(x, g[col_m].values, **STYLE[role])
        _logx(ax, Ls)
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs input length (probe-per-length)")
        ax.legend(fontsize=8)
    fig.suptitle("CAP major-topic probe — frozen features vs input length", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("  [plot] %s", out)


def fig_gap_vs_length(gap: pd.DataFrame, out: Path):
    if gap.empty:
        return
    gap = gap.sort_values("L")
    Ls = gap["L"].values
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axhline(0, color="grey", lw=1, ls="--")
    ax.plot(Ls, gap["gap_acc"].values, marker="o", color="#8e44ad", label="Δ accuracy")
    ax.plot(Ls, gap["gap_f1"].values, marker="^", color="#16a085", label="Δ macro-F1")
    _logx(ax, Ls)
    ax.set_ylabel("COMET − XLM-R")
    ax.set_title("COMET-encoder advantage over raw XLM-R vs length\n(negative ⇒ COMET worse)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("  [plot] %s", out)


def fig_per_language(per_lang: pd.DataFrame, out: Path):
    if per_lang.empty:
        return
    per_lang = per_lang.copy()
    per_lang["role"] = per_lang["encoder"].map(_role)
    langs = sorted(per_lang["lang"].unique())
    Ls = sorted(per_lang["L"].unique())
    ncol = min(3, len(langs))
    nrow = int(np.ceil(len(langs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for i, lang in enumerate(langs):
        ax = axes[i // ncol][i % ncol]
        sub = per_lang[per_lang["lang"] == lang]
        for role, g in sub.groupby("role"):
            g = g.sort_values("L")
            ax.plot(g["L"].values, g["acc_mean"].values, **STYLE[role])
        _logx(ax, Ls)
        ax.set_title(f"lang = {lang}")
        ax.set_ylabel("accuracy")
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(langs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Accuracy vs length per language", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("  [plot] %s", out)


def fig_regime2(regime2: pd.DataFrame, out: Path):
    s = regime2[regime2["lang"] == "all"].copy()
    if s.empty:
        return
    s["role"] = s["encoder"].map(_role)
    Ls = sorted(s["L"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for role, g in s.groupby("role"):
        g = g.sort_values("L")
        x = g["L"].values
        lo = g["acc_lo"].values if "acc_lo" in g else None
        hi = g["acc_hi"].values if "acc_hi" in g else None
        _band(ax, x, g["acc_mean"].values, lo, hi, g["acc_std"].values, STYLE[role]["color"])
        ax.plot(x, g["acc_mean"].values, **STYLE[role])
    _logx(ax, Ls)
    ax.set_ylabel("accuracy")
    ax.set_title("Stability: probe trained at short L, evaluated across length")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("  [plot] %s", out)


def fig_geometry(geo: pd.DataFrame, out: Path):
    if geo is None or geo.empty:
        return
    geo = geo.copy()
    geo["role"] = geo["encoder"].map(_role)
    Ls = sorted(geo["L"].unique())
    cols = [("mean_norm", "mean embedding norm"),
            ("anisotropy", "anisotropy (avg pairwise cosine)"),
            ("paired_l2_to_Lref", "‖feat(L) − feat(Lref)‖ (paired)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (col, title) in zip(axes, cols):
        for role, g in geo.groupby("role"):
            g = g.sort_values("L")
            ax.plot(g["L"].values, g[col].values, **STYLE[role])
        _logx(ax, Ls)
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.suptitle("Representation geometry vs input length", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("  [plot] %s", out)


def make_plots(cfg: dict) -> None:
    ad = Path(cfg["paths"]["analysis_dir"])
    fd = Path(cfg["paths"]["figures_dir"])
    fd.mkdir(parents=True, exist_ok=True)

    def _read(name):
        p = ad / f"{name}.csv"
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    summary = _read("summary")
    fig_accuracy_vs_length(summary, fd / "accuracy_vs_length.png")
    fig_gap_vs_length(_read("gap_by_L"), fd / "gap_vs_length.png")
    fig_per_language(_read("per_language"), fd / "per_language_accuracy.png")
    fig_regime2(_read("regime2"), fd / "regime2_stability.png")
    fig_geometry(_read("geometry"), fd / "geometry_vs_length.png")
    print(f"  figures → {fd}/")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    make_plots(load_config(args.config))


if __name__ == "__main__":
    main()
