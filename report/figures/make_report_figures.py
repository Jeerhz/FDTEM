#!/usr/bin/env python3
"""Figures for the FDTEM report that do not exist elsewhere.

Reads results JSONs, writes PDF figures into report/figures/.
Palette: validated categorical slots (light mode).
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b9b8b3"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#eceae4", "grid.linewidth": 0.6,
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ── 1. validation curves of the invalid sweeps (bell shape) ──────────────────
diag = json.load(open(ROOT / "results/length_training/wandb_curve_diagnosis.json"))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=False)

ax = axes[0]
picks = {
    "mix-frac000-20260817-1145": (BLUE, "DA frac000"),
    "mix-frac100-20260817-1145": (ORANGE, "DA frac100"),
    "kiwi-mix-frac000agg-frozen-20260818-1025": (AQUA, "QE frac000agg fr."),
    "kiwi-mix-frac020-frozen-20260818-0419": (YELLOW, "QE frac020 fr."),
}
for run in diag["comet-retrain-wmt"]["runs"]:
    if run["name"] not in picks or not run.get("history"):
        continue
    c, lab = picks[run["name"]]
    ep = [h["epoch"] for h in run["history"]]
    v = [h["value"] for h in run["history"]]
    ax.plot(ep, v, color=c, lw=1.6)
    pk = run["peak_index"]
    ax.plot(ep[pk], v[pk], "o", color=c, ms=4)
    ax.annotate(lab, (ep[-1], v[-1]), textcoords="offset points",
                xytext=(3, 0), fontsize=7.5, color=c, va="center")
ax.set_xlim(0, 75)
ax.set_title("WMT sweep (invalid, 2026-08-17)", fontsize=9)
ax.set_xlabel("epoch")
ax.set_ylabel(r"validation Kendall $\tau$")

ax = axes[1]
bio = [r for r in diag["comet-retrain"]["runs"]
       if r["name"].startswith("mix-frac") and r.get("history")
       and r.get("n_val_points", 0) >= 20]
for run in bio:
    ep = [h["epoch"] for h in run["history"]]
    v = [h["value"] for h in run["history"]]
    ax.plot(ep, v, color=MUTED, lw=0.9, alpha=0.8)
hl = next(r for r in bio if r["name"].startswith("mix-frac000-2"))
ep = [h["epoch"] for h in hl["history"]]
v = [h["value"] for h in hl["history"]]
ax.plot(ep, v, color=BLUE, lw=1.8)
pk = hl["peak_index"]
ax.plot(ep[pk], v[pk], "o", color=BLUE, ms=4)
ax.annotate("frac000", (ep[-1], v[-1]), textcoords="offset points",
            xytext=(3, 0), fontsize=7.5, color=BLUE, va="center")
ax.set_title("Bio-MQM sweep (invalid, 2026-08-14)", fontsize=9)
ax.set_xlabel("epoch")
save(fig, "wandb_bell.pdf")

# ── 2. baseline Kendall vs window size k (WMT eval portions) ─────────────────
corr = json.load(open(ROOT / "results/length_training/correlation.json"))
ks = ["1", "2", "3", "4", "6"]
fig, ax = plt.subplots(figsize=(3.6, 2.5))
for label, c, name in (("da-base", BLUE, "COMET-DA"),
                       ("qe-base", ORANGE, "CometKiwi")):
    m = corr["models"][label]["_mean_by_k"]
    ax.plot(range(len(ks)), [m[k]["kendall"] for k in ks], "-o", color=c,
            lw=1.6, ms=4)
    ax.plot([len(ks) + 0.5], [m["0"]["kendall"]], "s", color=c, ms=5)
    ax.annotate(name, (len(ks) - 1, m["6"]["kendall"]),
                textcoords="offset points", xytext=(-2, 6), fontsize=8,
                color=c, ha="right")
ax.set_xticks(list(range(len(ks))) + [len(ks) + 0.5])
ax.set_xticklabels([f"k={k}" for k in ks] + ["docs"])
ax.set_ylabel(r"Kendall $\tau$ (mean over sets)")
ax.set_xlabel("evaluation window (segments per input)")
save(fig, "baseline_tau_vs_k.pdf")

# ── 3. MetaDocEval baselines: accuracy vs context window w ───────────────────
mde = json.load(open(ROOT / "results/metadoceval/accuracy.json"))
WINDOWS = ["1", "3", "6", "9"]
G1, G2, G3 = "#6b6a66", "#93928d", "#b9b8b3"
groups = {
    "tense_consistency": (BLUE, "tense"),
    "lexical_consistency": (ORANGE, "lexical"),
    "conjunction_substitution": (AQUA, "conjunction"),
    "sentence_repetition": (G2, "repetition"),
    "sentence_removal": (G1, "removal"),
    "sentence_shuffling": (G3, "shuffling"),
    "sentence_splitting": (MAGENTA, "splitting (FPR)"),
}
# manual y-offsets (points) for the right-edge labels, per model
NUDGE = {"kiwi": {"tense": -6, "conjunction": 3, "lexical": -3,
                  "splitting (FPR)": 4, "repetition": -4, "shuffling": 4},
         "wmt22": {}}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
for ax, model, title in ((axes[0], "wmt22", "COMET-DA"),
                         (axes[1], "kiwi", "CometKiwi")):
    micro = mde["models"][model]["micro"]
    for cat, (c, lab) in groups.items():
        ys = [micro[f"{cat}|w{w}"]["accuracy"] for w in WINDOWS]
        ls = ":" if cat == "sentence_splitting" else "-"
        lw = 1.2 if c in (G1, G2, G3) else 1.6
        ax.plot(range(4), ys, ls, color=c, lw=lw, ms=3, marker="o")
        if model == "kiwi":
            dy = NUDGE[model].get(lab, 0)
            ax.annotate(lab, (3, ys[-1]), textcoords="offset points",
                        xytext=(5, dy), fontsize=7, color=c, va="center")
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(WINDOWS)
    ax.set_xlabel("context window $w$ (SLIDE)")
axes[0].set_ylabel("contrastive accuracy")
axes[0].set_ylim(0.45, 1.03)
axes[1].set_xlim(-0.15, 4.9)
axes[0].set_xlim(-0.15, 3.15)
save(fig, "metadoceval_base.pdf")

# ── 4. matched-core: detection vs L, per filler, from the exact JSON ─────────
mc = json.load(open(ROOT / "results/matched_core/matched_core.json"))
Ls = ["60", "120", "240", "480"]
ENCS = [("comet:wmt22-comet-da", BLUE, "COMET"),
        ("xlmr:xlm-roberta-large", MUTED, "XLM-R"),
        ("labse", ORANGE, "LaBSE"),
        ("e5:multilingual-e5-base", AQUA, "E5")]
FILLERS = [("inert", "inert filler"), ("neutral", "neutral filler"),
           ("natural", "natural filler")]
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.4), sharey=True)
for ax, (phi, title) in zip(axes, FILLERS):
    for key, c, lab in ENCS:
        ys = [mc["encoders"][key]["_mean"][phi][L]["detection"] for L in Ls]
        ax.plot(range(4), ys, "-o", color=c, lw=1.6, ms=3.5)
        if phi == "natural":
            ax.annotate(lab, (3, ys[-1]), textcoords="offset points",
                        xytext=(5, 0), fontsize=7.5, color=c, va="center")
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.set_xticks(range(4)); ax.set_xticklabels(Ls)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("input length $L$ (tokens)")
axes[0].set_ylabel("detection rate")
axes[0].set_ylim(0.28, 1.02)
axes[2].set_xlim(-0.15, 4.2)
save(fig, "matched_core_report.pdf")
