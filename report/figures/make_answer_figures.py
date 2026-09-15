#!/usr/bin/env python3
"""Figures answering the three evaluation questions, from the results JSONs.

Q1  agreement with human judgement per input regime (single sentences,
    concatenated windows, natively long documents)   → results/length_training/
Q2  MetaDocEval contrastive accuracy per model       → results/{metadoceval,length_training}/
Q3  encoders on the FLORES+ concatenated-block frame → results/{block_xsim,matched_core}/

Writes PDF + PNG into report/figures/answers/.
Palette: the validated categorical slots (light mode).
"""
from pathlib import Path
import json
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "answers"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
VIOLET, RED = "#4a3aa7", "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8f8e8a", "#eceae4"
SURFACE = "#fcfcfb"
GREYS = ["#6b6a66", "#93928d", "#b9b8b3"]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#104281"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f7f9fc"] + BLUES)

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.major.size": 0, "ytick.major.size": 0,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def note(fig, text, y=-0.04, width=130):
    """Caption under the figure, hard-wrapped. Call AFTER tight_layout()."""
    wrapped = "\n".join(textwrap.fill(par, width) for par in text.split("\n"))
    fig.text(0.0, y, wrapped, fontsize=7, color=MUTED, ha="left", va="top",
             linespacing=1.5, transform=fig.transFigure)


# ════════════════════════════════════════════════════════════════════════════
# Q1 — agreement with human judgement, per input regime (held-out sets)
# ════════════════════════════════════════════════════════════════════════════
held = json.load(open(ROOT / "results/length_training/correlation_heldout.json"))["models"]
KS = ["1", "2", "3", "4", "6"]
CONCAT = ["2", "3", "4", "6"]

# Arms are named by WHAT THEY WERE TRAINED ON, not by the sweep's f-fraction code:
#   frac000 → "texte long"  (0 % de phrases isolées : moitié fenêtres agrégées, moitié documents)
#   frac100 → "phrases"     (100 % de phrases isolées)
PRETTY = {
    "da-base": "COMET-DA publié", "da-frac000": "DA · texte long",
    "da-frac000-frozen": "DA · texte long, gelé", "da-frac100": "DA · phrases",
    "da-frac100-frozen": "DA · phrases, gelé",
    "qe-base": "CometKiwi publié", "qe-frac000": "Kiwi · texte long",
    "qe-frac000-frozen": "Kiwi · texte long, gelé", "qe-frac100": "Kiwi · phrases",
    "qe-frac100-frozen": "Kiwi · phrases, gelé",
}
ORDER = ["da-base", "da-frac000", "da-frac000-frozen", "da-frac100", "da-frac100-frozen",
         "qe-base", "qe-frac000", "qe-frac000-frozen", "qe-frac100", "qe-frac100-frozen"]


def tau(model, k):
    return held[model]["_mean_by_k"][k]["kendall"]


def tau_concat(model):
    return float(np.mean([tau(model, k) for k in CONCAT]))


# ── Fig 1: the three regimes, ranked ────────────────────────────────────────
REGIMES = [
    ("Phrases seules", "$k{=}1$ · WMT22 en-de/en-ru/zh-en", lambda m: tau(m, "1")),
    ("Phrases concaténées", "moyenne $k{\\in}\\{2,3,4,6\\}$ · mêmes paires",
     tau_concat),
    ("Documents natifs", "$k{=}0$ · WMT23/24 + WMT25, 8 jeux", lambda m: tau(m, "0")),
]
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5))
for ax, (title, sub, fn) in zip(axes, REGIMES):
    vals = sorted(((fn(m), m) for m in ORDER), key=lambda t: t[0])
    ys = np.arange(len(vals))
    for y, (v, m) in zip(ys, vals):
        fam_blue = m.startswith("da")
        base = m.endswith("base")
        c = (BLUE if fam_blue else ORANGE)
        ax.barh(y, v, height=0.68, color=c, alpha=0.35 if base else 1.0,
                edgecolor=c, linewidth=1.2 if base else 0, zorder=3)
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=7.5,
                color=INK if not base else INK2,
                fontweight="bold" if y == len(vals) - 1 else "normal")
    ax.set_yticks(ys)
    ax.set_yticklabels([PRETTY[m] for _, m in vals], fontsize=7.5)
    ax.set_xlim(0, max(v for v, _ in vals) * 1.22)
    ax.set_title(title, fontsize=9.5, pad=10)
    ax.text(0.5, 1.005, sub, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.3, color=MUTED)
    ax.set_xlabel(r"Kendall $\tau$ vs. jugement humain")
    ax.grid(axis="y", visible=False)
handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
           plt.Rectangle((0, 0), 1, 1, color=ORANGE),
           plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK2)]
fig.legend(handles, ["famille COMET-DA (avec référence)",
                     "famille CometKiwi (sans référence)",
                     "métrique publiée, non ré-entraînée"],
           loc="lower center", ncol=3, frameon=False, fontsize=8,
           bbox_to_anchor=(0.5, -0.09))
fig.tight_layout()
note(fig,
     "Évalué sur — jeux hors entraînement : segments et fenêtres MQM WMT22 (3 paires, 400 items par cellule) "
     "pour k≥1 ; paragraphes MQM WMT23/24 et documents ESA WMT25 (8 jeux) pour k=0. Colonnes non comparables "
     "entre elles.\n"
     "« texte long » = mélange d'entraînement sans phrase isolée (code frac000) ; « phrases » = 100 % de "
     "phrases isolées (frac100) ; « gelé » = encodeur figé. Bras Kiwi = bornes inférieures (troncature v1).",
     y=-0.10)
save(fig, "q1_regimes")

# ── Fig 2: tau vs window size, per family ───────────────────────────────────
ARMS = [("frac000", BLUE, "-"), ("frac000-frozen", BLUE, "--"),
        ("frac100", ORANGE, "-"), ("frac100-frozen", ORANGE, "--")]
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1), sharey=False)
for ax, fam, title in ((axes[0], "da", "COMET-DA  (avec référence)"),
                       (axes[1], "qe", "CometKiwi  (sans référence)")):
    xs = np.arange(len(KS))
    b = held[f"{fam}-base"]["_mean_by_k"]
    ax.plot(xs, [b[k]["kendall"] for k in KS], "-o", color=MUTED, lw=2.4, ms=5, zorder=4)
    for suffix, c, ls in ARMS:
        m = held[f"{fam}-{suffix}"]["_mean_by_k"]
        ax.plot(xs, [m[k]["kendall"] for k in KS], ls, color=c, lw=2, zorder=5,
                marker="o", ms=4.5, markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"k={k}" for k in KS], fontsize=8)
    ax.set_xlim(-0.25, len(KS) - 0.75)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("segments par entrée évaluée")
axes[0].set_ylabel(r"Kendall $\tau$ (held-out)")
handles = [Line2D([], [], color=MUTED, lw=2, marker="o", ms=5),
           Line2D([], [], color=BLUE, lw=2), Line2D([], [], color=BLUE, lw=2, ls="--"),
           Line2D([], [], color=ORANGE, lw=2), Line2D([], [], color=ORANGE, lw=2, ls="--")]
fig.legend(handles, ["métrique publiée", "ré-entraîné sur texte long", "texte long, encodeur gelé",
                     "ré-entraîné sur phrases", "phrases, encodeur gelé"],
           loc="lower center", ncol=5, frameon=False, fontsize=7.8,
           bbox_to_anchor=(0.5, -0.10))
fig.tight_layout()
note(fig,
     "Évalué sur — segments et fenêtres MQM WMT22 (en-de, en-ru, zh-en), hors entraînement. La note d'une "
     "fenêtre est la moyenne des notes de segments : la cible se débruite quand k grandit, d'où la remontée.",
     y=-0.14)
save(fig, "q1_tau_vs_k")

# ── Fig 3: per test set, document-level (k=0) ───────────────────────────────
DOCSETS = ["heldout-wmt23-en-de", "heldout-wmt24-en-de", "heldout-wmt24-en-es",
           "heldout-wmt24-ja-zh", "wmt25-cs-de", "wmt25-en-sr", "wmt25-en-uk", "wmt25-en-zh"]
SETLAB = ["wmt23\nen-de", "wmt24\nen-de", "wmt24\nen-es", "wmt24\nja-zh",
          "wmt25\ncs-de", "wmt25\nen-sr", "wmt25\nen-uk", "wmt25\nen-zh"]
M = np.array([[held[m][s]["0"]["kendall"] for s in DOCSETS] for m in ORDER])
means = M.mean(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(8.0, 3.6))
full = np.hstack([M, np.full((len(ORDER), 1), np.nan), means])
im = ax.imshow(full, cmap=SEQ, vmin=0.10, vmax=0.45, aspect="auto")
for i in range(full.shape[0]):
    for j in range(full.shape[1]):
        v = full[i, j]
        if np.isnan(v):
            continue
        col = "white" if v > 0.33 else INK
        ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                fontsize=7.2, color=col,
                fontweight="bold" if j == full.shape[1] - 1 else "normal")
ax.set_xticks(range(full.shape[1]))
ax.set_xticklabels(SETLAB + ["", "moy."], fontsize=7.5)
ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels([PRETTY[m] for m in ORDER], fontsize=7.8)
for lab, m in zip(ax.get_yticklabels(), ORDER):
    lab.set_color(BLUE if m.startswith("da") else ORANGE)
ax.grid(False)
ax.set_title(r"Documents natifs — Kendall $\tau$ par jeu de test", fontsize=9.5, pad=8)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=7, color=MUTED)
note(fig,
     "Évalué sur — les 8 jeux document hors entraînement : paragraphes MQM WMT23/24 (1 000 items) et "
     "documents ESA WMT25 (413–559).", y=-0.05)
save(fig, "q1_docsets")

# ════════════════════════════════════════════════════════════════════════════
# Q2 — MetaDocEval (Dahan, Bawden & Yvon, EAMT 2026)
# ════════════════════════════════════════════════════════════════════════════
mde = json.load(open(ROOT / "results/length_training/metadoceval.json"))
MM = mde["models"]
COUNTS = mde["counts_per_perturbation"]
W = ["1", "3", "6", "9"]
LEXICAL = ["tense_consistency", "lexical_consistency", "conjunction_substitution"]
STRUCT = ["sentence_removal", "sentence_repetition", "sentence_shuffling"]
CATLAB = {
    "tense_consistency": "temps verbaux", "lexical_consistency": "cohérence lexicale",
    "conjunction_substitution": "connecteurs", "pronoun_swap_singular": "pronom sg.",
    "pronoun_swap_plural": "pronom pl.", "sentence_repetition": "répétition",
    "sentence_removal": "suppression", "sentence_shuffling": "permutation",
    "sentence_splitting": "découpage (TFP)",
}


def acc(model, cat, w):
    return MM[model]["micro"][f"{cat}|w{w}"]["accuracy"]


# ── Fig 4: baselines, accuracy vs context window ────────────────────────────
GROUPS = [("tense_consistency", BLUE), ("lexical_consistency", ORANGE),
          ("conjunction_substitution", AQUA), ("sentence_removal", GREYS[0]),
          ("sentence_repetition", GREYS[1]), ("sentence_shuffling", GREYS[2]),
          ("sentence_splitting", MAGENTA)]
NUDGE = {"da-base": {"sentence_repetition": 7, "sentence_removal": -1,
                     "sentence_shuffling": -10, "lexical_consistency": 5,
                     "tense_consistency": -6, "conjunction_substitution": 1},
         "qe-base": {"sentence_shuffling": 5, "sentence_removal": -2,
                     "sentence_repetition": -8, "tense_consistency": -6,
                     "conjunction_substitution": 5}}
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1), sharey=True)
for ax, model, title in ((axes[0], "da-base", "COMET-DA (publié)"),
                         (axes[1], "qe-base", "CometKiwi (publié)")):
    for cat, c in GROUPS:
        ys = [acc(model, cat, w) for w in W]
        ax.plot(range(4), ys, ":" if cat == "sentence_splitting" else "-",
                color=c, lw=1.4 if c in GREYS else 2, marker="o", ms=4,
                markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.annotate(CATLAB[cat], (3, ys[-1]), textcoords="offset points",
                    xytext=(6, NUDGE[model].get(cat, 0)), fontsize=7, color=c, va="center")
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.text(-0.1, 0.505, "hasard", fontsize=6.5, color=MUTED, va="bottom")
    ax.set_xticks(range(4)); ax.set_xticklabels(W)
    ax.set_xlim(-0.15, 4.9)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("fenêtre de contexte $w$ (SLIDE($w$,1))")
axes[0].set_ylabel("exactitude contrastive")
axes[0].set_ylim(0.45, 1.04)
fig.tight_layout()
note(fig,
     "Évalué sur — MetaDocEval : 8 705 paires original / perturbé (documents WMT24++ en→fr/es/de, AYA23 et "
     "Gemini-1.5-Pro). Exactitude = P[score(original) > score(perturbé)], hasard .50.\n"
     "Gris = perturbations structurelles, saturées. Pointillé = découpage, qui préserve la qualité : c'est un "
     "taux de faux positifs.", y=-0.04)
save(fig, "q2_baselines")

# ── Fig 5: models × perturbations at w=1 ────────────────────────────────────
CATS = LEXICAL + ["pronoun_swap_singular", "pronoun_swap_plural"] + STRUCT + ["sentence_splitting"]
A = np.array([[acc(m, c, "1") for c in CATS] for m in ORDER])
fig, ax = plt.subplots(figsize=(8.2, 4.1))
im = ax.imshow(A, cmap=SEQ, vmin=0.5, vmax=1.0, aspect="auto")
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        ax.text(j, i, f"{A[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                fontsize=7.2, color="white" if A[i, j] > 0.80 else INK)
ax.set_xticks(range(len(CATS)))
ax.set_xticklabels([f"{CATLAB[c]}\nn={COUNTS[c]['docs']}" for c in CATS],
                   fontsize=7.2, rotation=38, ha="right", rotation_mode="anchor")
for lab, c in zip(ax.get_xticklabels(), CATS):
    lab.set_color(MAGENTA if c == "sentence_splitting" else
                  (INK2 if c in LEXICAL + ["pronoun_swap_singular", "pronoun_swap_plural"] else GREYS[0]))
ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels([PRETTY[m] for m in ORDER], fontsize=7.8)
for lab, m in zip(ax.get_yticklabels(), ORDER):
    lab.set_color(BLUE if m.startswith("da") else ORANGE)
for x in (2.5, 4.5, 7.5):
    ax.axvline(x, color=SURFACE, lw=2.5)
ax.axhline(4.5, color=SURFACE, lw=2.5)
ax.grid(False)
ax.set_title("MetaDocEval — exactitude contrastive à $w{=}1$ (micro, 9 perturbations)",
             fontsize=9.5, pad=8)
cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
note(fig,
     "Évalué sur — les mêmes paires, à fenêtre d'un segment. n = documents mesurés par catégorie ; les deux "
     "catégories pronom (n=19) sont trop petites pour être lues. Colonne magenta = taux de faux positifs, "
     "plus bas est mieux.", y=-0.05)
fig.tight_layout()
save(fig, "q2_heatmap_w1")

# ── Fig 6: does context help? lexical mean vs w ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1), sharey=True)
for ax, fam, title in ((axes[0], "da", "COMET-DA (avec référence)"),
                       (axes[1], "qe", "CometKiwi (sans référence)")):
    base = [np.mean([acc(f"{fam}-base", c, w) for c in LEXICAL]) for w in W]
    ax.plot(range(4), base, "-o", color=MUTED, lw=2.4, ms=5, zorder=4)
    ax.annotate("publié", (3, base[-1]), textcoords="offset points", xytext=(6, 0),
                fontsize=7.5, color=INK2, va="center")
    for suffix, c, ls in ARMS:
        ys = [np.mean([acc(f"{fam}-{suffix}", c2, w) for c2 in LEXICAL]) for w in W]
        ax.plot(range(4), ys, ls, color=c, lw=2, marker="o", ms=4,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=5)
    ax.set_xticks(range(4)); ax.set_xticklabels(W)
    ax.set_xlim(-0.15, 3.75)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("fenêtre de contexte $w$")
axes[0].set_ylabel("exactitude moyenne\n(3 catégories lexicales)")
fig.legend(handles, ["métrique publiée", "ré-entraîné sur texte long", "texte long, encodeur gelé",
                     "ré-entraîné sur phrases", "phrases, encodeur gelé"],
           loc="lower center", ncol=5, frameon=False, fontsize=7.8,
           bbox_to_anchor=(0.5, -0.10))
fig.tight_layout()
note(fig,
     "Évalué sur — les 3 catégories lexicales de MetaDocEval (404 / 356 / 562 documents), micro-moyennées. "
     "Les structurelles sont saturées et ne discriminent pas.", y=-0.14)
save(fig, "q2_context")

# ════════════════════════════════════════════════════════════════════════════
# Q3 — encoders on the FLORES+ concatenated-block framework
#
# The number of candidates a source block is compared against is a PARAMETER,
# fixed by design — never a by-product of k (a k=5 block admits more possible
# perturbations than a k=2 block, so a pool that grows with k would confound
# length with task difficulty). Two fixed-candidate protocols:
#   duel D=5 (run_duel.py)  — gold vs EXACTLY 5 single-error negatives; the
#       win probability is averaged in closed form over ALL C(m, 5) ways of
#       drawing 5 negatives from the block's m available ones: C(w,5)/C(m,5).
#       Blocks with m < 5 are skipped, so every scored duel is equally hard.
#   duel D=1 (run_xsim.py `detection_rate`) — gold vs one negative, every
#       negative used, coverage 100 %.
# `xsimpp_err` (pool = all true blocks + all their perturbed copies) is NOT
# plotted here: that pool grows with k, which is the confound the duel removes.
# ════════════════════════════════════════════════════════════════════════════
duel = json.load(open(ROOT / "results/block_duel/block_duel.json"))["encoders"]
enc = {}
for f in ["results/block_xsim/block_xsim_wave1_frac000.json",
          "results/block_xsim/block_xsim_frac100.json"]:
    enc.update(json.load(open(ROOT / f))["encoders"])
BK = ["2", "3", "4", "5"]
D = "5"                                   # negatives per duel — a free parameter
CHANCE_D = int(D) / (int(D) + 1)          # gold ranked at random among D+1 candidates
LANGS = ["de", "es", "fr", "ru"]

PUBLIC = [("e5:multilingual-e5-base", AQUA, "E5 multilingue"),
          ("labse", ORANGE, "LaBSE"),
          ("comet:wmt22-comet-da", BLUE, "encodeur COMET"),
          ("xlmr:xlm-roberta-large", VIOLET, "XLM-R (mean-pool)")]
COMETS = [("comet:wmt22-comet-da", MUTED, "-", "COMET publié"),
          ("comet:4yeqp7cn-last", MUTED, "--", "+ finetuning Bio-MQM"),
          ("comet:mw5cryt7-epoch=5-step=4500-val_kendall=0.311", BLUE, "-", "+ texte long"),
          ("comet:4cnnyi3x-epoch=5-step=4500-val_kendall=0.310", ORANGE, "-", "+ phrases")]


def duel_err(key, k, d=D):
    return duel[key]["_mean_by_k"][k]["by_duel_size"][d]["duel_err"]


def coverage(k, d=D):
    return float(np.mean([duel["labse"][L][k]["by_duel_size"][d]["coverage"] for L in LANGS]))


def n_used(k, d=D):
    return int(sum(duel["labse"][L][k]["by_duel_size"][d]["n_used"] for L in LANGS))


# ── Fig 7: the duel — the same number of candidates at every k ──────────────
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1),
                         gridspec_kw={"width_ratios": [2.0, 1]})
ax = axes[0]
ax.axhspan(CHANCE_D, 1.05, color="#f6f3ee", zorder=0)
ax.axhline(CHANCE_D, color=MUTED, lw=1, ls="--")
ax.text(3.05, CHANCE_D + 0.012, f"hasard = {CHANCE_D:.2f}", fontsize=6.8,
        color=MUTED, va="bottom")
for key, c, lab in PUBLIC:
    ys = [duel_err(key, k) for k in BK]
    ax.plot(range(4), ys, "-o", color=c, lw=2, ms=5,
            markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.annotate(lab, (3, ys[-1]), textcoords="offset points",
                xytext=(6, {"xlmr:xlm-roberta-large": 8,
                            "comet:wmt22-comet-da": -6}.get(key, 0)),
                fontsize=7.3, color=c, va="center")
ax.set_xticks(range(4))
ax.set_xticklabels([f"k={k}" for k in BK])
ax.set_xlim(-0.15, 5.4)
ax.set_ylim(0.15, 1.02)
ax.set_xlabel("phrases concaténées par bloc")
ax.set_ylabel("erreur du duel  ↓")
ax.set_title("Duel à 5 candidats — le bloc correct contre 5 négatifs", fontsize=9.5)

ax = axes[1]
covs = [coverage(k) for k in BK]
ax.bar(range(4), covs, width=0.62, color=BLUES[2], zorder=3)
for x, c in enumerate(covs):
    ax.text(x, c + 0.025, f"{c:.0%}", ha="center", fontsize=7.2, color=INK2)
ax.set_xticks(range(4))
ax.set_xticklabels([f"k={k}\nn={n_used(k)}" for k in BK], fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_yticks([0, 0.5, 1.0])
ax.set_yticklabels(["0", "50 %", "100 %"])
ax.grid(axis="x", visible=False)
ax.set_xlabel("phrases concaténées par bloc")
ax.set_title("Blocs éligibles (≥ 5 négatifs)", fontsize=9.5)
fig.tight_layout()
note(fig,
     "Évalué sur — FLORES+ (de/es/fr/ru, en→xx). Chaque duel oppose le bloc correct à exactement 5 négatifs à "
     "une erreur, moyenné en forme close sur tous les tirages C(m,5) : le budget de candidats ne dépend pas de k.\n"
     "Réserve : il faut m ≥ 5 négatifs disponibles, d'où 14 % de blocs éligibles à k=2 contre 72 % à k=5.",
     y=-0.06)
save(fig, "q3_duel")

# ── Fig 8: the 1-vs-1 duel, full coverage, with the retrained encoders ──────
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
ax = axes[0]
for key, c, lab in PUBLIC:
    ys = [enc[key]["_mean_by_k"][k]["detection_rate"] for k in BK]
    ax.plot(range(4), ys, "-o", color=c, lw=2, ms=5,
            markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.annotate(lab, (3, ys[-1]), textcoords="offset points", xytext=(6, 0),
                fontsize=7.3, color=c, va="center")
ax.set_title("Encodeurs publics", fontsize=9.5)
ax = axes[1]
for key, c, ls, lab in COMETS:
    ys = [enc[key]["_mean_by_k"][k]["detection_rate"] for k in BK]
    ax.plot(range(4), ys, ls, color=c, lw=2, marker="o", ms=4.5,
            markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.annotate(lab, (3, ys[-1]), textcoords="offset points",
                xytext=(6, {"comet:4cnnyi3x-epoch=5-step=4500-val_kendall=0.310": 12,
                            "comet:mw5cryt7-epoch=5-step=4500-val_kendall=0.311": 3,
                            "comet:wmt22-comet-da": -6,
                            "comet:4yeqp7cn-last": -15}.get(key, 0)),
                fontsize=7.3, color=c, va="center")
ax.set_title("Variantes de l'encodeur COMET (même échelle)", fontsize=9.5)
for ax in axes:
    ax.axhspan(0, 0.5, color="#f6f3ee", zorder=0)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"k={k}" for k in BK])
    ax.set_xlim(-0.15, 5.1)
    ax.set_xlabel("phrases concaténées par bloc")
axes[0].set_ylim(0.25, 1.0)
axes[0].set_ylabel("détection  ↑")
axes[0].text(3.45, 0.505, "hasard = .50", fontsize=6.8, color=MUTED, va="bottom")
fig.tight_layout()
note(fig,
     "Évalué sur — les mêmes blocs, en duel à 1 candidat : le bloc correct contre une seule copie à une erreur. "
     "Tous les négatifs servent, donc couverture 100 % et aucune sélection de blocs — au prix d'une tâche plus "
     "facile (hasard .50). « + texte long » / « + phrases » : encodeur COMET ré-entraîné.", y=-0.06)
save(fig, "q3_detection")

# ── Fig 9: matched core — length isolated ───────────────────────────────────
mc = json.load(open(ROOT / "results/matched_core/matched_core.json"))["encoders"]
LS = ["60", "120", "240", "480"]
MC_ENC = [("e5:multilingual-e5-base", AQUA, "E5"), ("labse", ORANGE, "LaBSE"),
          ("comet:wmt22-comet-da", BLUE, "COMET"), ("xlmr:xlm-roberta-large", VIOLET, "XLM-R")]
PHIS = [("inert", "remplissage inerte", "phrase neutre répétée — aucun sens ajouté"),
        ("neutral", "remplissage neutre", "phrases FLORES sans rapport"),
        ("distractor", "remplissage distracteur", "phrases du même article"),
        ("natural", "remplissage naturel", "texte contigu du corpus")]
fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.9), sharey=True)
for ax, (phi, title, sub) in zip(axes, PHIS):
    for key, c, lab in MC_ENC:
        ys = [mc[key]["_mean"][phi][L]["detection"] for L in LS]
        ax.plot(range(4), ys, "-o", color=c, lw=2, ms=4.5,
                markeredgecolor=SURFACE, markeredgewidth=0.8)
        if phi == "natural":
            ax.annotate(lab, (3, ys[-1]), textcoords="offset points", xytext=(6, 0),
                        fontsize=7.5, color=c, va="center")
    ax.axhspan(0, 0.5, color="#f6f3ee", zorder=0)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.set_xticks(range(4))
    ax.set_xticklabels(LS, fontsize=8)
    ax.set_title(f"{title}\n", fontsize=9, linespacing=1.6)
    ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=6.8, color=MUTED)
    ax.set_xlabel("longueur totale $L$ (tokens)")
axes[0].set_ylim(0.28, 1.02)
axes[0].set_ylabel("taux de détection")
axes[3].set_xlim(-0.15, 3.9)
fig.tight_layout()
note(fig,
     "Évalué sur — 120 phrases-cœurs par langue (de/es/fr/ru), replacées dans un remplissage qui porte l'entrée "
     "à L tokens. Duel à 1 candidat, propre contre perturbé, remplissage identique octet pour octet ; tout reste "
     "sous 512 tokens.", y=-0.06)
save(fig, "q3_matched_core")

# ── Fig 10: length-sensitivity index ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.6, 3.0))
width = 0.2
xs = np.arange(len(PHIS))
for i, (key, c, lab) in enumerate(MC_ENC):
    vals = [mc[key]["_lsi"][p]["lsi"] for p, _, _ in PHIS]
    lo = [v - mc[key]["_lsi"][p]["ci_lo"] for v, (p, _, _) in zip(vals, PHIS)]
    hi = [mc[key]["_lsi"][p]["ci_hi"] - v for v, (p, _, _) in zip(vals, PHIS)]
    ax.bar(xs + (i - 1.5) * width, vals, width * 0.88, color=c, zorder=3,
           yerr=[np.abs(lo), np.abs(hi)], error_kw=dict(ecolor=INK2, lw=0.9, capsize=2))
    if key.startswith("comet:") or key == "labse":
        for x, v, lo_ in zip(xs + (i - 1.5) * width, vals,
                             [mc[key]["_lsi"][p_]["ci_lo"] for p_, _, _ in PHIS]):
            ax.text(x, min(v, lo_) - 0.005, f"{v:.3f}".replace("-0.", "−."),
                    ha="center", va="top", fontsize=6.8, color=INK2)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xticks(xs)
ax.set_xticklabels([t for _, t, _ in PHIS], fontsize=8.5)
ax.set_ylabel("LSI — Δ détection\npar doublement de $L$")
ax.set_ylim(-0.165, 0.012)
ax.grid(axis="x", visible=False)
h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in MC_ENC]
ax.legend(h, [l for _, _, l in MC_ENC], frameon=False, fontsize=8, ncol=4,
          loc="lower center", bbox_to_anchor=(0.5, -0.26))
ax.set_title("Coût de la longueur, en un chiffre par encodeur", fontsize=9.5)
fig.tight_layout()
note(fig,
     "Pente intra-item de la détection par doublement de L, IC 95 % bootstrap sur les items. Mêmes données que "
     "la figure précédente.", y=-0.10)
save(fig, "q3_lsi")

print(f"\n10 figures → {OUT}")
