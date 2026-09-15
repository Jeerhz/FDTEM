#!/usr/bin/env python3
"""Figures for part 1 of the report — isolating length on FLORES+ blocks.

Reads
    results/block_duel/block_duel.json    (run_duel.py)   encoder cosine, D ∈ {4,5,6}
    results/comet_align/comet_align.json  (run_comet_align.py)  COMET score vs cosine
and writes PDF + PNG into report/figures/isolation/.

Two metrics, one vocabulary, used everywhere in the report:

  taux de réussite A(D)   P[the reference block scores highest among the D+1
                          candidates], averaged in closed form over every
                          C(m, D) subset of the block's m negatives.
                          chance = 1/(D+1).
  rang moyen R(D)         mean rank of the reference among the D+1 candidates,
                          1 = best. chance = (D+2)/2.

For the duel run R(D) is reconstructed from the pairwise win rate p as
R = 1 + D(1-p): exact per block (E[#negatives above the gold] = D(m-w)/m), and
an approximation once p is pooled across blocks with different m. The shortlist
protocol reports R exactly (`shortlist_mean_rank`), which is what the
COMET-score figure uses.

  python report/figures/make_isolation_figures.py
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "isolation"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
VIOLET, RED = "#4a3aa7", "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8f8e8a", "#eceae4"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.major.size": 0, "ytick.major.size": 0,
})

KS = ["2", "3", "4", "5"]
DS = [4, 5, 6]
CATLAB = {"causality": "causalité", "entity": "entité", "number": "nombre"}

# encoder key in block_duel.json -> (label, colour, linestyle)
ENC = [
    ("e5:multilingual-e5-base", "mE5", AQUA, "-"),
    ("labse", "LaBSE", ORANGE, "-"),
    ("comet:wmt22-comet-da", "encodeur COMET-DA", BLUE, "-"),
    ("xlmr:xlm-roberta-large", "XLM-R (moyenne)", VIOLET, "--"),
]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def note(fig, text, y=-0.04, width=130):
    wrapped = "\n".join(textwrap.fill(par, width) for par in text.split("\n"))
    fig.text(0.0, y, wrapped, fontsize=7, color=MUTED, ha="left", va="top",
             linespacing=1.5, transform=fig.transFigure)


def legend(fig, entries, y=-0.16, ncol=4):
    handles = [Line2D([], [], color=c, ls=ls, lw=2, marker="o", ms=4.5,
                      markeredgecolor=SURFACE, markeredgewidth=0.8)
               for _lab, c, ls in entries]
    fig.legend(handles, [lab for lab, _c, _ls in entries], loc="lower center",
               ncol=ncol, frameon=False, fontsize=8, bbox_to_anchor=(0.5, y))


duel = json.load(open(ROOT / "results/block_duel/block_duel.json"))["encoders"]
align = json.load(open(ROOT / "results/comet_align/comet_align.json"))


def cell(key, k):
    return duel[key]["_mean_by_k"][k]


def success(key, k, D):
    return 1.0 - cell(key, k)["by_duel_size"][str(D)]["duel_err"]


def rank(key, k, D):
    """Mean rank of the reference among D+1 candidates, from the pairwise rate."""
    return 1.0 + D * (1.0 - cell(key, k)["detection_rate"])


# ════════════════════════════════════════════════════════════════════════════
# Fig 1 — the data: how many blocks, and how many perturbed blocks
# ════════════════════════════════════════════════════════════════════════════
sc0 = list(align["scorers"].values())[0]
LANGS = [l for l, v in sc0.items() if isinstance(v, dict) and not l.startswith("_")]
K1 = ["1", "2", "3", "4", "5"]
n_blocks = [sc0[LANGS[0]][k]["n_blocks"] for k in K1]          # identical per language
n_max = [2009 // int(k) for k in K1]
by_cat = {c: [sum(sc0[l][k]["variant_stats"][c] for l in LANGS) for k in K1]
          for c in ("causality", "entity", "number")}

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
ax = axes[0]
x = np.arange(len(K1))
ax.bar(x - 0.2, n_max, 0.38, color=GRID, zorder=2, label="maximum théorique $\\lfloor 2009/k \\rfloor$")
ax.bar(x + 0.2, n_blocks, 0.38, color=BLUE, zorder=3, label="blocs construits")
for xi, v in zip(x + 0.2, n_blocks):
    ax.text(xi, v + 30, str(v), ha="center", fontsize=7, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in K1])
ax.set_ylabel("blocs par langue")
ax.set_title("Blocs disponibles — les blocs ne traversent pas un article", fontsize=9.5)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")

ax = axes[1]
w = 0.26
for i, (c, col) in enumerate(zip(("causality", "entity", "number"), (MAGENTA, YELLOW, AQUA))):
    ax.bar(x + (i - 1) * w, by_cat[c], w * 0.9, color=col, zorder=3, label=CATLAB[c])
ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in K1])
ax.set_ylabel("blocs perturbés (4 langues)")
ax.set_title("Blocs perturbés disponibles, par type de perturbation", fontsize=9.5)
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout()
note(fig,
     "FLORES+ dev + devtest : 2009 phrases alignées par langue, de/es/fr/ru, source anglaise. "
     "Un bloc = k phrases consécutives d'un même article, sans recouvrement (pas = k). "
     "À droite : nombre de blocs perturbés produits par le générateur spaCy, avec au plus deux variantes "
     "par (position, catégorie). Un bloc long offre plus de positions, donc plus de perturbations possibles ; "
     "c'est exactement le déséquilibre que le nombre fixe de distracteurs D vient corriger.",
     y=-0.05)
save(fig, "iso_data")

# ════════════════════════════════════════════════════════════════════════════
# Fig 2 — success rate against k, one panel per D
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=True)
for ax, D in zip(axes, DS):
    for key, lab, col, ls in ENC:
        ys = [success(key, k, D) for k in KS]
        ax.plot(range(len(KS)), ys, ls, color=col, lw=2, marker="o", ms=4.5,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4)
    ax.axhline(1 / (D + 1), color=MUTED, lw=1, ls=":")
    ax.text(0.05, 1 / (D + 1) + 0.02, f"hasard = {1/(D+1):.2f}", fontsize=6.8, color=MUTED)
    ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"k={k}" for k in KS])
    ax.set_xlabel("phrases par bloc")
    ax.set_title(f"D = {D} distracteurs", fontsize=9.5)
    ax.set_ylim(0, 0.85)
axes[0].set_ylabel("taux de réussite  ↑")
legend(fig, [(l, c, ls) for _k, l, c, ls in ENC], y=-0.14)
fig.tight_layout()
note(fig,
     "Taux de réussite = part des décisions où le bloc de référence obtient le meilleur score parmi les D+1 "
     "candidats. Moyenne exacte sur tous les sous-ensembles de D distracteurs, puis moyenne sur de/es/fr/ru. "
     "Un bloc qui ne fournit pas D distracteurs est écarté (voir la figure de couverture).",
     y=-0.20)
save(fig, "iso_success_vs_k")

# ════════════════════════════════════════════════════════════════════════════
# Fig 3 — success rate against D, one panel per k
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.0), sharey=True)
for ax, k in zip(axes, KS):
    for key, lab, col, ls in ENC:
        ax.plot(range(len(DS)), [success(key, k, D) for D in DS], ls, color=col,
                lw=2, marker="o", ms=4.5, markeredgecolor=SURFACE,
                markeredgewidth=0.8, zorder=4)
    ax.plot(range(len(DS)), [1 / (D + 1) for D in DS], ":", color=MUTED, lw=1.2)
    ax.set_xticks(range(len(DS))); ax.set_xticklabels([f"D={D}" for D in DS])
    ax.set_xlabel("distracteurs par décision")
    ax.set_title(f"blocs de k = {k} phrases", fontsize=9.5)
    ax.set_ylim(0, 0.85)
axes[0].set_ylabel("taux de réussite  ↑")
axes[-1].text(0.05, 0.215, "hasard", fontsize=6.8, color=MUTED)
legend(fig, [(l, c, ls) for _k, l, c, ls in ENC], y=-0.15)
fig.tight_layout()
note(fig,
     "Même quantité que la figure précédente, lue dans l'autre sens : à longueur fixée, ce que coûte "
     "l'ajout d'un distracteur. La ligne pointillée est le hasard, 1/(D+1).",
     y=-0.21)
save(fig, "iso_success_vs_D")

# ════════════════════════════════════════════════════════════════════════════
# Fig 4 — mean rank of the reference against D, one panel per k
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.0), sharey=True)
for ax, k in zip(axes, KS):
    for key, lab, col, ls in ENC:
        ax.plot(range(len(DS)), [rank(key, k, D) for D in DS], ls, color=col,
                lw=2, marker="o", ms=4.5, markeredgecolor=SURFACE,
                markeredgewidth=0.8, zorder=4)
    ax.plot(range(len(DS)), [(D + 2) / 2 for D in DS], ":", color=MUTED, lw=1.2)
    ax.set_xticks(range(len(DS))); ax.set_xticklabels([f"D={D}" for D in DS])
    ax.set_xlabel("distracteurs par décision")
    ax.set_title(f"blocs de k = {k} phrases", fontsize=9.5)
    ax.set_ylim(1, 5.2)
axes[0].set_ylabel("rang moyen de la référence  ↓")
axes[-1].text(0.08, 2.66, "hasard", fontsize=6.8, color=MUTED)
legend(fig, [(l, c, ls) for _k, l, c, ls in ENC], y=-0.15)
fig.tight_layout()
note(fig,
     "Rang moyen de la référence parmi les D+1 candidats, 1 = meilleur. Reconstruit à partir du taux de "
     "victoire par paire p : R = 1 + D(1-p). Le hasard vaut (D+2)/2. Un rang au-dessus du hasard signifie "
     "que l'encodeur préfère activement les copies perturbées.",
     y=-0.21)
save(fig, "iso_rank_vs_D")

# ════════════════════════════════════════════════════════════════════════════
# Fig 5 — per perturbation category (pairwise win rate, full coverage)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=True)
for ax, cat in zip(axes, ("causality", "entity", "number")):
    for key, lab, col, ls in ENC:
        ys = [cell(key, k)["beat_by_category"][cat] for k in KS]
        ax.plot(range(len(KS)), ys, ls, color=col, lw=2, marker="o", ms=4.5,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4)
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.axhspan(0, 0.5, color="#f6f3ee", zorder=0)
    ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"k={k}" for k in KS])
    ax.set_xlabel("phrases par bloc")
    ax.set_title(CATLAB[cat], fontsize=9.5)
    ax.set_ylim(0.25, 1.0)
axes[0].set_ylabel("taux de victoire par paire  ↑")
axes[0].text(0.05, 0.515, "hasard = .50", fontsize=6.8, color=MUTED)
legend(fig, [(l, c, ls) for _k, l, c, ls in ENC], y=-0.14)
fig.tight_layout()
note(fig,
     "Taux de victoire par paire : part des couples (référence, distracteur) où la référence est mieux notée. "
     "Tous les distracteurs disponibles servent, donc la couverture est de 100 % et aucune sélection de blocs "
     "n'intervient. Zone grisée : sous le hasard, la copie perturbée est préférée.",
     y=-0.20)
save(fig, "iso_by_category")

# ════════════════════════════════════════════════════════════════════════════
# Fig 6 — coverage: which blocks can supply D distractors
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.6, 3.2))
for D, col in zip(DS, (AQUA, BLUE, VIOLET)):
    ys = [cell(ENC[0][0], k)["by_duel_size"][str(D)]["coverage"] for k in KS]
    ax.plot(range(len(KS)), ys, "-o", color=col, lw=2, ms=4.5,
            markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.annotate(f"D = {D}", (len(KS) - 1, ys[-1]), textcoords="offset points",
                xytext=(6, 0), fontsize=7.5, color=col, va="center")
ax.set_xticks(range(len(KS))); ax.set_xticklabels([f"k={k}" for k in KS])
ax.set_xlim(-0.15, 3.6)
ax.set_ylim(0, 1.0)
ax.set_xlabel("phrases par bloc")
ax.set_ylabel("part des blocs éligibles")
ax.set_title("Couverture : blocs offrant au moins D distracteurs", fontsize=9.5)
fig.tight_layout()
note(fig,
     "Un bloc de k phrases admet au plus 3k distracteurs (k positions × 3 catégories), et moins si une phrase "
     "ne contient ni entité ni valeur numérique. À k=2 et D=6, seuls 3 % des blocs sont éligibles : les points "
     "correspondants portent sur un échantillon particulier, pas sur le corpus.",
     y=-0.06)
save(fig, "iso_coverage")

# ════════════════════════════════════════════════════════════════════════════
# Fig 7 — aligning with the COMET score instead of the encoder cosine
# ════════════════════════════════════════════════════════════════════════════
SCORERS = [
    ("comet-score:comet:wmt22-cometkiwi-da", "score CometKiwi", ORANGE, "-", 1.0),
    ("encoder-cos:comet:wmt22-comet-da", "cosinus · encodeur COMET-DA", BLUE, "--", 1.0),
    ("encoder-cos:comet:wmt22-cometkiwi-da", "cosinus · encodeur CometKiwi", ORANGE, "--", 0.5),
]
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.3))
ax = axes[0]
for key, lab, col, ls, al in SCORERS:
    ys = [align["scorers"][key]["_mean_by_k"][k]["shortlist_accuracy"] for k in K1]
    ax.plot(range(len(K1)), ys, ls, color=col, lw=2, marker="o", ms=4.5,
            markeredgecolor=SURFACE, markeredgewidth=0.8, alpha=al, zorder=4)
ax.axhline(1 / 3, color=MUTED, lw=1, ls=":")
ax.text(0.05, 1 / 3 + 0.02, "hasard = .33", fontsize=6.8, color=MUTED)
ax.set_ylim(0, 1.04)
ax.set_ylabel("taux de réussite  ↑")
ax.set_title("Taux de réussite, D = 2 distracteurs", fontsize=9.5)

ax = axes[1]
for key, lab, col, ls, al in SCORERS:
    ys = [align["scorers"][key]["_mean_by_k"][k]["shortlist_mean_rank"] for k in K1]
    ax.plot(range(len(K1)), ys, ls, color=col, lw=2, marker="o", ms=4.5,
            markeredgecolor=SURFACE, markeredgewidth=0.8, alpha=al, zorder=4)
ax.axhline(2.0, color=MUTED, lw=1, ls=":")
ax.text(0.05, 2.03, "hasard = 2.0", fontsize=6.8, color=MUTED)
ax.set_ylim(1, 3.05)
ax.set_ylabel("rang moyen de la référence  ↓")
ax.set_title("Rang moyen, D = 2 distracteurs", fontsize=9.5)
for ax in axes:
    ax.set_xticks(range(len(K1))); ax.set_xticklabels([f"k={k}" for k in K1])
    ax.set_xlabel("phrases par bloc")
legend(fig, [(l, c, ls) for _k, l, c, ls, _a in SCORERS], y=-0.14, ncol=3)
fig.tight_layout()
note(fig,
     "Mêmes blocs, mêmes distracteurs, seule la règle de décision change : argmax du score COMET contre argmax "
     "du cosinus entre plongements. Le rang moyen est ici calculé exactement, décision par décision. "
     "CometKiwi encode « mt </s></s> src » en une seule séquence : son cosinus est un usage dégénéré du modèle, "
     "tracé en transparence, et il reste sous le hasard à tout k. Couverture : 87 % des blocs à k=1 (une phrase "
     "isolée ne fournit pas toujours deux distracteurs), 100 % dès k=3.",
     y=-0.20)
save(fig, "iso_comet_score")

print(f"\n7 figures → {OUT}")
