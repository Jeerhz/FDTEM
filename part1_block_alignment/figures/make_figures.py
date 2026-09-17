"""Report figures of part 1 — isolating length on FLORES+ blocks.

    python -m part1_block_alignment.figures.make_figures [--out_dir DIR]

Reads results/duel.json (evaluate_duel.py; encoder cosine, D in {4,5,6}) and
results/comet_score.json (evaluate_comet_score.py; COMET score vs cosine) and
writes nine PDF+PNG pairs into figures/: iso_* (the isolation figures) and
align_* (the alignment-rule figures).

Two quantities, one vocabulary, used everywhere in the report:

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
"""
from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from part1_block_alignment import FIGURES_DIR, RESULTS_DIR
from part1_block_alignment.models import DuelMetrics, DuelRunResult, ScoreCells, ScoreRunResult

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
K1 = ["1", "2", "3", "4", "5"]
DS = [4, 5, 6]
CATLAB = {"causality": "causalité", "entity": "entité", "number": "nombre"}

# encoder key in duel.json -> (label, colour, linestyle)
ENC = [
    ("e5:multilingual-e5-base", "mE5", AQUA, "-"),
    ("labse", "LaBSE", ORANGE, "-"),
    ("comet:wmt22-comet-da", "encodeur COMET-DA", BLUE, "-"),
    ("xlmr:xlm-roberta-large", "XLM-R (moyenne)", VIOLET, "--"),
]
# scorer key in comet_score.json -> (label, colour, linestyle, alpha)
SCORERS = [
    ("comet-score:comet:wmt22-cometkiwi-da", "score CometKiwi", ORANGE, "-", 1.0),
    ("encoder-cos:comet:wmt22-comet-da", "cosinus · encodeur COMET-DA", BLUE, "--", 1.0),
    ("encoder-cos:comet:wmt22-cometkiwi-da", "cosinus · encodeur CometKiwi", ORANGE, "--", 0.5),
]
# arm directory (from the checkpoint path) -> what the arm was trained on
ARM_LABEL = {"frac100": "phrases", "frac000agg": "phrases concaténées",
             "frac000nat": "documents natifs", "frac000": "mixte",
             "uncontrolled": "non contrôlé"}


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def note(fig, text, y=-0.04, width=130) -> None:
    wrapped = "\n".join(textwrap.fill(par, width) for par in text.split("\n"))
    fig.text(0.0, y, wrapped, fontsize=7, color=MUTED, ha="left", va="top",
             linespacing=1.5, transform=fig.transFigure)


def legend(fig, entries, y=-0.16, ncol=4) -> None:
    handles = [Line2D([], [], color=c, ls=ls, lw=2, marker="o", ms=4.5,
                      markeredgecolor=SURFACE, markeredgewidth=0.8)
               for _lab, c, ls in entries]
    fig.legend(handles, [lab for lab, _c, _ls in entries], loc="lower center",
               ncol=ncol, frameon=False, fontsize=8, bbox_to_anchor=(0.5, y))


# ── isolation figures (duel.json + comet_score.json) ─────────────────────────
def isolation_figures(duel: DuelRunResult, score: ScoreRunResult, out: Path) -> None:
    def cell(key, k) -> DuelMetrics:
        return duel.models[key].mean_by_k[k]

    def success(key, k, D):
        return 1.0 - cell(key, k).by_duel_size[str(D)].duel_err

    def rank(key, k, D):
        """Mean rank of the reference among D+1 candidates, from the pairwise rate."""
        return 1.0 + D * (1.0 - cell(key, k).detection_rate)

    # Fig 1 — the data: how many blocks, and how many perturbed blocks
    sc0 = next(iter(score.models.values()))
    langs = list(sc0.by_lang)
    n_blocks = [sc0.by_lang[langs[0]][k].n_blocks for k in K1]          # identical per language
    n_max = [2009 // int(k) for k in K1]
    by_cat = {c: [sum(sc0.by_lang[l][k].variant_stats[c] for l in langs) for k in K1]
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
    save(fig, out, "iso_data")

    # Fig 2 — success rate against k, one panel per D
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
    save(fig, out, "iso_success_vs_k")

    # Fig 3 — success rate against D, one panel per k
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
    save(fig, out, "iso_success_vs_D")

    # Fig 4 — mean rank of the reference against D, one panel per k
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
    save(fig, out, "iso_rank_vs_D")

    # Fig 5 — per perturbation category (pairwise win rate, full coverage)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=True)
    for ax, cat in zip(axes, ("causality", "entity", "number")):
        for key, lab, col, ls in ENC:
            ys = [cell(key, k).beat_by_category[cat] for k in KS]
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
    save(fig, out, "iso_by_category")

    # Fig 6 — coverage: which blocks can supply D distractors
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for D, col in zip(DS, (AQUA, BLUE, VIOLET)):
        ys = [cell(ENC[0][0], k).by_duel_size[str(D)].coverage for k in KS]
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
    save(fig, out, "iso_coverage")

    # Fig 7 — aligning with the COMET score instead of the encoder cosine
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.3))
    ax = axes[0]
    for key, lab, col, ls, al in SCORERS:
        ys = [score.models[key].mean_by_k[k].shortlist_accuracy for k in K1]
        ax.plot(range(len(K1)), ys, ls, color=col, lw=2, marker="o", ms=4.5,
                markeredgecolor=SURFACE, markeredgewidth=0.8, alpha=al, zorder=4)
    ax.axhline(1 / 3, color=MUTED, lw=1, ls=":")
    ax.text(0.05, 1 / 3 + 0.02, "hasard = .33", fontsize=6.8, color=MUTED)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("taux de réussite  ↑")
    ax.set_title("Taux de réussite, D = 2 distracteurs", fontsize=9.5)

    ax = axes[1]
    for key, lab, col, ls, al in SCORERS:
        ys = [score.models[key].mean_by_k[k].shortlist_mean_rank for k in K1]
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
    save(fig, out, "iso_comet_score")


# ── alignment figures (comet_score.json, every scorer incl. retrained arms) ──
def describe(name: str, spec: str = "") -> dict:
    """`comet-score:comet:wmt22-cometkiwi-da` -> how to draw and label it.

    For a retrained arm the JSON key only carries the W&B run id, which says
    nothing to a reader. The arm directory is in the checkpoint path, so the
    label is derived from `spec` instead — no run-id table to keep in sync.
    """
    rule, _, ref = name.partition(":")
    tag = ref.split(":", 1)[-1]
    is_score = rule == "comet-score"

    if "cometkiwi" in tag:
        model, colour = "CometKiwi publié", ORANGE
    elif "comet-da" in tag:
        model, colour = "COMET-DA publié", BLUE
    else:
        fam, arm = "", ""
        for part in Path(spec.partition(":")[2]).parts:
            if part.startswith("kiwi-mix-"):
                fam, arm = "Kiwi", part[len("kiwi-mix-"):]
            elif part.startswith("mix-"):
                fam, arm = "DA", part[len("mix-"):]
        ep = re.search(r"epoch=(\d+)", tag)
        model = (f"{fam} · {ARM_LABEL.get(arm, arm)}" if arm
                 else f"bras {tag.split('-')[0]}")
        if ep:
            model += f" (ép. {ep.group(1)})"
        colour = ORANGE if fam == "Kiwi" else BLUE
    return {
        "label": f"{'score' if is_score else 'cosinus encodeur'} · {model}",
        "color": colour,
        "ls": "-" if is_score else "--",
        "lw": 2.3 if is_score else 1.8,
        "retrained": not ("cometkiwi" in tag or "comet-da" in tag),
        "is_score": is_score,
        # CometKiwi is a UnifiedMetric (`mt </s></s> src` as ONE sequence): it has no
        # single-text encoder, so its cosine is a degenerate use of the model.
        "degenerate": (not is_score) and "cometkiwi" in tag,
    }


def series(sc: ScoreCells, ks: list[str], key: str, cat: str | None = None):
    xs, ys = [], []
    for i, k in enumerate(ks):
        cell = sc.mean_by_k.get(k)
        if not cell:
            continue
        v = cell.duel_accuracy_by_category.get(cat) if cat else getattr(cell, key)
        if v is not None:
            xs.append(i); ys.append(v)
    return xs, ys


def align_figures(score: ScoreRunResult, out: Path) -> None:
    scorers = score.models
    ks = sorted({k for s in scorers.values() for k in s.mean_by_k}, key=int)
    size = score.config.get("shortlist_size", 3)
    styles = {n: describe(n, scorers[n].spec or "") for n in scorers}

    # order: metric scores first, then cosines — the contrast the figure is about
    order = sorted(scorers, key=lambda n: (not styles[n]["is_score"], n))

    def draw(ax, n, xs, ys):
        st = styles[n]
        ax.plot(xs, ys, color=st["color"], ls=st["ls"], lw=st["lw"],
                marker=("^" if st["retrained"] else
                        ("o" if st["is_score"] else "s")), ms=4.5,
                markerfacecolor=SURFACE if st["retrained"] else st["color"],
                markeredgecolor=st["color"], markeredgewidth=1.1,
                alpha=0.55 if st["degenerate"] else 1.0, zorder=4)

    handles = [Line2D([], [], color=styles[n]["color"], ls=styles[n]["ls"],
                      lw=styles[n]["lw"],
                      marker=("^" if styles[n]["retrained"] else
                              ("o" if styles[n]["is_score"] else "s")), ms=4.5,
                      markerfacecolor=SURFACE if styles[n]["retrained"] else styles[n]["color"],
                      markeredgecolor=styles[n]["color"],
                      alpha=0.55 if styles[n]["degenerate"] else 1.0)
               for n in order]
    labels = [styles[n]["label"] + (" ⚠" if styles[n]["degenerate"] else "")
              for n in order]

    # Fig 1: the two protocols
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    panels = [("duel_accuracy", "Duel — le bon contre UN perturbé", 0.5, "hasard .50"),
              ("shortlist_accuracy",
               f"Liste courte — le bon contre {size - 1} perturbés\n"
               f"(couverture 74 % à $k{{=}}1$ → 99 % à $k{{=}}5$)",
               1 / size, f"hasard {1/size:.2f}")]
    for ax, (key, title, chance, clab) in zip(axes, panels):
        for n in order:
            draw(ax, n, *series(scorers[n], ks, key))
        ax.axhline(chance, color=MUTED, lw=1, ls="--")
        ax.text(0.02, chance + 0.02, clab, fontsize=6.8, color=MUTED)
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks)
        ax.set_xlabel("longueur du bloc $k$ (phrases)")
        ax.set_ylim(0, 1.04)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("exactitude")
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.22))
    fig.suptitle("Aligner avec le score COMET plutôt qu'avec le cosinus de l'encodeur",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    note(fig,
         "Triangles creux = bras ré-entraîné. Candidats — le bloc de référence et ses seules variantes perturbées "
         "(une phrase du bloc modifiée : causalité, entité ou nombre). Les blocs "
         "non perturbés des autres articles sont exclus : les distinguer est une "
         "compétence différente et facile, qui gonflerait la sensibilité apparente "
         "à l'édition injectée.\n"
         "⚠ CometKiwi est un UnifiedMetric : il encode « mt </s></s> src » en une "
         "seule séquence et n'a donc pas d'encodeur de phrase isolée. Sa courbe "
         "cosinus est un usage dégénéré du modèle, tracée en transparence. Une "
         "sonde directe confirme que l'espace reste aligné (la diagonale gagne), "
         "mais avec cos ≈ .97 pour une traduction contre .95 pour une phrase sans "
         "rapport : la marge utile est de ~.02, bien trop peu pour qu'une édition "
         "d'un mot y survive.",
         y=-0.28, width=132)
    save(fig, out, "align_protocols")

    # Fig 2: per perturbation category
    cats = sorted({c for s in scorers.values()
                   for cell in s.mean_by_k.values()
                   for c, v in cell.duel_accuracy_by_category.items() if v is not None})
    fig, axes = plt.subplots(1, len(cats), figsize=(3.2 * len(cats), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cat in zip(axes, cats):
        for n in order:
            draw(ax, n, *series(scorers[n], ks, "duel_accuracy", cat))
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks)
        ax.set_xlabel("longueur du bloc $k$")
        ax.set_ylim(0, 1.04)
        ax.set_title(CATLAB.get(cat, cat), fontsize=9.5)
    axes[0].set_ylabel("exactitude du duel")
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.24))
    fig.suptitle("Par type de perturbation — le duel, à couverture complète",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    note(fig,
         "Le duel emploie tous les négatifs échantillonnés de chaque bloc, donc son "
         "effectif ne dépend pas d'un seuil — c'est la courbe à lire en premier. La "
         "liste courte, elle, écarte les blocs qui ne fournissent pas assez de "
         "négatifs propres, et une phrase isolée en admet peu : sa couverture monte "
         "de 74 % à $k{=}1$ à 99 % à $k{=}5$, donc une part de sa pente vient du "
         "changement d'échantillon et non du modèle.\n"
         "Le cosinus de l'encodeur CometKiwi n'est pas simplement peu sensible : il "
         "préfère systématiquement la version perturbée, et de plus en plus avec $k$ "
         "(entité : .24 à $k{=}1$, .03 à $k{=}5$).",
         y=-0.30, width=132)
    save(fig, out, "align_by_category")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duel", default=str(RESULTS_DIR / "duel.json"))
    ap.add_argument("--comet_score", default=str(RESULTS_DIR / "comet_score.json"))
    ap.add_argument("--out_dir", default=str(FIGURES_DIR))
    args = ap.parse_args()

    duel = DuelRunResult.load(Path(args.duel))
    score = ScoreRunResult.load(Path(args.comet_score))
    out = Path(args.out_dir)
    isolation_figures(duel, score, out)
    align_figures(score, out)
    print(f"\n9 figures -> {out}")


if __name__ == "__main__":
    main()
