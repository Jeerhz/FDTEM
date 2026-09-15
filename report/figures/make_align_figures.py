#!/usr/bin/env python3
"""Figures for the alignment experiment: retrieving a translation with the COMET
SCORE versus with the cosine of the encoder it is built on.

Reads results/comet_align/comet_align.json (run_comet_align.py) and writes
PDF + PNG into report/figures/align/.

The candidate set is the reference block and ITS OWN perturbed variants — no
other-block distractors — so nothing separates the candidates but the injected
single-sentence edit.

One caveat is carried into the figure itself rather than left to a caption:
CometKiwi is a UnifiedMetric, which encodes `mt </s></s> src` as ONE sequence
and therefore has no single-text sentence encoder. Its `encoder-cos` curve is a
degenerate use of the model, and is drawn hatched. A direct probe (3 en/de pairs)
confirms the space is still topically aligned — the diagonal wins every row — but
with cos ≈ 0.97 on a translation and 0.95 on an unrelated sentence, the usable
margin is ~0.02, far below what a one-word edit needs to survive.

  python report/figures/make_align_figures.py
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]

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

CATLAB = {"causality": "causalité", "entity": "entité", "number": "nombre"}


# arm directory (from the checkpoint path) -> what the arm was trained on
ARM_LABEL = {"frac100": "phrases", "frac000agg": "phrases concaténées",
             "frac000nat": "documents natifs", "frac000": "mixte",
             "uncontrolled": "non contrôlé"}


def describe(name: str, spec: str = "") -> dict:
    """`comet-score:comet:wmt22-cometkiwi-da` → how to draw and label it.

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
        # CometKiwi has no single-text encoder — see the module docstring.
        "degenerate": (not is_score) and "cometkiwi" in tag,
    }


def note(fig, text, y=-0.05, width=132):
    wrapped = "\n".join(textwrap.fill(p, width) for p in text.split("\n"))
    fig.text(0.0, y, wrapped, fontsize=7, color=MUTED, ha="left", va="top",
             linespacing=1.5, transform=fig.transFigure)


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def series(sc: dict, ks: list[str], key: str, cat: str | None = None):
    mb = sc["_mean_by_k"]
    xs, ys = [], []
    for i, k in enumerate(ks):
        cell = mb.get(k)
        if not cell:
            continue
        v = (cell["duel_accuracy_by_category"].get(cat) if cat
             else cell.get(key))
        if v is not None:
            xs.append(i); ys.append(v)
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results/comet_align/comet_align.json"))
    ap.add_argument("--out_dir", default=str(Path(__file__).resolve().parent / "align"))
    args = ap.parse_args()

    d = json.load(open(args.results))
    scorers = d["scorers"]
    out = Path(args.out_dir)
    ks = sorted({k for s in scorers.values() for k in s["_mean_by_k"]}, key=int)
    size = d["config"].get("shortlist_size", 3)
    styles = {n: describe(n, scorers[n].get("spec", "")) for n in scorers}

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

    # ── Fig 1: the two protocols ─────────────────────────────────────────────
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
         y=-0.28)
    save(fig, out, "align_protocols")

    # ── Fig 2: per perturbation category ─────────────────────────────────────
    cats = sorted({c for s in scorers.values()
                   for cell in s["_mean_by_k"].values()
                   for c, v in cell["duel_accuracy_by_category"].items() if v is not None})
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
         y=-0.30)
    save(fig, out, "align_by_category")


if __name__ == "__main__":
    main()
