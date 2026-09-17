"""Report figures of part 2. Writes PDF + PNG into figures/.

    python -m part2_length_training.figures.make_figures [--lens heldout val] [--results_dir DIR] [--out_dir DIR]

Three things a results table cannot show:

  A. MetaDocEval PER PHENOMENON. The published figure micro-averages the lexical
     categories, and averaging over a context window w mixes phenomena that move
     in opposite directions: a metric that gets better at spotting a deleted
     sentence while getting worse at spotting a swapped connector shows a flat
     average and no effect at all. One panel per perturbation, plus the same grid
     as a delta against each family's published baseline, separates them.

  B. The DISTRIBUTION of the scores per k, not only their rank agreement.
     Kendall tau is invariant to any monotone squashing of the scale, so a metric
     whose long-text scores collapse into a narrow band keeps its tau and loses
     its usefulness. Median, interquartile band and the spread relative to k=1
     show the collapse directly.

  C. Everything again against the NUMBER OF INPUT TOKENS instead of a sentence
     count. k is a proxy: a k=6 window of short segments and a native document
     differ by an order of magnitude in tokens, and the pools have different
     sentence lengths, so a curve against k confounds "more sentences" with
     "more tokens". The token axis is the one the model actually sees — and it
     is where CometKiwi's 512-token budget becomes visible.

Inputs (whichever are present; missing ones are reported and skipped):
results/metadoceval.json, results/token_length_by_k.json, results/correlation_<lens>.json,
results/length_profile_<lens>.json.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from part2_length_training import FIGURES_DIR, RESULTS_DIR
from part2_length_training.arms import BASE_LABEL, MIX_SPECS
from part2_length_training.models import ArmLabel, CorrelationResults, LengthProfileResults, MetaDocEvalResults

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

# Arms are named by WHAT THEY WERE TRAINED ON, never by the sweep's fraction code.
ARM_LABEL = {"base": BASE_LABEL, **{name: spec.description for name, spec in MIX_SPECS.items()}}
ARM_COLOR = {"base": MUTED, "frac100": BLUE, "frac000agg": AQUA,
             "frac000nat": ORANGE, "frac000": VIOLET, "uncontrolled": RED}
ARM_ORDER = ["base", "frac100", "frac000agg", "frac000nat", "frac000", "uncontrolled"]
FAMILIES = [("da", "COMET-DA (avec référence)"), ("qe", "CometKiwi (sans référence)")]

CATLAB = {
    "tense_consistency": "temps verbaux", "lexical_consistency": "cohérence lexicale",
    "conjunction_substitution": "connecteurs", "pronoun_swap_singular": "pronom sg.",
    "pronoun_swap_plural": "pronom pl.", "sentence_repetition": "répétition",
    "sentence_removal": "suppression", "sentence_shuffling": "permutation",
    "sentence_splitting": "découpage (préserve la qualité)",
}
LEXICAL = ["tense_consistency", "lexical_consistency", "conjunction_substitution",
           "pronoun_swap_singular", "pronoun_swap_plural"]
STRUCTURAL = ["sentence_removal", "sentence_repetition", "sentence_shuffling",
              "sentence_splitting"]
QUALITY_PRESERVING = {"sentence_splitting"}
# 19 documents each outside en-fr — the paper leaves them out of its main figure.
# Kept here so nothing is hidden, but marked, because at that size one document
# moves the accuracy by five points.
SPARSE = {"pronoun_swap_singular", "pronoun_swap_plural"}
# Token bins thinner than this are dropped rather than drawn — see the token
# figure. A Kendall tau on 13 rows swings by tenths and would be read as signal.
MIN_BIN_N = 100


def split_label(label: str) -> tuple[str, str, bool]:
    """`qe-frac000nat-frozen` -> ('qe', 'frac000nat', True)."""
    arm = ArmLabel.parse(label)
    return arm.family, arm.mix, arm.frozen


def pretty(label: str) -> str:
    fam, arm, frozen = split_label(label)
    name = ARM_LABEL.get(arm, arm)
    return f"{name}, gelé" if frozen else name


def sort_key(label: str):
    fam, arm, frozen = split_label(label)
    order = ARM_ORDER.index(arm) if arm in ARM_ORDER else len(ARM_ORDER)
    return (order, frozen, label)


def style(label: str) -> dict:
    _fam, arm, frozen = split_label(label)
    return {"color": ARM_COLOR.get(arm, INK2),
            "ls": ":" if frozen else "-",
            "lw": 1.4 if frozen else 2.0,
            "marker": "o", "ms": 3.5,
            "markeredgecolor": SURFACE, "markeredgewidth": 0.7}


class Out:
    def __init__(self, path: Path):
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self.written: list[str] = []
        self.skipped: list[str] = []

    def save(self, fig, name):
        fig.savefig(self.path / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(self.path / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        self.written.append(name)
        print("wrote", self.path / f"{name}.png")

    def skip(self, name, why):
        self.skipped.append(f"{name}: {why}")
        print(f"skipped {name} — {why}")


def note(fig, text, y=-0.04, width=130):
    wrapped = "\n".join(textwrap.fill(par, width) for par in text.split("\n"))
    fig.text(0.0, y, wrapped, fontsize=7, color=MUTED, ha="left", va="top",
             linespacing=1.5, transform=fig.transFigure)


def grid_axes(n, ncol, figsize_per=(2.5, 2.3)):
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(figsize_per[0] * ncol, figsize_per[1] * nrow))
    flat = [ax for row in axes for ax in row]
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat[:n]


# ════════════════════════════════════════════════════════════════════════════
# 0a — why the token axis is not the same axis as k
# ════════════════════════════════════════════════════════════════════════════
def token_length_figure(tl: dict, out: Out, lens: str) -> None:
    """How many tokens each window size actually is.

    The reason the rest of the report carries a token axis: on the held-out sets
    a "whole document" is SHORTER than a 6-sentence window (median 70 vs 181
    tokens), so a curve drawn against k is not a curve against length. The two
    orderings genuinely disagree, and only one of them is what the model sees.
    """
    d = tl.get(lens)
    if not d:
        out.skip(f"token_length_{lens}", "no token-length data for this lens")
        return
    bins = np.asarray(d["bins"], dtype=float)
    centres = bins + (bins[1] - bins[0]) / 2
    ks = [k for k in ("1", "2", "3", "4", "6", "0") if k in d["per_k"]]
    K_LAB = {"1": "1 phrase", "2": "2 phrases", "3": "3 phrases", "4": "4 phrases",
             "6": "6 phrases", "0": "document entier"}
    K_COL = {"1": BLUE, "2": AQUA, "3": AQUA, "4": AQUA, "6": AQUA, "0": ORANGE}

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    for i, k in enumerate(ks):
        h = np.asarray(d["per_k"][k]["n_mt"]["hist"], dtype=float)
        if h.sum() == 0:
            continue
        h = h / h.sum()
        y = len(ks) - 1 - i
        ax.fill_between(centres[:len(h)], y, y + 2.6 * h, color=K_COL[k],
                        alpha=0.75, lw=0.7, edgecolor=SURFACE, zorder=3 + i)
        med = d["per_k"][k]["n_mt"]["q"][2]
        ax.plot([med, med], [y, y + 0.30], color=INK, lw=1.1, zorder=30)
        ax.text(med + 6, y + 0.10, f"{med:.0f}", fontsize=7, color=INK)
    ax.set_yticks([len(ks) - 1 - i + 0.12 for i in range(len(ks))])
    ax.set_yticklabels([f"{K_LAB[k]}  (n={d['per_k'][k]['n']:,})" for k in ks],
                       fontsize=7.8)
    ax.set_xlim(0, 420)
    ax.set_xlabel("tokens XLM-R du mt")
    ax.set_title("Ce que « une fenêtre de k phrases » pèse vraiment", fontsize=9.5)
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    xs = range(len(ks))
    for field, marker, lab in (("n_mt", "o", "mt seul (modèles avec référence)"),
                               ("n_concat", "s", "src+mt (CometKiwi)")):
        med = [d["per_k"][k][field]["q"][2] for k in ks]
        lo = [d["per_k"][k][field]["q"][1] for k in ks]
        hi = [d["per_k"][k][field]["q"][3] for k in ks]
        ax.plot(xs, med, marker=marker, ms=5, lw=2, label=lab,
                color=VIOLET if field == "n_mt" else ORANGE,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4)
        ax.fill_between(xs, lo, hi, alpha=0.15, lw=0,
                        color=VIOLET if field == "n_mt" else ORANGE)
    ax.axhline(512, color=RED, ls=":", lw=1)
    ax.set_ylim(0, 640)                      # headroom so the legend clears the line
    ax.text(len(ks) - 1, 522, "budget 512 de CometKiwi ", fontsize=6.8, color=RED,
            ha="right", va="bottom")
    ax.set_xticks(list(xs)); ax.set_xticklabels([K_LAB[k].split()[0] for k in ks],
                                                fontsize=7.5)
    ax.set_xlabel("taille de la fenêtre")
    ax.set_ylabel("tokens en entrée")
    ax.set_title("Médiane et interquartile", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False, loc="upper left",
              bbox_to_anchor=(0.0, 1.0))

    fig.tight_layout()
    note(fig,
         f"Lentille {lens}. À gauche, la distribution des longueurs pour chaque taille de "
         "fenêtre (aire normalisée, trait = médiane). À droite, la même chose résumée.\n"
         "Le point : sur les jeux held-out, un « document entier » fait 70 tokens de "
         "médiane et une fenêtre de 6 phrases en fait 181 — le document est PLUS COURT. "
         "Ordonner les régimes par nombre de phrases et les ordonner par longueur réelle "
         "ne donnent donc pas le même ordre, et c'est la longueur réelle que le modèle "
         "encode. D'où l'axe en tokens partout ailleurs.", y=-0.06)
    out.save(fig, f"token_length_{lens}")


# ════════════════════════════════════════════════════════════════════════════
# 0 — the headline: agreement with human judgement, per input regime
# ════════════════════════════════════════════════════════════════════════════
def regime_figure(corr: CorrelationResults, out: Out, lens: str) -> None:
    """Kendall tau per input regime, one bar per arm — the result table as a plot.

    Three regimes rather than six k values: single sentences (what the published
    metrics were trained on), concatenated windows (k=2..6 averaged), and whole
    documents (k=0). Averaging k=2..6 is safe here because they move together;
    the per-k detail lives in the token figure.
    """
    M = corr.models
    labels = sorted(M, key=sort_key)
    if len(labels) < 2:
        out.skip(f"tau_by_regime_{lens}", "fewer than two models in the results")
        return

    def tau(m, k):
        cell = M[m].mean_by_k.get(k)
        return cell.kendall if cell else None

    def concat(m):
        vals = [tau(m, k) for k in ("2", "3", "4", "6")]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    REGIMES = [("Phrases seules", "$k{=}1$", lambda m: tau(m, "1")),
               ("Phrases concaténées", "moyenne $k{\\in}\\{2,3,4,6\\}$", concat),
               ("Documents entiers", "$k{=}0$", lambda m: tau(m, "0"))]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    for ax, (title, sub, fn) in zip(axes, REGIMES):
        vals = [(fn(m), m) for m in labels if fn(m) is not None]
        if not vals:
            ax.axis("off"); continue
        vals.sort(key=lambda t: t[0])
        ys = np.arange(len(vals))
        for y, (v, m) in zip(ys, vals):
            fam, arm, _fr = split_label(m)
            c = ARM_COLOR.get(arm, INK2)
            base = arm == "base"
            # matplotlib draws a hatch in the EDGE colour, so a bar whose edge
            # matches its face shows no hatch at all — which would make the
            # legend's "hachuré = sans référence" a false claim on every
            # coloured Kiwi bar.
            ax.barh(y, v, height=0.7, color=c, alpha=0.35 if base else 1.0,
                    edgecolor=c if fam == "da" else SURFACE,
                    linewidth=1.2 if base else (0.9 if fam == "qe" else 0),
                    zorder=3, hatch="////" if fam == "qe" else None)
            ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=7.2,
                    color=INK if not base else INK2,
                    fontweight="bold" if y == len(vals) - 1 else "normal")
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{'Kiwi' if split_label(m)[0] == 'qe' else 'DA'} · {pretty(m)}"
                            for _v, m in vals], fontsize=7.2)
        ax.set_xlim(0, max(v for v, _ in vals) * 1.24)
        ax.set_title(title, fontsize=9.5, pad=10)
        ax.text(0.5, 1.005, sub, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.2, color=MUTED)
        ax.set_xlabel(r"Kendall $\tau$ vs jugement humain")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    # The two lenses do NOT have the same status and must not be captioned alike:
    # `val` is the split early stopping monitors, so the checkpoint was SELECTED on
    # it — a development signal. `heldout` is the reportable one.
    provenance = (
        "Lentille validation — documents disjoints de l'entraînement, mais c'est le "
        "split que l'early stopping surveille : le checkpoint a été SÉLECTIONNÉ "
        "dessus. Signal de développement, pas un chiffre reportable."
        if lens == "val" else
        "Lentille held-out — portions WMT22/23/24/25 jamais vues, ni à l'entraînement "
        "ni à la sélection. Ce sont les chiffres reportables.")
    note(fig,
         f"{provenance} Hachuré = famille sans référence (CometKiwi), plein = avec "
         "référence (COMET-DA) ; barre creuse = métrique publiée.\n"
         "Chaque bras est nommé par ce sur quoi il a été ré-entraîné, à composition "
         "seule variable : même nombre de lignes (24 000), même budget (60 époques max, "
         "patience 10), même schéma d'optimisation.", y=-0.06)
    out.save(fig, f"tau_by_regime_{lens}")


# ════════════════════════════════════════════════════════════════════════════
# A — MetaDocEval, one panel per phenomenon
# ════════════════════════════════════════════════════════════════════════════
def metadoceval_figures(mde: dict, out: Out) -> None:
    models = mde["models"]
    counts = mde.get("counts_per_perturbation", {})
    ws = [str(w) for w in mde.get("windows", [1, 3, 6, 9])]
    present = sorted({k.split("|")[0] for m in models.values() for k in m["micro"]})
    cats = [c for c in LEXICAL + STRUCTURAL if c in present]
    labels = sorted(models, key=sort_key)

    def acc(model, cat, w):
        return models[model]["micro"].get(f"{cat}|w{w}", {}).get("accuracy")

    def draw(ax, cat, subset, delta_of=None):
        for label in subset:
            ys, xs = [], []
            for i, w in enumerate(ws):
                a = acc(label, cat, w)
                if a is None:
                    continue
                if delta_of is not None:
                    b = acc(delta_of, cat, w)
                    if b is None:
                        continue
                    a = a - b
                xs.append(i); ys.append(a)
            if xs:
                ax.plot(xs, ys, **style(label))
        ax.set_xticks(range(len(ws)))
        ax.set_xticklabels(ws, fontsize=7.5)
        n = counts.get(cat, {}).get("docs")
        title = CATLAB.get(cat, cat.replace("_", " "))
        sub = f"$n$={n} documents" if n else ""
        if cat in SPARSE:
            sub += " — trop peu pour être lu"
        ax.set_title(f"{title}\n{sub}" if sub else title, fontsize=8,
                     color=MAGENTA if cat in QUALITY_PRESERVING else
                           (MUTED if cat in SPARSE else INK))
        if cat in SPARSE:
            ax.set_facecolor("#f5f4f1")

    # ── A1: absolute accuracy, one panel per phenomenon, one column per family
    for fam, fam_title in FAMILIES:
        subset = [l for l in labels if split_label(l)[0] == fam]
        if not subset:
            out.skip(f"mde_phenomena_{fam}", f"no {fam} model in metadoceval.json")
            continue
        fig, axes = grid_axes(len(cats), 5, (2.45, 2.35))
        for ax, cat in zip(axes, cats):
            draw(ax, cat, subset)
            ax.axhline(0.5, color=MUTED, lw=0.9, ls="--")
            ax.set_ylim(0.42, 1.03)
        for i, ax in enumerate(axes):
            if i % 5 == 0:
                ax.set_ylabel("exactitude", fontsize=8)
            ax.set_xlabel("fenêtre $w$", fontsize=8)
        handles = [Line2D([], [], **{k: v for k, v in style(l).items()
                                     if k != "marker"}) for l in subset]
        fig.legend(handles, [pretty(l) for l in subset], loc="lower center",
                   ncol=min(len(subset), 6), frameon=False, fontsize=7.8,
                   bbox_to_anchor=(0.5, -0.06))
        fig.suptitle(f"MetaDocEval par phénomène — {fam_title}", fontsize=11)
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        note(fig,
             "Exactitude contrastive = P[score(original) > score(perturbé)], hasard .50 "
             "(trait tireté). Chaque panneau est un phénomène : la moyenne sur les "
             "catégories, elle, laisse un phénomène qui s'améliore compenser un "
             "phénomène qui se dégrade, et sort plate.\n"
             "Le découpage de phrases préserve la qualité par construction — sa "
             "« exactitude » est un taux de faux positifs, plus bas est mieux.",
             y=-0.10)
        out.save(fig, f"mde_phenomena_{fam}")

    # ── A2: delta against the published metric of the same family ────────────
    # One figure per family: colour already encodes the arm, so superposing the
    # two families would leave them indistinguishable.
    for fam, fam_title in FAMILIES:
        base = f"{fam}-base"
        subset = [l for l in labels
                  if split_label(l)[0] == fam and split_label(l)[1] != "base"]
        if base not in models or not subset:
            out.skip(f"mde_phenomena_delta_{fam}",
                     f"needs {base} and at least one trained {fam} arm")
            continue
        fig, axes = grid_axes(len(cats), 5, (2.45, 2.35))
        for ax, cat in zip(axes, cats):
            draw(ax, cat, subset, delta_of=base)
            ax.axhline(0.0, color=MUTED, lw=0.9, ls="--")
        for i, ax in enumerate(axes):
            if i % 5 == 0:
                ax.set_ylabel(r"$\Delta$ vs publié", fontsize=8)
            ax.set_xlabel("fenêtre $w$", fontsize=8)
        handles = [Line2D([], [], **{k: v for k, v in style(l).items()
                                     if k != "marker"}) for l in subset]
        fig.legend(handles, [pretty(l) for l in subset], loc="lower center",
                   ncol=min(len(subset), 6), frameon=False, fontsize=7.8,
                   bbox_to_anchor=(0.5, -0.06))
        fig.suptitle(f"MetaDocEval par phénomène — écart à la métrique publiée "
                     f"({fam_title})", fontsize=11)
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        note(fig,
             "Chaque courbe est exactitude(bras) − exactitude(métrique publiée de la même "
             "famille), à fenêtre égale. Au-dessus de zéro : le ré-entraînement a acheté "
             "de la détection pour ce phénomène ; au-dessous : il en a coûté. Un bras dont "
             "la moyenne est nulle mais dont les panneaux vont dans les deux sens n'est pas "
             "un bras sans effet — c'est un bras dont les effets se compensent.\n"
             "Noter les échelles : chaque panneau a la sienne, les amplitudes ne se "
             "comparent pas d'un panneau à l'autre.",
             y=-0.10)
        out.save(fig, f"mde_phenomena_delta_{fam}")


# ════════════════════════════════════════════════════════════════════════════
# B / C — score distributions, per k and per input-token count
# ════════════════════════════════════════════════════════════════════════════
def _pool(profile: dict, label: str, axis: str, key: str) -> dict:
    """Merge a model's per-file cells into one cell per bin.

    Quantiles cannot be averaged, so the pooling is done on the histograms —
    which is exactly what the histogram is stored for — and the quantiles are
    read back off the pooled counts. Means and IQRs then describe the whole
    lens, not whichever file happened to be biggest.
    """
    edges = np.asarray(profile["score_grid"], dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    acc: dict = {}
    for _fname, entry in profile["models"][label].items():
        cells = entry[key] if axis == "k" else entry["by_tokens"].get(axis, {})
        for b, c in cells.items():
            if not c["pred"].get("n"):
                continue
            slot = acc.setdefault(b, {"hist": np.zeros(len(centres)), "n": 0,
                                      "taus": [], "tok": []})
            slot["hist"] += np.asarray(c["pred"]["hist"], dtype=float)
            slot["n"] += c["pred"]["n"]
            if c["corr"].get("kendall") is not None:
                slot["taus"].append(c["corr"]["kendall"])
            if c["tokens"].get("median") is not None:
                slot["tok"].append(c["tokens"]["median"])
    out = {}
    for b, s in acc.items():
        tot = s["hist"].sum()
        if tot == 0:
            continue
        cdf = np.cumsum(s["hist"]) / tot
        q = {p: float(np.interp(p / 100, cdf, centres)) for p in (5, 25, 50, 75, 95)}
        out[b] = {"n": s["n"], "q": q, "iqr": q[75] - q[25],
                  "mean": float((s["hist"] * centres).sum() / tot),
                  "kendall": float(np.mean(s["taus"])) if s["taus"] else None,
                  "tokens": float(np.median(s["tok"])) if s["tok"] else None}
    return out


def _band(ax, xs, cells, keys, label=None, color=None, alpha=(0.10, 0.22),
          line=True):
    q = [cells[k]["q"] for k in keys]
    c = color or style(label)["color"]
    ax.fill_between(xs, [x[5] for x in q], [x[95] for x in q],
                    color=c, alpha=alpha[0], lw=0)
    ax.fill_between(xs, [x[25] for x in q], [x[75] for x in q],
                    color=c, alpha=alpha[1], lw=0)
    if line:
        st = style(label) if label else {"color": c, "lw": 1.5}
        ax.plot(xs, [x[50] for x in q], **st)


def band_grid(out: Out, name: str, labels, pooled, xkeys, xticklabels, xlabel,
              title, caption, ncol=None):
    """One panel per model: with six arms a single axes of overlaid bands is mud.

    Each panel also carries its family's published metric as a grey band behind
    the arm, so a panel is read as "what did this training change", not as an
    absolute the eye has to remember across panels.
    """
    shown = [l for l in labels if len([k for k in xkeys if k in pooled.get(l, {})]) >= 2]
    if not shown:
        out.skip(name, "no model has two populated bins")
        return
    ncol = ncol or min(len(shown), 6)
    fig, axes = grid_axes(len(shown), ncol, (2.35, 2.25))
    ymin = min(pooled[l][k]["q"][5] for l in shown for k in xkeys if k in pooled[l])
    ymax = max(pooled[l][k]["q"][95] for l in shown for k in xkeys if k in pooled[l])
    pad = 0.06 * (ymax - ymin or 1)
    for ax, label in zip(axes, shown):
        fam = split_label(label)[0]
        base = f"{fam}-base"
        if base in pooled and base != label:
            bk = [k for k in xkeys if k in pooled[base]]
            if len(bk) >= 2:
                _band(ax, [xkeys.index(k) for k in bk], pooled[base], bk,
                      color=MUTED, alpha=(0.06, 0.14), line=False)
        keys = [k for k in xkeys if k in pooled[label]]
        _band(ax, [xkeys.index(k) for k in keys], pooled[label], keys, label)
        ax.set_xticks(range(len(xkeys)))
        rot = max(len(t) for t in xticklabels) > 4      # rotate only if needed
        ax.set_xticklabels(xticklabels, fontsize=6.5,
                           **({"rotation": 40, "ha": "right",
                               "rotation_mode": "anchor"} if rot else {}))
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_title(f"{'DA' if fam == 'da' else 'Kiwi'} · {pretty(label)}",
                     fontsize=8, color=ARM_COLOR.get(split_label(label)[1], INK))
    for i, ax in enumerate(axes):
        if i % ncol == 0:
            ax.set_ylabel("score", fontsize=8)
        if i >= len(shown) - ncol:
            ax.set_xlabel(xlabel, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    note(fig, caption, y=-0.03)
    out.save(fig, name)


def distribution_figures(profile: dict, out: Out, lens: str) -> None:
    labels = sorted(profile["models"], key=sort_key)
    K_LAB = {"1": "1 phrase", "2": "2", "3": "3", "4": "4", "6": "6",
             "0": "doc. natif"}

    # ── B1: score distribution per k, one panel per model ────────────────────
    pooled_k = {l: _pool(profile, l, "k", "by_k") for l in labels}
    ks = [k for k in ("1", "2", "3", "4", "6", "0")
          if any(k in p for p in pooled_k.values())]
    if len(ks) < 2:
        out.skip(f"dist_by_k_{lens}", "fewer than two window sizes in the profile")
        return

    band_grid(
        out, f"dist_by_k_{lens}", labels, pooled_k, ks,
        [K_LAB.get(k, k) for k in ks], "fenêtre $k$",
        "Distribution des scores selon la longueur du texte",
        f"Lentille {lens}. Trait = médiane, bande foncée = intervalle interquartile, "
        "bande claire = 5–95 %, agrégés sur tous les fichiers d'évaluation via les "
        "histogrammes. La bande grise en arrière-plan est la métrique publiée de la "
        "même famille, à échelle identique.\n"
        "Ce que la corrélation ne peut pas montrer : une métrique dont les scores se "
        "resserrent sur les textes longs garde son tau — le rang est invariant à toute "
        "transformation monotone — et perd sa résolution.")

    # ── B2: spread relative to single sentences ──────────────────────────────
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    drew = False
    for l in labels:
        p = pooled_k[l]
        ref = p.get("1", {}).get("iqr")
        keys = [k for k in ks if k in p]
        if not ref or ref <= 0 or len(keys) < 2:
            continue
        st = style(l)
        if split_label(l)[0] == "qe" and st["ls"] == "-":
            st["ls"] = "--"
        ax.plot([ks.index(k) for k in keys], [p[k]["iqr"] / ref for k in keys], **st)
        drew = True
    if not drew:
        out.skip(f"spread_by_k_{lens}", "no model has a k=1 cell to normalise by")
        plt.close(fig)
    else:
        ax.axhline(1.0, color=MUTED, lw=1, ls="--")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([K_LAB.get(k, k) for k in ks], fontsize=7.5)
        ax.set_xlabel("taille de la fenêtre $k$ (segments)")
        ax.set_ylabel("écart interquartile ÷ celui à $k{=}1$")
        ax.set_title("Compression de l'échelle avec la longueur", fontsize=9.5)
        handles = [Line2D([], [], color=ARM_COLOR.get(a, INK2), lw=2)
                   for a in ARM_ORDER if any(split_label(l)[1] == a for l in labels)]
        names = [ARM_LABEL[a] for a in ARM_ORDER
                 if any(split_label(l)[1] == a for l in labels)]
        ax.legend(handles, names, fontsize=7, frameon=False, loc="best")
        fig.tight_layout()
        note(fig,
             f"Lentille {lens}. Sous 1 : la métrique répartit les textes longs sur une "
             "plage plus étroite que les phrases isolées, donc deux traductions doivent "
             "différer davantage pour que l'écart de score soit lisible. Tireté = famille "
             "sans référence, pointillé = encodeur gelé.", y=-0.06)
        out.save(fig, f"spread_by_k_{lens}")

    # ── C: the same quantities against the input token count ─────────────────
    # Each family is plotted against the length ITS encoder sees: the DA arms
    # encode src, mt and ref separately (n_mt), CometKiwi packs src+mt into one
    # 512-token sequence (n_concat) — and that budget is why the QE panel is the
    # one where the last bin matters.
    field_of = {"da": "n_mt", "qe": "n_concat"}
    bins = profile["token_bins"]
    pooled_t = {l: _pool(profile, l, field_of[split_label(l)[0]], "by_tokens")
                for l in labels if split_label(l)[0] in field_of}
    if not any(pooled_t.values()):
        out.skip(f"tokens_{lens}", "no token bins in the profile")
        return
    # Thin bins are worse than no bins: on the held-out lens the DA 448-512 bin
    # holds 13 rows and 512+ holds 2. Drop any bin whose family baseline is below
    # MIN_BIN_N, and print the surviving counts on the axis.
    def fam_n(fam, b):
        return pooled_t.get(f"{fam}-base", {}).get(b, {}).get("n", 0)

    keep = {fam: {b for b in bins if fam_n(fam, b) >= MIN_BIN_N}
            for fam, _t in FAMILIES}
    used = [b for b in bins
            if any(b in p for p in pooled_t.values())
            and any(b in keep[fam] for fam, _t in FAMILIES)]
    for fam, _t in FAMILIES:
        thin = [(b, fam_n(fam, b)) for b in bins if 0 < fam_n(fam, b) < MIN_BIN_N]
        if thin:
            print(f"  [{lens}] {fam}: dropped thin token bin(s) "
                  + ", ".join(f"{b} (n={n})" for b, n in thin))

    # C1: rank agreement and scale width per token bin — one line per model, so
    # both rows stay readable with six arms on them.
    # sharex="col", not True: a shared x axis shares the tick FORMATTER across
    # all four panels, so the per-family row counts written under the DA panel
    # would silently be CometKiwi's. Sharing within a column still aligns the
    # tau panel with the spread panel below it, which is the part that matters.
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.0), sharex="col")
    for col, (fam, fam_title) in enumerate(FAMILIES):
        subset = [l for l in labels if split_label(l)[0] == fam and pooled_t.get(l)]
        top, bot = axes[0][col], axes[1][col]
        for l in subset:
            p = pooled_t[l]
            keys = [b for b in used if b in p and b in keep[fam]
                    and p[b]["kendall"] is not None]
            if keys:
                top.plot([used.index(b) for b in keys],
                         [p[b]["kendall"] for b in keys], **style(l))
            keys = [b for b in used if b in p and b in keep[fam]]
            if keys:
                bot.plot([used.index(b) for b in keys],
                         [p[b]["iqr"] for b in keys], **style(l))
        top.set_title(f"{fam_title}\nlongueur = {field_of[fam]}", fontsize=9)
        bot.set_xticks(range(len(used)))
        bot.set_xticklabels(
            [f"{b}\nn={fam_n(fam, b):,}" if b in keep[fam] else f"{b}\n—"
             for b in used],
            fontsize=6.0, rotation=40, ha="right", rotation_mode="anchor")
        bot.set_xlabel("tokens XLM-R en entrée")
        if fam == "qe" and len(used) >= 2:
            for ax in (top, bot):
                ax.axvline(len(used) - 1.5, color=RED, lw=1, ls=":")
            top.text(len(used) - 1.45, top.get_ylim()[1], " budget 512", fontsize=6.5,
                     color=RED, va="top")
    axes[0][0].set_ylabel(r"Kendall $\tau$ vs jugement humain")
    axes[1][0].set_ylabel("écart interquartile des scores")
    handles = [Line2D([], [], color=ARM_COLOR.get(a, INK2), lw=2)
               for a in ARM_ORDER if any(split_label(l)[1] == a for l in labels)]
    names = [ARM_LABEL[a] for a in ARM_ORDER
             if any(split_label(l)[1] == a for l in labels)]
    fig.legend(handles, names, loc="lower center", ncol=min(len(names), 6),
               frameon=False, fontsize=7.8, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Longueur en tokens plutôt qu'en segments", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    note(fig,
         f"Lentille {lens}. Abscisse : nombre de tokens XLM-R de l'entrée réellement "
         "encodée — le mt seul pour les modèles avec référence (les trois côtés sont "
         "encodés séparément), src+mt pour CometKiwi, qui les concatène en une seule "
         "séquence de 512 tokens.\n"
         "Haut : accord de rang par tranche. Bas : largeur de l'échelle utilisée dans la "
         "tranche (écart interquartile des scores) — c'est la résolution, que le tau ne "
         "voit pas.\n"
         f"Les tranches de moins de {MIN_BIN_N} lignes ne sont pas tracées : sur la "
         "lentille held-out, le mt des modèles avec référence dépasse rarement 320 tokens "
         "(448-512 : 13 lignes, 512+ : 2), et un tau calculé sur treize items varie de "
         "plusieurs dixièmes. L'effectif conservé est écrit sous chaque tranche.",
         y=-0.09)
    out.save(fig, f"tokens_{lens}")

    # C2: the distributions themselves, per token bin, one panel per model
    band_grid(
        out, f"dist_by_tokens_{lens}", labels, pooled_t, used, used,
        "tokens XLM-R", "Distribution des scores selon le nombre de tokens en entrée",
        f"Lentille {lens}. Même lecture que la figure par $k$, mais sur l'axe que le "
        "modèle voit réellement : mt seul pour les modèles avec référence, src+mt pour "
        "CometKiwi. Bande grise = métrique publiée de la même famille.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default=str(RESULTS_DIR))
    ap.add_argument("--out_dir", default=str(FIGURES_DIR))
    ap.add_argument("--lens", nargs="+", default=["heldout", "val"])
    args = ap.parse_args()

    res = Path(args.results_dir)
    out = Out(Path(args.out_dir))

    mde_path = res / "metadoceval.json"
    if mde_path.exists():
        metadoceval_figures(MetaDocEvalResults.model_validate_json(mde_path.read_text()).model_dump(), out)
    else:
        out.skip("mde_phenomena*", f"{mde_path} not found — run eval_metadoceval")

    tl_path = res / "token_length_by_k.json"
    tl = json.load(open(tl_path)) if tl_path.exists() else {}

    for lens in args.lens:
        if tl:
            token_length_figure(tl, out, lens)
        else:
            out.skip(f"token_length_{lens}", f"{tl_path} not found")

        c = res / f"correlation_{lens}.json"
        if c.exists():
            regime_figure(CorrelationResults.model_validate_json(c.read_text()), out, lens)
        else:
            out.skip(f"tau_by_regime_{lens}", f"{c} not found")

        p = res / f"length_profile_{lens}.json"
        if not p.exists():
            out.skip(f"*_{lens}", f"{p} not found — run eval_length_profile")
            continue
        distribution_figures(LengthProfileResults.model_validate_json(p.read_text()).model_dump(), out, lens)

    print(f"\n{len(out.written)} figure(s) → {out.path}")
    for s in out.skipped:
        print(f"  not produced — {s}")


if __name__ == "__main__":
    main()
