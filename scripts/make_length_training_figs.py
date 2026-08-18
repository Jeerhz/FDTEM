#!/usr/bin/env python3
"""Figures for the length-training experiment deck. Minimalist, direct-labelled."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pathlib import Path

OUT = Path("/tmp/claude-679803/-home-abensale-FDTEM/49dda269-ec34-44a9-b6f4-2f5e9153d255/scratchpad/figs")
OUT.mkdir(parents=True, exist_ok=True)

# reference palette, slots 1-3 (validated all-pairs, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b9b8b3"
SURF = "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURF, "axes.facecolor": SURF,
})


def clean(ax, keep_x=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(keep_x)
    ax.set_yticks([])
    ax.grid(False)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("wrote", name)


# ── 1. the gap: what COMET was trained on vs what it is used on ──────────────
fig, ax = plt.subplots(figsize=(9, 2.6))
ax.barh([1], [19], color=BLUE, height=.5)
ax.barh([0], [82], color=ORANGE, height=.5)
ax.text(20, 1, "  19 words — one sentence", va="center", color=INK, fontsize=14)
ax.text(83, 0, "  82 words — one document", va="center", color=INK, fontsize=14)
ax.set_yticks([1, 0]); ax.set_yticklabels(["COMET is TRAINED on", "COMET is USED on"],
                                          fontsize=14, color=INK)
ax.tick_params(left=False)
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)
ax.set_xticks([]); ax.set_xlim(0, 175)
save(fig, "fig_gap.png")

# ── 2. sentence aggregation schematic ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 3.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 4.2); ax.axis("off")
for i in range(6):
    ax.add_patch(FancyBboxPatch((0.3 + i * 1.35, 3.1), 1.15, .62,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=BLUE, ec="none"))
    ax.text(0.875 + i * 1.35, 3.41, f"s{i+1}", ha="center", va="center",
            color="white", fontsize=12)
    ax.text(0.875 + i * 1.35, 2.82, f"{-2*(i%3)}", ha="center", va="center",
            color=INK2, fontsize=10)
ax.text(8.6, 3.41, "sentences + human score", va="center", color=INK2, fontsize=12)

for row, k in enumerate([2, 3]):
    y = 1.85 - row * 1.0
    w = k * 1.35 - .2
    ax.add_patch(FancyBboxPatch((0.3, y), w, .62,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=ORANGE, ec="none"))
    ax.text(0.3 + w / 2, y + .31, f"s1 … s{k}", ha="center", va="center",
            color="white", fontsize=13)
    ax.text(0.3 + w + .35, y + .31,
            f"k={k}  ·  concatenated  ·  score = mean of {k} scores",
            va="center", color=INK, fontsize=12)
ax.text(0.3, 0.15, "k ∈ {2, 3, 4, 6}    stride 1 (train) · stride k (validation)",
        color=INK2, fontsize=12)
save(fig, "fig_aggregation.png")

# ── 3. the pool ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 3.4))
labels = ["Sentences\nWMT22 MQM", "Aggregated windows\nWMT22, k=2·3·4·6",
          "Native documents\nWMT25 human eval"]
vals = [59144, 160377, 42558]
cols = [BLUE, ORANGE, AQUA]
words = ["19 words avg", "71 words avg", "82 words avg"]
bars = ax.bar(range(3), vals, color=cols, width=.55)
for i, (v, w) in enumerate(zip(vals, words)):
    ax.text(i, v + 5000, f"{v:,}", ha="center", color=INK, fontsize=17)
    ax.text(i, v + 1000, w, ha="center", va="top", color="white", fontsize=11)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=12, color=INK)
ax.set_ylim(0, 190000)
clean(ax)
ax.text(0, 178000, "262,080 training examples   ·   21,988 validation (separate documents)",
        color=INK2, fontsize=13)
save(fig, "fig_pool.png")

# ── 4. the mixtures ──────────────────────────────────────────────────────────
mixes = ["frac000", "frac010", "frac020", "frac040", "frac060",
         "frac080", "frac100", "frac000nat", "frac000agg"]
sent = [0, 2400, 4800, 9600, 14400, 19200, 24000, 0, 0]
agg = [12000, 10800, 9600, 7200, 4800, 2400, 0, 0, 24000]
nat = [12000, 10800, 9600, 7200, 4800, 2400, 0, 24000, 0]
fig, ax = plt.subplots(figsize=(13.5, 3.5))
x = range(9)
gap = dict(edgecolor=SURF, linewidth=2)
ax.bar(x, sent, color=BLUE, width=.6, label="sentences", **gap)
ax.bar(x, agg, bottom=sent, color=ORANGE, width=.6, label="aggregated long", **gap)
ax.bar(x, nat, bottom=[s + a for s, a in zip(sent, agg)], color=AQUA, width=.6,
       label="native long", **gap)
for i in x:
    ax.text(i, 24700, "24,000", ha="center", color=INK2, fontsize=10)
ax.set_xticks(list(x))
ax.set_xticklabels(["0%", "10%", "20%", "40%", "60%", "80%", "100%",
                    "native\nonly", "aggreg.\nonly"], fontsize=12, color=INK)
ax.set_xlabel("share of single sentences in the mixture", fontsize=12)
ax.set_ylim(0, 27500)
clean(ax)
ax.legend(frameon=False, fontsize=12, ncol=3, loc="lower center",
          bbox_to_anchor=(.5, -.40))
ax.axvline(6.5, color=MUTED, lw=1, ls=":")
ax.text(7.5, 26200, "ablations", ha="center", color=INK2, fontsize=11)
save(fig, "fig_mixes.png")

# ── 5. what we finetune: the 36-run grid ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(11.5, 3.4))
ax.set_xlim(-2.6, 9.4); ax.set_ylim(-.6, 4.4); ax.axis("off")
rows = [("COMET-DA", "encoder frozen"), ("COMET-DA", "encoder trained"),
        ("CometKiwi QE", "encoder frozen"), ("CometKiwi QE", "encoder trained")]
for r, (m, reg) in enumerate(rows):
    y = 3.4 - r * .95
    col = BLUE if "DA" in m else AQUA
    ax.text(-2.5, y + .22, m, color=INK, fontsize=13)
    ax.text(-2.5, y - .12, reg, color=INK2, fontsize=11)
    for c in range(9):
        ax.add_patch(Rectangle((c * .95 + .2, y - .18), .8, .55,
                               fc=col, ec="none",
                               alpha=1.0 if reg == "encoder trained" else .45))
ax.text(4.5, -.45, "9 data mixtures", ha="center", color=INK2, fontsize=12)
ax.text(9.3, 1.9, "= 36 runs", va="center", color=INK, fontsize=17)
save(fig, "fig_grid.png")

# ── 6. evaluation ────────────────────────────────────────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.3),
                             gridspec_kw={"width_ratios": [1.3, 1]})
sets = ["WMT23\nen-de", "WMT24\nen-de", "WMT24\nen-es", "WMT24\nja-zh"]
rows_ = [5759, 9234, 9330, 8385]
a1.bar(range(4), rows_, color=ORANGE, width=.55)
for i, v in enumerate(rows_):
    a1.text(i, v + 250, f"{v:,}", ha="center", color=INK, fontsize=13)
a1.set_xticks(range(4)); a1.set_xticklabels(sets, fontsize=11, color=INK)
a1.set_ylim(0, 11500); clean(a1)
a1.set_title("Held-out paragraphs — 32,708 scored translations",
             fontsize=13, color=INK, loc="left", pad=14)

perts = ["sentence\nremoval", "sentence\nrepetit.", "conjunc-\ntion", "shuff-\nling",
         "lexical\nconsist.", "other"]
pv = [1929, 1929, 1533, 1290, 957, 1067]
a2.bar(range(6), pv, color=BLUE, width=.55)
for i, v in enumerate(pv):
    a2.text(i, v + 45, f"{v:,}", ha="center", color=INK, fontsize=11)
a2.set_xticks(range(6)); a2.set_xticklabels(perts, fontsize=9, color=INK)
a2.set_ylim(0, 2350); clean(a2)
a2.set_title("MetaDocEval — 8,705 contrastive pairs",
             fontsize=13, color=INK, loc="left", pad=14)
save(fig, "fig_eval.png")

# ── 7. expected result (hypothesis sketch) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(11.5, 3.3))
ks = [1, 2, 3, 4, 6]
base = [.360, .340, .325, .310, .290]
longonly = [.300, .348, .360, .362, .358]
mixed = [.360, .365, .372, .372, .365]
ax.plot(ks, base, "-o", color=MUTED, lw=2, ms=8, label="baseline COMET")
ax.plot(ks, longonly, "-o", color=ORANGE, lw=2, ms=8, label="trained on long only")
ax.plot(ks, mixed, "-o", color=BLUE, lw=2, ms=8, label="mixed (expected best)")
# gap annotations, placed off the curves
ax.annotate("", xy=(.82, .360), xytext=(.82, .300),
            arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.6))
ax.text(.74, .330, "expected\nloss on\nsentences", ha="right", va="center",
        color=ORANGE, fontsize=11)
ax.annotate("", xy=(6.18, .365), xytext=(6.18, .290),
            arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.6))
ax.text(6.26, .327, "expected\ngain on\nlong text", va="center",
        color=BLUE, fontsize=11)
ax.set_xlim(-.15, 7.9)
ax.set_xticks(ks); ax.set_xticklabels([f"k={k}" for k in ks], fontsize=12)
ax.set_xlabel("input length (sentences per input)", fontsize=12)
ax.set_ylabel("agreement with humans", fontsize=12)
ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=11, loc="lower center", ncol=3, bbox_to_anchor=(.45,-.30))
ax.set_title("HYPOTHESIS — shape we expect, not measured results",
             fontsize=12, color=ORANGE, loc="left", pad=12)
save(fig, "fig_hypothesis.png")

print("all figures in", OUT)
