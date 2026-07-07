#!/usr/bin/env python3
"""
plot_alignment_bar.py — per-model bar plot of cross-lingual alignment (xSIM).

Reads an E1/E3 retrieval JSON (produced by run_retrieval.py) and draws one
bar per encoder = its xSIM retrieval accuracy at sentence level (k=1, en↔xx), the
headline "how well does this encoder align translations?" number. Pure json +
matplotlib (no torch), so it regenerates the figure without re-running the GPU job:

    python scripts/plot_alignment_bar.py results/flores_plus/e1_e3_retrieval.json
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Optional


def _pretty(name: str) -> str:
    """Readable axis label for an encoder result key."""
    if name.startswith("comet:"):
        tail = name.split(":", 1)[1]
        return "COMET\n(wmt22-da)" if "wmt22" in tail else f"COMET-bio\n({tail})"
    if name.startswith("xlmr:"):
        return "XLM-R\n(raw)"
    if name == "labse":
        return "LaBSE"
    if name.startswith("e5:"):
        return "mE5"
    return name


def _family_color(name: str) -> str:
    """Colour by encoder family: COMET subject vs XLM-R control vs aligners."""
    if name.startswith("comet:"):
        return "#1f77b4"   # COMET (the subject)
    if name.startswith("xlmr:"):
        return "#7f7f7f"   # raw XLM-R (untrained control)
    return "#2ca02c"       # dedicated aligners (LaBSE / E5)


def alignment_barplot(results: Dict, plot_dir: Path, length: Optional[int] = None) -> Path:
    """One bar per encoder = xSIM accuracy at the given pseudo-doc length (default
    the shortest available = sentence level). Returns the saved figure path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    lengths = results.get("lengths", [1])
    k = length if length is not None else min(lengths)

    names = list(results["encoders"].keys())
    accs = [results["encoders"][n][str(k)]["mean_acc"] for n in names]
    labels = [_pretty(n) for n in names]
    colors = [_family_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.5), 5))
    bars = ax.bar(range(len(names)), accs, color=colors, edgecolor="black", linewidth=0.6)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.004, f"{a:.3f}",
                ha="center", va="bottom", fontsize=9)

    ymin = max(0.0, min(accs) - 0.07)
    ax.set_ylim(ymin, 1.015)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(f"xSIM retrieval accuracy (en↔xx, k={k})")
    src = results.get("flores_source", "?")
    split = results.get("split", "?")
    langs = "·".join(results.get("langs", []))
    ax.set_title(f"Cross-lingual alignment of translations — FLORES+ ({src}/{split})\n"
                 f"languages: {langs}")
    ax.grid(axis="y", alpha=0.3)

    # legend for the family colours
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#1f77b4", edgecolor="black", label="COMET encoder (subject)"),
        Patch(facecolor="#7f7f7f", edgecolor="black", label="XLM-R raw (control)"),
        Patch(facecolor="#2ca02c", edgecolor="black", label="dedicated aligners"),
    ], fontsize=8, loc="lower right")

    plt.tight_layout()
    out = plot_dir / "e1_alignment_barplot.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json", help="E1/E3 retrieval JSON (run_retrieval.py output).")
    p.add_argument("--length", type=int, default=None,
                   help="Pseudo-doc length k to plot (default: shortest available).")
    p.add_argument("--plot_dir", default=None,
                   help="Output dir (default: <json dir>/plots).")
    args = p.parse_args()

    with open(args.json, encoding="utf-8") as fh:
        results = json.load(fh)
    plot_dir = Path(args.plot_dir) if args.plot_dir else Path(args.json).parent / "plots"
    out = alignment_barplot(results, plot_dir, args.length)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
