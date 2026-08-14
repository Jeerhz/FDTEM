#!/usr/bin/env python3
"""
run_matched_core.py — matched-core length sensitivity of frozen encoders (Q1+Q2).

The same CORE (one FLORES+ sentence pair + its single-perturbation negatives)
is embedded in fillers of total length L ∈ {60,120,240,480} reference tokens,
of four types φ (inert / neutral / distractor / natural) and three core
positions π (start / middle / end). See matched_core_common.py.

Task: pairwise detection — P[cos(q, x_clean) > cos(q, x_perturbed)] where q is
the bare source sentence and x_clean / x_perturbed share byte-identical filler
and differ ONLY in the core sentence. Chance = 0.5.

Reported per encoder:
  * detection per (lang, φ, π, L) and per error category      (Q2)
  * LSI_φ: within-item slope of detection per doubling of L, with a bootstrap
    CI over items, plus L50 (length where half the 0.5→1.0 headroom is lost)   (Q1)
  * core-retention cosine cos(x_clean(L,φ,π), CORE) — how far the whole-input
    embedding drifts from the bare core's embedding as L grows (geometry).

The φ₀(inert) vs φ₃(natural) contrast separates "length alone" from
"length + semantic drift": a flat inert curve with a collapsing natural curve
blames content, not length; a collapsing inert curve is architectural.

Usage
-----
  python scripts/run_matched_core.py \
      --encoders comet:Unbabel/wmt22-comet-da labse e5 hf-mean:xlm-roberta-large \
      --langs de es fr ru --output results/matched_core/matched_core.json

  python scripts/run_matched_core.py --dry_run --langs de --n_items 4
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from block_xsim_common import CATEGORIES, Perturber
from matched_core_common import (
    PHIS, FillerBank, TokenCounter, build_input, pick_items,
)
from representation_analysis_common import (
    build_embedder, cached_embed, enc_tag as _enc_tag, load_flores_source,
    pick_device,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


class Registry:
    """Deduplicating text list with (key → index) lookup."""

    def __init__(self):
        self.texts: List[str] = []
        self.key2i: Dict[tuple, int] = {}

    def add(self, key: tuple, text: str) -> int:
        if key not in self.key2i:
            self.key2i[key] = len(self.texts)
            self.texts.append(text)
        return self.key2i[key]

    def __getitem__(self, key: tuple) -> int:
        return self.key2i[key]

    def __contains__(self, key: tuple) -> bool:
        return key in self.key2i


def build_lang_registry(items, bank, L_grid, phis, pis) -> Registry:
    reg = Registry()
    for ii, it in enumerate(items):
        reg.add(("q", ii), it.src)
        reg.add(("core", ii), it.core)
        for L in L_grid:
            for phi in phis:
                for pi in pis:
                    xc = build_input(bank, it, it.core, L, phi, pi)
                    if xc is None:
                        continue
                    reg.add(("x", ii, L, phi, pi, "clean"), xc)
                    for cat, neg in it.negs.items():
                        xn = build_input(bank, it, neg, L, phi, pi)
                        reg.add(("x", ii, L, phi, pi, cat), xn)
    return reg


def evaluate_lang(E: np.ndarray, reg: Registry, items, L_grid, phis, pis,
                  categories) -> Dict:
    """Per-(φ, π, L) detection / retention for one (encoder, lang)."""
    out: Dict = {}
    # per-item detection means, keyed for the LSI mixed design
    item_det: Dict = {}         # (phi, ii) -> {L: [0/1 per (pi, cat)]}
    item_det_cat: Dict = {}     # (phi, cat, ii) -> {L: [...]}
    item_det_pi: Dict = {}      # (phi, pi, ii) -> {L: [...]}
    for phi in phis:
        out[phi] = {}
        for pi in pis:
            out[phi][str(pi)] = {}
            for L in L_grid:
                dets, rets = [], []
                by_cat: Dict[str, List[float]] = {c: [] for c in categories}
                for ii, it in enumerate(items):
                    kc = ("x", ii, L, phi, pi, "clean")
                    if kc not in reg:
                        continue
                    q = E[reg[("q", ii)]]
                    xc = E[reg[kc]]
                    sim_clean = float(q @ xc)
                    rets.append(float(xc @ E[reg[("core", ii)]]))
                    for cat in it.negs:
                        d = float(sim_clean > float(q @ E[reg[("x", ii, L, phi, pi, cat)]]))
                        dets.append(d)
                        by_cat[cat].append(d)
                        item_det.setdefault((phi, ii), {}).setdefault(L, []).append(d)
                        item_det_cat.setdefault((phi, cat, ii), {}).setdefault(L, []).append(d)
                        item_det_pi.setdefault((phi, pi, ii), {}).setdefault(L, []).append(d)
                out[phi][str(pi)][str(L)] = {
                    "detection": float(np.mean(dets)) if dets else None,
                    "n_pairs": len(dets),
                    "core_retention": float(np.mean(rets)) if rets else None,
                    "by_category": {c: (float(np.mean(v)) if v else None)
                                    for c, v in by_cat.items()},
                }
    return {"cells": out, "_item_det": item_det, "_item_det_cat": item_det_cat,
            "_item_det_pi": item_det_pi}


def _slopes(per_item: Dict, L_grid) -> List[float]:
    """Within-item OLS slope of detection on log2(L) — one slope per item."""
    x_all = np.log2(np.asarray(L_grid, dtype=float))
    slopes = []
    for by_L in per_item.values():
        Ls = sorted(by_L)
        if len(Ls) < 2:
            continue
        x = np.log2(np.asarray(Ls, dtype=float))
        y = np.asarray([np.mean(by_L[L]) for L in Ls])
        xc = x - x.mean()
        slopes.append(float((xc @ (y - y.mean())) / (xc @ xc)))
    _ = x_all
    return slopes


def _boot_ci(vals: List[float], n_boot: int = 2000, seed: int = 0):
    if not vals:
        return None, None, None
    rng = np.random.default_rng(seed)
    v = np.asarray(vals)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _l50(curve: Dict[str, float]) -> float | None:
    """Length where detection crosses 0.75 (half of the 0.5→1.0 headroom),
    log2-interpolated; None if the curve never crosses."""
    pts = sorted(((int(L), d) for L, d in curve.items() if d is not None),
                 key=lambda t: t[0])
    for (l0, d0), (l1, d1) in zip(pts, pts[1:]):
        if d0 >= 0.75 > d1:
            if d0 == d1:
                return float(l1)
            f = (d0 - 0.75) / (d0 - d1)
            return float(2 ** (np.log2(l0) + f * (np.log2(l1) - np.log2(l0))))
    if pts and pts[0][1] < 0.75:
        return float(pts[0][0])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+",
                    default=["comet:Unbabel/wmt22-comet-da"])
    ap.add_argument("--langs", nargs="+", default=["de", "es", "fr", "ru"])
    ap.add_argument("--pivot", default="en")
    ap.add_argument("--L_grid", nargs="+", type=int, default=[60, 120, 240, 480],
                    help="Total input lengths in reference tokens (doublings; "
                         "480 + specials stays under every encoder's 512 cap "
                         "so nothing is ever truncated).")
    ap.add_argument("--phis", nargs="+", default=list(PHIS), choices=list(PHIS))
    ap.add_argument("--pis", nargs="+", type=float, default=[0.0, 0.5, 1.0],
                    help="Fraction of filler placed before the core "
                         "(0 = core first, 1 = core last).")
    ap.add_argument("--n_items", type=int, default=120,
                    help="Cores per language.")
    ap.add_argument("--core_min_tok", type=int, default=20)
    ap.add_argument("--core_max_tok", type=int, default=48)
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES),
                    choices=list(CATEGORIES))
    ap.add_argument("--flores_source", choices=["plus", "raw"], default="plus")
    ap.add_argument("--splits", nargs="+", default=["dev", "devtest"])
    ap.add_argument("--max_sents", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", default="results/matched_core/emb_cache")
    ap.add_argument("--output", default="results/matched_core/matched_core.json")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    langs = [args.pivot] + [l for l in args.langs if l != args.pivot]
    loaded = [(sp, load_flores_source(args.flores_source, langs, sp, args.max_sents))
              for sp in args.splits]
    tc = TokenCounter()

    # ── per-language items, filler banks and text registries ──────────────────
    per_lang_data: Dict[str, dict] = {}
    for lang in args.langs:
        if lang == args.pivot:
            continue
        rows, sents, urls = [], [], []
        for _sp, d in loaded:
            base = len(sents)
            for i in range(d.n):
                rows.append((base + i, d.urls[i], d.sentences[args.pivot][i],
                             d.sentences[lang][i]))
                sents.append(d.sentences[lang][i])
                urls.append(d.urls[i])
        pert = Perturber.for_corpus(sents, lang, args.seed)
        items = pick_items(rows, pert, tc, args.n_items, args.core_min_tok,
                           args.core_max_tok, args.seed, args.categories)
        bank = FillerBank(sents, urls, tc, lang, args.seed)
        reg = build_lang_registry(items, bank, args.L_grid, args.phis, args.pis)
        per_lang_data[lang] = {"items": items, "reg": reg}
        n_negs = sum(len(it.negs) for it in items)
        logger.info(f"  [{lang}] {len(items)} cores ({n_negs} negatives), "
                    f"{len(reg.texts)} texts to embed")

    if args.dry_run:
        lang = next(iter(per_lang_data))
        items = per_lang_data[lang]["items"]
        bank = FillerBank([r for r in per_lang_data[lang]["reg"].texts], [],
                          tc, lang, args.seed)  # unused; examples below use reg
        it = items[0]
        logger.info(f"\n══ example core [{lang}] ══")
        logger.info(f"  q    : {it.src}")
        logger.info(f"  core : {it.core}  ({len(it.core_ids)} ref tokens)")
        for cat, neg in it.negs.items():
            logger.info(f"  neg[{cat}]: {neg}")
        reg = per_lang_data[lang]["reg"]
        for phi in args.phis:
            key = ("x", 0, args.L_grid[1], phi, 0.5, "clean")
            if key in reg:
                x = reg.texts[reg[key]]
                logger.info(f"\n  x(L={args.L_grid[1]}, {phi}, π=0.5) "
                            f"[{len(tc.ids(x))} ref tokens]:\n    {x[:400]}")
        return

    device = pick_device(args.device)
    logger.info(f"Device: {device}")
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.run_name or "matched_core",
                job_type="matched_core",
                tags=[_enc_tag(s) for s in args.encoders], config=vars(args))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"W&B init failed: {exc}")

    results: Dict = {"experiment": "matched_core",
                     "timestamp": datetime.now(timezone.utc).isoformat(),
                     "config": vars(args), "encoders": {}}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for spec in args.encoders:
        logger.info(f"\n══ {spec} ══")
        emb = build_embedder(spec, device)
        enc_res: Dict = {}
        agg_item_det: Dict = {}
        agg_item_det_cat: Dict = {}
        agg_item_det_pi: Dict = {}
        for lang, D in per_lang_data.items():
            E = cached_embed(emb, D["reg"].texts, f"mc_{lang}",
                             args.cache_dir, args.batch_size)
            ev = evaluate_lang(E, D["reg"], D["items"], args.L_grid,
                               args.phis, args.pis, args.categories)
            enc_res[lang] = ev["cells"]
            for (phi, ii), by_L in ev["_item_det"].items():
                agg_item_det.setdefault(phi, {})[(lang, ii)] = by_L
            for (phi, cat, ii), by_L in ev["_item_det_cat"].items():
                agg_item_det_cat.setdefault((phi, cat), {})[(lang, ii)] = by_L
            for (phi, pi, ii), by_L in ev["_item_det_pi"].items():
                agg_item_det_pi.setdefault((phi, pi), {})[(lang, ii)] = by_L
            line = " ".join(
                f"{phi}:{ev['cells'][phi]['0.5'][str(args.L_grid[-1])]['detection']:.2f}"
                for phi in args.phis
                if ev['cells'][phi]['0.5'][str(args.L_grid[-1])]['detection'] is not None)
            logger.info(f"  {lang}: det@L={args.L_grid[-1]} π=0.5  {line}")

        # ── means over languages ──────────────────────────────────────────────
        mean: Dict = {}
        for phi in args.phis:
            mean[phi] = {}
            for L in args.L_grid:
                dets, rets = [], []
                by_pi: Dict[str, List[float]] = {str(p): [] for p in args.pis}
                by_cat: Dict[str, List[float]] = {c: [] for c in args.categories}
                for lang in enc_res:
                    for pi in args.pis:
                        cell = enc_res[lang][phi][str(pi)][str(L)]
                        if cell["detection"] is None:
                            continue
                        dets.append(cell["detection"])
                        rets.append(cell["core_retention"])
                        by_pi[str(pi)].append(cell["detection"])
                        for c in args.categories:
                            if cell["by_category"][c] is not None:
                                by_cat[c].append(cell["by_category"][c])
                mean[phi][str(L)] = {
                    "detection": float(np.mean(dets)) if dets else None,
                    "core_retention": float(np.mean(rets)) if rets else None,
                    "by_position": {p: (float(np.mean(v)) if v else None)
                                    for p, v in by_pi.items()},
                    "by_category": {c: (float(np.mean(v)) if v else None)
                                    for c, v in by_cat.items()},
                }

        # ── Q1: LSI per φ (within-item slopes, bootstrap CI over items) ──────
        lsi: Dict = {}
        for phi in args.phis:
            m, lo, hi = _boot_ci(_slopes(agg_item_det.get(phi, {}), args.L_grid))
            lsi[phi] = {
                "lsi": m, "ci_lo": lo, "ci_hi": hi,
                "l50": _l50({L: mean[phi][L]["detection"] for L in mean[phi]}),
                "by_category": {}, "by_position": {},
            }
            for cat in args.categories:
                cm, cl, ch = _boot_ci(
                    _slopes(agg_item_det_cat.get((phi, cat), {}), args.L_grid))
                lsi[phi]["by_category"][cat] = {"lsi": cm, "ci_lo": cl, "ci_hi": ch}
            for pi in args.pis:
                pm, pl, ph = _boot_ci(
                    _slopes(agg_item_det_pi.get((phi, pi), {}), args.L_grid))
                lsi[phi]["by_position"][str(pi)] = {"lsi": pm, "ci_lo": pl, "ci_hi": ph}

        enc_res["_mean"] = mean
        enc_res["_lsi"] = lsi
        results["encoders"][emb.name] = enc_res
        for phi in args.phis:
            e = lsi[phi]
            logger.info(f"  LSI[{phi}] = {e['lsi']:+.4f} "
                        f"[{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}] per doubling  "
                        f"L50={e['l50']}")
            if wandb_run is not None:
                wandb_run.log({f"{emb.name}/lsi_{phi}": e["lsi"],
                               **({f"{emb.name}/l50_{phi}": e["l50"]}
                                  if e["l50"] is not None else {})})

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        del emb
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plots = _plots(results, out_path.parent / "plots", args)
    if wandb_run is not None:
        import wandb
        for p in plots:
            wandb_run.log({f"plots/{p.stem}": wandb.Image(str(p))})
        wandb_run.finish()
    logger.info(f"\nResults → {out_path}")


def _plots(results: Dict, plot_dir: Path, args) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    encoders = list(results["encoders"])
    Ls = [str(L) for L in args.L_grid]
    xs = [np.log2(L) for L in args.L_grid]

    def panelize(n):
        fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.8), squeeze=False,
                                 sharey=True)
        return fig, axes[0]

    # 1. detection vs length, lines per φ
    fig, axes = panelize(len(encoders))
    for ax, enc in zip(axes, encoders):
        mean = results["encoders"][enc]["_mean"]
        for phi in args.phis:
            y = [mean[phi][L]["detection"] for L in Ls]
            ax.plot(xs, y, marker="o", label=phi)
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels(args.L_grid)
        ax.set_xlabel("input length L (ref tokens)")
        ax.set_title(enc, fontsize=9); ax.grid(alpha=0.3)
    axes[0].set_ylabel("detection  P[clean closer than perturbed]")
    axes[-1].legend(fontsize=7)
    p = plot_dir / "detection_vs_length.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # 2. LSI bars per encoder × φ with bootstrap CIs
    fig, ax = plt.subplots(figsize=(2.2 + 1.6 * len(encoders), 4.2))
    w = 0.8 / len(args.phis)
    xpos = np.arange(len(encoders))
    for j, phi in enumerate(args.phis):
        ys = [results["encoders"][e]["_lsi"][phi]["lsi"] for e in encoders]
        los = [results["encoders"][e]["_lsi"][phi]["ci_lo"] for e in encoders]
        his = [results["encoders"][e]["_lsi"][phi]["ci_hi"] for e in encoders]
        yerr = np.array([[y - lo for y, lo in zip(ys, los)],
                         [hi - y for y, hi in zip(ys, his)]])
        ax.bar(xpos + j * w, ys, width=w, yerr=yerr, capsize=2, label=phi)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xpos + 0.4 - w / 2)
    ax.set_xticklabels(encoders, fontsize=7, rotation=15)
    ax.set_ylabel("LSI: Δ detection per doubling of L")
    ax.set_title("Length-sensitivity index by filler type (bootstrap 95% CI)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    p = plot_dir / "lsi_bars.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # 3. position heatmaps (rows: encoders, cols: φ) — detection[L × π]
    fig, axes = plt.subplots(len(encoders), len(args.phis),
                             figsize=(2.6 * len(args.phis), 2.2 * len(encoders)),
                             squeeze=False)
    for r, enc in enumerate(encoders):
        mean = results["encoders"][enc]["_mean"]
        for c, phi in enumerate(args.phis):
            M = np.array([[mean[phi][L]["by_position"][str(pi)] or np.nan
                           for pi in args.pis] for L in Ls], dtype=float)
            ax = axes[r][c]
            im = ax.imshow(M, vmin=0.3, vmax=1.0, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(args.pis)))
            ax.set_xticklabels(["start", "mid", "end"][:len(args.pis)], fontsize=7)
            ax.set_yticks(range(len(Ls))); ax.set_yticklabels(args.L_grid, fontsize=7)
            if r == 0:
                ax.set_title(phi, fontsize=9)
            if c == 0:
                ax.set_ylabel(enc, fontsize=7)
    fig.colorbar(im, ax=[a for row in axes for a in row], shrink=0.6,
                 label="detection")
    p = plot_dir / "position_heatmap.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # 4. per-category curves (φ = neutral: in-distribution, interference-free)
    phi0 = "neutral" if "neutral" in args.phis else args.phis[0]
    fig, axes = panelize(len(encoders))
    for ax, enc in zip(axes, encoders):
        mean = results["encoders"][enc]["_mean"]
        for cat in args.categories:
            y = [mean[phi0][L]["by_category"][cat] for L in Ls]
            ax.plot(xs, y, marker="o", label=cat)
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels(args.L_grid)
        ax.set_xlabel("L (ref tokens)"); ax.set_title(enc, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"detection (φ = {phi0})")
    axes[-1].legend(fontsize=7)
    p = plot_dir / "category_vs_length.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # 5. core-retention cosine — the geometric dilution of the whole-input
    #    embedding away from the bare core embedding
    fig, axes = panelize(len(encoders))
    for ax, enc in zip(axes, encoders):
        mean = results["encoders"][enc]["_mean"]
        for phi in args.phis:
            y = [mean[phi][L]["core_retention"] for L in Ls]
            ax.plot(xs, y, marker="o", label=phi)
        ax.set_xticks(xs); ax.set_xticklabels(args.L_grid)
        ax.set_xlabel("L (ref tokens)"); ax.set_title(enc, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cos( x(L), bare core )")
    axes[-1].legend(fontsize=7)
    p = plot_dir / "core_retention.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    for pth in paths:
        logger.info(f"  [plot] {pth}")
    return paths


if __name__ == "__main__":
    main()
