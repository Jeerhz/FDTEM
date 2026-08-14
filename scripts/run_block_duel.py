#!/usr/bin/env python3
"""
run_block_duel.py — the 1-vs-(D+1) duel variant of block-level xSIM++.

Motivation
----------
In run_block_xsim.py the candidate pool changes with k (fewer blocks, but more
negatives per block: k positions × categories × variants), so pool composition
is a residual confound when comparing block lengths. Here every retrieval is a
*duel with a fixed number of candidates*, at every k:

  * the gold block (the k true translations, concatenated, unperturbed), and
  * D single-error hard negatives of that same block.

D = 6 is the k=2 budget: with one variant per (position, category), a
2-sentence block yields at most 2 positions × 3 categories = 6 negatives.
A block yields up to 3k negatives; the duel is scored over ALL C(m, D) subsets
of its m available negatives. That average has a closed form — the gold wins
against a D-subset S iff it beats every member of S, hence

    P[win] = C(w, D) / C(m, D),   w = #negatives with cos(q, neg) < cos(q, gold)

so no subset enumeration is needed and the average is exact. Blocks with
m < D negatives are skipped (and counted): every scored duel is exactly one
gold against D single-error negatives, so task difficulty is identical
across k and the only variable left is block length.

Coverage vs pool size
---------------------
D = 6 at k = 2 keeps only blocks whose BOTH sentences host all three error
types (~9% of de blocks). Smaller duels qualify more blocks: D = 5 needs any
5 of the 6 possible edits, D = 4 any 4. Since the negatives and embeddings do
not depend on D, several duel sizes are scored in one pass over the same
similarities (--duel_sizes 6 5 4) for a clean coverage-vs-difficulty
comparison.

Usage
-----
  python scripts/run_block_duel.py \
      --encoders comet:Unbabel/wmt22-comet-da labse e5 \
      --langs de es fr ru --k_list 2 3 4 5 --splits dev devtest \
      --duel_sizes 6 5 4 \
      --output results/block_duel/block_duel.json

  # inspect negative coverage (how many blocks reach each duel size) on CPU
  python scripts/run_block_duel.py --dry_run --langs de --k_list 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from block_xsim_common import CATEGORIES, Perturber, block_text, build_blocks
from representation_analysis_common import (
    build_embedder, cached_embed, enc_tag as _enc_tag, load_flores_source,
    pick_device,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class DuelBlock:
    query: str
    gold: str
    negs: List[Tuple[int, str, str]]      # (position, category, text)


def build_duels(loaded, k: int, query_lang: str, pool_lang: str,
                categories, pert: Perturber, joiner: str,
                max_blocks: Optional[int] = None) -> List[DuelBlock]:
    """One duel per block: gold + one negative per (position, category)."""
    duels: List[DuelBlock] = []
    for sp, data in loaded:
        blocks = build_blocks(data, k, split=sp)
        if max_blocks:
            blocks = blocks[:max_blocks]
        sents = data.sentences[pool_lang]
        for blk in blocks:
            parts = [sents[r] for r in blk.rows]
            negs: List[Tuple[int, str, str]] = []
            for pos in range(blk.k):
                for cat in categories:
                    vs = pert.variants(parts[pos], cat, 1)
                    if vs:
                        newp = list(parts)
                        newp[pos] = vs[0]
                        negs.append((pos, cat, joiner.join(newp)))
            duels.append(DuelBlock(query=block_text(data, blk, query_lang),
                                   gold=joiner.join(parts), negs=negs))
    return duels


def evaluate_duels(q_emb: np.ndarray, c_emb: np.ndarray,
                   duels: List[DuelBlock], spans: List[Tuple[int, int]],
                   categories, duel_sizes: List[int]) -> Dict:
    """Duel metrics for one (encoder, lang, k) cell, at every duel size.

    spans[b] = (start, end) of block b's candidates in c_emb, gold first.
    A block enters the size-D duel only if it has m ≥ D negatives; its win
    probability over all D-subsets is C(w, D)/C(m, D).
    """
    acc: Dict[int, Dict[str, list]] = {D: {"acc": [], "all": [], "m": []}
                                       for D in duel_sizes}
    skipped: Dict[int, int] = {D: 0 for D in duel_sizes}
    beat_num: Dict[str, int] = {c: 0 for c in categories}
    beat_den: Dict[str, int] = {c: 0 for c in categories}
    beat_pos_num: Dict[int, int] = {}
    beat_pos_den: Dict[int, int] = {}

    for b, (db, (s, e)) in enumerate(zip(duels, spans)):
        sims = q_emb[b] @ c_emb[s:e].T
        gold_sim, neg_sims = sims[0], sims[1:]
        m = len(neg_sims)
        beats = neg_sims < gold_sim       # strict: a tie counts against the gold
        for (pos, cat, _), won in zip(db.negs, beats):
            beat_num[cat] += int(won); beat_den[cat] += 1
            beat_pos_num[pos] = beat_pos_num.get(pos, 0) + int(won)
            beat_pos_den[pos] = beat_pos_den.get(pos, 0) + 1
        w = int(beats.sum())
        for D in duel_sizes:
            if m < D:
                skipped[D] += 1
                continue
            acc[D]["acc"].append(comb(w, D) / comb(m, D))
            acc[D]["all"].append(float(w == m))
            acc[D]["m"].append(m)

    n = len(duels)
    by_size: Dict[str, dict] = {}
    for D in duel_sizes:
        a = acc[D]
        by_size[str(D)] = {
            "n_used": len(a["acc"]), "n_skipped": skipped[D],
            "coverage": len(a["acc"]) / n if n else None,
            "mean_negatives": float(np.mean(a["m"])) if a["m"] else None,
            "duel_err": float(1.0 - np.mean(a["acc"])) if a["acc"] else None,
            "all_negatives_err": float(1.0 - np.mean(a["all"])) if a["all"] else None,
        }
    beat_by_cat = {c: (beat_num[c] / beat_den[c] if beat_den[c] else None)
                   for c in categories}
    beat_by_pos = {str(p): beat_pos_num[p] / beat_pos_den[p]
                   for p in sorted(beat_pos_den)}
    total_beat = sum(beat_num.values())
    total_den = sum(beat_den.values())
    return {
        "n_blocks": n, "by_duel_size": by_size,
        "detection_rate": total_beat / total_den if total_den else None,
        "beat_by_category": beat_by_cat, "beat_by_position": beat_by_pos,
    }


def _dry_run(args, loaded) -> None:
    for lang in args.langs:
        if lang == args.pivot:
            continue
        pool_lang = lang if args.direction == "en2xx" else args.pivot
        query_lang = args.pivot if args.direction == "en2xx" else lang
        corpus = [s for _sp, d in loaded for s in d.sentences[pool_lang]]
        pert = Perturber.for_corpus(corpus, pool_lang, args.seed)
        jn = "" if pool_lang in ("zh", "ja", "th") else " "
        logger.info(f"\n══ {pool_lang} ══")
        for k in args.k_list:
            duels = build_duels(loaded, k, query_lang, pool_lang,
                                args.categories, pert, jn, args.max_blocks)
            ms = [len(d.negs) for d in duels]
            cov = "  ".join(
                f"≥{D}: {sum(1 for m in ms if m >= D)} "
                f"({sum(1 for m in ms if m >= D) / max(1, len(duels)):.0%})"
                for D in args.duel_sizes)
            hist = {m: ms.count(m) for m in sorted(set(ms))}
            logger.info(f"  k={k}: {len(duels)} blocks  {cov}  m-histogram={hist}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+",
                    default=["comet:Unbabel/wmt22-comet-da"])
    ap.add_argument("--langs", nargs="+", default=["de", "es", "fr", "ru"])
    ap.add_argument("--pivot", default="en")
    ap.add_argument("--direction", choices=["en2xx", "xx2en"], default="en2xx")
    ap.add_argument("--k_list", nargs="+", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES),
                    choices=list(CATEGORIES))
    ap.add_argument("--duel_sizes", nargs="+", type=int, default=[6, 5, 4],
                    help="Negatives per duel. 6 is the full k=2 budget "
                         "(2 positions × 3 categories); 5 and 4 trade pool "
                         "size for k=2 coverage. All sizes are scored from "
                         "the same embeddings in one pass.")
    ap.add_argument("--flores_source", choices=["plus", "raw"], default="plus")
    ap.add_argument("--splits", nargs="+", default=["dev", "devtest"])
    ap.add_argument("--max_sents", type=int, default=None)
    ap.add_argument("--max_blocks", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", default="results/block_duel/emb_cache")
    ap.add_argument("--output", default="results/block_duel/block_duel.json")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    args.duel_sizes = sorted(set(args.duel_sizes), reverse=True)

    langs = [args.pivot] + [l for l in args.langs if l != args.pivot]
    loaded = [(sp, load_flores_source(args.flores_source, langs, sp, args.max_sents))
              for sp in args.splits]

    if args.dry_run:
        _dry_run(args, loaded)
        return

    device = pick_device(args.device)
    logger.info(f"Device: {device}")

    run_name = args.run_name or f"block_duel_{args.direction}"
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project, name=run_name, job_type="block_duel",
                tags=[f"dir:{args.direction}", f"flores:{args.flores_source}"]
                     + [_enc_tag(s) for s in args.encoders],
                config=vars(args))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"W&B init failed: {exc}")

    results: Dict = {"experiment": "block_duel",
                     "timestamp": datetime.now(timezone.utc).isoformat(),
                     "config": vars(args), "encoders": {}}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── pre-build every (lang, k) duel set once; encoders then just embed ──────
    pools: Dict[str, Dict[int, dict]] = {}
    for lang in langs:
        if lang == args.pivot:
            continue
        query_lang = args.pivot if args.direction == "en2xx" else lang
        pool_lang = lang if args.direction == "en2xx" else args.pivot
        corpus = [s for _sp, d in loaded for s in d.sentences[pool_lang]]
        pert = Perturber.for_corpus(corpus, pool_lang, args.seed)
        jn = "" if pool_lang in ("zh", "ja", "th") else " "
        pools[lang] = {}
        for k in args.k_list:
            duels = build_duels(loaded, k, query_lang, pool_lang,
                                args.categories, pert, jn, args.max_blocks)
            if len(duels) < 2:
                logger.warning(f"  [{lang}] k={k}: {len(duels)} blocks — skipping")
                continue
            texts, spans = [], []
            for d in duels:
                s = len(texts)
                texts.append(d.gold)
                texts.extend(t for _p, _c, t in d.negs)
                spans.append((s, len(texts)))
            cov = "  ".join(f"≥{D}:{sum(1 for d in duels if len(d.negs) >= D)}"
                            for D in args.duel_sizes)
            pools[lang][k] = {"duels": duels, "texts": texts, "spans": spans}
            logger.info(f"  [{lang}] k={k}: {len(duels)} blocks ({cov}), "
                        f"{len(texts)} candidate texts")

    for spec in args.encoders:
        logger.info(f"\n══ {spec} ══")
        emb = build_embedder(spec, device)
        per_lang: Dict[str, Dict[str, dict]] = {}
        for lang, by_k in pools.items():
            per_lang[lang] = {}
            for k, P in by_k.items():
                tag = f"duel_{args.direction}_{lang}_k{k}"
                q = cached_embed(emb, [d.query for d in P["duels"]], f"{tag}_q",
                                 args.cache_dir, args.batch_size)
                c = cached_embed(emb, P["texts"], f"{tag}_c",
                                 args.cache_dir, args.batch_size)
                m = evaluate_duels(q, c, P["duels"], P["spans"],
                                   args.categories, args.duel_sizes)
                per_lang[lang][str(k)] = m
                errs = "  ".join(
                    f"d{D}={_fmt(m['by_duel_size'][str(D)]['duel_err'])}"
                    f"(n={m['by_duel_size'][str(D)]['n_used']})"
                    for D in args.duel_sizes)
                logger.info(f"  {lang} k={k}: {errs} "
                            f"detect={_fmt(m['detection_rate'])}")
                if wandb_run is not None:
                    pref = f"{emb.name}/{lang}/k{k}"
                    log = {f"{pref}/detection_rate": m["detection_rate"]}
                    for D in args.duel_sizes:
                        bs = m["by_duel_size"][str(D)]
                        log[f"{pref}/duel_err_d{D}"] = bs["duel_err"]
                        log[f"{pref}/coverage_d{D}"] = bs["coverage"]
                    wandb_run.log({k2: v for k2, v in log.items() if v is not None})

        mean_by_k: Dict[str, dict] = {}
        for k in args.k_list:
            cells = [per_lang[l][str(k)] for l in per_lang if str(k) in per_lang[l]]
            if not cells:
                continue
            by_size = {}
            for D in args.duel_sizes:
                key = str(D)
                by_size[key] = {
                    "duel_err": _nanmean([c["by_duel_size"][key]["duel_err"] for c in cells]),
                    "all_negatives_err": _nanmean(
                        [c["by_duel_size"][key]["all_negatives_err"] for c in cells]),
                    "coverage": _nanmean([c["by_duel_size"][key]["coverage"] for c in cells]),
                    "n_used": int(np.sum([c["by_duel_size"][key]["n_used"] for c in cells])),
                    "n_skipped": int(np.sum([c["by_duel_size"][key]["n_skipped"] for c in cells])),
                }
            mean_by_k[str(k)] = {
                "by_duel_size": by_size,
                "detection_rate": _nanmean([c["detection_rate"] for c in cells]),
                "beat_by_category": {cat: _nanmean([c["beat_by_category"][cat] for c in cells])
                                     for cat in args.categories},
                "beat_by_position": _mean_dicts([c["beat_by_position"] for c in cells]),
            }
        per_lang["_mean_by_k"] = mean_by_k
        results["encoders"][emb.name] = per_lang

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        del emb
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plots = _plots(results, out_path.parent / "plots", args.categories,
                   args.duel_sizes)
    if wandb_run is not None:
        import wandb
        for p in plots:
            wandb_run.log({f"plots/{p.stem}": wandb.Image(str(p))})
        wandb_run.finish()
    logger.info(f"\nResults → {out_path}")


def _fmt(v) -> str:
    return f"{v:.4f}" if v is not None else "—"


def _nanmean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def _mean_dicts(dicts):
    keys = sorted({k for d in dicts for k in d}, key=int)
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts if d.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def _plots(results: Dict, plot_dir: Path, categories, duel_sizes) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    encoders = list(results["encoders"])

    def ks(enc):
        return sorted(results["encoders"][enc]["_mean_by_k"], key=int)

    # 1. headline — duel error against block length, one panel per duel size
    fig, axes = plt.subplots(1, len(duel_sizes),
                             figsize=(4.6 * len(duel_sizes), 4.4),
                             squeeze=False, sharey=True)
    for ax, D in zip(axes[0], duel_sizes):
        for enc in encoders:
            mk = results["encoders"][enc]["_mean_by_k"]
            x = [int(k) for k in ks(enc)]
            y = [mk[k]["by_duel_size"][str(D)]["duel_err"] for k in ks(enc)]
            if any(v is not None for v in y):
                ax.plot(x, y, marker="o", label=enc)
        ax.axhline(D / (D + 1), color="grey", ls=":", lw=1,
                   label=f"chance ({D}/{D + 1})")
        ax.set_xlabel("block length k (sentences)")
        ax.set_title(f"1 gold vs {D} negatives", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("duel error (mean over all D-subsets)")
    axes[0][-1].legend(fontsize=7)
    p = plot_dir / "duel_err_vs_length.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # 2. coverage — fraction of blocks with ≥ D negatives (encoder-independent)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    enc0 = encoders[0]
    mk0 = results["encoders"][enc0]["_mean_by_k"]
    for D in duel_sizes:
        x = [int(k) for k in ks(enc0)]
        y = [mk0[k]["by_duel_size"][str(D)]["coverage"] for k in ks(enc0)]
        ax.plot(x, y, marker="s", label=f"D = {D}")
    ax.set_xlabel("block length k (sentences)")
    ax.set_ylabel("fraction of blocks with ≥ D negatives")
    ax.set_ylim(0, 1.05)
    ax.set_title("Duel coverage vs block length (mean over languages)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    p = plot_dir / "duel_coverage.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # 3. which category the gold fails to beat, per k
    fig, axes = plt.subplots(1, len(encoders), figsize=(4.2 * len(encoders), 4),
                             squeeze=False, sharey=True)
    for ax, enc in zip(axes[0], encoders):
        mk = results["encoders"][enc]["_mean_by_k"]
        x = np.arange(len(ks(enc)))
        w = 0.8 / max(1, len(categories))
        for j, cat in enumerate(categories):
            y = [mk[k]["beat_by_category"][cat] for k in ks(enc)]
            ax.bar(x + j * w, [v if v is not None else 0 for v in y],
                   width=w, label=cat)
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(ks(enc))
        ax.set_xlabel("k"); ax.set_title(enc, fontsize=9)
    axes[0][0].set_ylabel("P[gold beats the negative]")
    axes[0][-1].legend(fontsize=8)
    p = plot_dir / "beat_by_category.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    for pth in paths:
        logger.info(f"  [plot] {pth}")
    return paths


if __name__ == "__main__":
    main()
