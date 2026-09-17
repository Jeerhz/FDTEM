"""The 1-vs-(D+1) duel: retrieval with a FIXED number of candidates at every k.

    python -m part1_block_alignment.evaluate_duel \
        --encoders comet:Unbabel/wmt22-comet-da labse e5 --langs de es fr ru \
        --k_list 2 3 4 5 --duel_sizes 6 5 4 --backend spacy

In evaluate_encoders.py the candidate pool changes with k (fewer blocks, more
negatives per block), a residual confound. Here every retrieval is the gold block
against D single-error negatives of the same block. D = 6 is the k=2 budget
(2 positions x 3 categories, one variant per pair). A block with m negatives is
scored over ALL C(m, D) subsets in closed form: the gold wins a subset iff it
beats every member, so P[win] = C(w, D) / C(m, D) with w the negatives it beats.
Blocks with m < D are skipped and counted (coverage). All duel sizes are scored
from the same embeddings in one pass.

Negatives = the first variant of every (position, category) of the pool
(data/pools_<backend>_<lang>_k<k>.json). Writes results/duel.json
(RunResult[DuelMetrics]) and results/plots/<output stem>_*.png.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from common.auth import init_wandb
from part1_block_alignment import DATA_DIR, RESULTS_DIR
from part1_block_alignment.evaluate_encoders import load_pools
from part1_block_alignment.models import (DuelItem, DuelMetrics, DuelRunResult, DuelSizeMetrics,
                                          ModelCells, mean_by_k)
from part1_block_alignment.perturb import pool_categories

logger = logging.getLogger(__name__)


def evaluate_duels(q_emb: np.ndarray, c_emb: np.ndarray,
                   duels: List[DuelItem], spans: List[Tuple[int, int]],
                   categories: Sequence[str], duel_sizes: List[int]) -> DuelMetrics:
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
        for neg, won in zip(db.negatives, beats):
            beat_num[neg.category] += int(won); beat_den[neg.category] += 1
            beat_pos_num[neg.position] = beat_pos_num.get(neg.position, 0) + int(won)
            beat_pos_den[neg.position] = beat_pos_den.get(neg.position, 0) + 1
        w = int(beats.sum())
        for D in duel_sizes:
            if m < D:
                skipped[D] += 1
                continue
            acc[D]["acc"].append(comb(w, D) / comb(m, D))
            acc[D]["all"].append(float(w == m))
            acc[D]["m"].append(m)

    n = len(duels)
    by_size: Dict[str, DuelSizeMetrics] = {}
    for D in duel_sizes:
        a = acc[D]
        by_size[str(D)] = DuelSizeMetrics(
            n_used=len(a["acc"]), n_skipped=skipped[D],
            coverage=len(a["acc"]) / n if n else None,
            mean_negatives=float(np.mean(a["m"])) if a["m"] else None,
            duel_err=float(1.0 - np.mean(a["acc"])) if a["acc"] else None,
            all_negatives_err=float(1.0 - np.mean(a["all"])) if a["all"] else None)
    beat_by_cat = {c: (beat_num[c] / beat_den[c] if beat_den[c] else None)
                   for c in categories}
    beat_by_pos = {str(p): beat_pos_num[p] / beat_pos_den[p]
                   for p in sorted(beat_pos_den)}
    total_beat = sum(beat_num.values())
    total_den = sum(beat_den.values())
    return DuelMetrics(
        n_blocks=n, by_duel_size=by_size,
        detection_rate=total_beat / total_den if total_den else None,
        beat_by_category=beat_by_cat, beat_by_position=beat_by_pos)


def _fmt(v) -> str:
    return f"{v:.4f}" if v is not None else "—"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", default=["comet:Unbabel/wmt22-comet-da"])
    ap.add_argument("--langs", nargs="+", default=["de", "es", "fr", "ru"])
    ap.add_argument("--k_list", nargs="+", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--duel_sizes", nargs="+", type=int, default=[6, 5, 4],
                    help="Negatives per duel. 6 is the full k=2 budget; 5 and 4 trade "
                         "pool size for k=2 coverage. All are scored in one pass.")
    ap.add_argument("--backend", choices=["spacy", "heuristic"], default="spacy",
                    help="Which pools to read (the perturber that built them).")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache_dir", default=str(DATA_DIR / "emb_cache"))
    ap.add_argument("--output", default=str(RESULTS_DIR / "duel.json"))
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()
    args.duel_sizes = sorted(set(args.duel_sizes), reverse=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from common.encoders import build_embedder, cached_embed, enc_tag, pick_device

    # one duel set per (lang, k): gold first, then its negatives, in one text list
    tasks: Dict[str, Dict[int, dict]] = {}
    for lang, by_k in load_pools(args.backend, args.langs, args.k_list).items():
        tasks[lang] = {}
        for k, pool in by_k.items():
            duels = pool.duel_items()
            texts, spans = [], []
            for d in duels:
                s = len(texts)
                texts.append(d.gold)
                texts.extend(n.text for n in d.negatives)
                spans.append((s, len(texts)))
            cov = "  ".join(f">={D}:{sum(1 for d in duels if len(d.negatives) >= D)}"
                            for D in args.duel_sizes)
            tasks[lang][k] = {"duels": duels, "texts": texts, "spans": spans,
                              "categories": pool_categories(pool)}
            logger.info(f"  [{lang}] k={k}: {len(duels)} blocks ({cov}), "
                        f"{len(texts)} candidate texts")

    device = pick_device(args.device)
    logger.info(f"Device: {device}")
    run_name = args.run_name or f"duel_{args.backend}"
    wandb_run = init_wandb(args.wandb_project, run_name, "evaluate_duel",
                           tags=[f"backend:{args.backend}"] + [enc_tag(s) for s in args.encoders],
                           config=vars(args))

    result = DuelRunResult(experiment="block_duel",
                           timestamp=datetime.now(timezone.utc).isoformat(),
                           config=vars(args))
    out_path = Path(args.output)

    for spec in args.encoders:
        logger.info(f"\n== {spec} ==")
        emb = build_embedder(spec, device)
        by_lang: Dict[str, Dict[str, DuelMetrics]] = {}
        for lang, by_k in tasks.items():
            by_lang[lang] = {}
            for k, T in by_k.items():
                tag = f"duel_{args.backend}_{lang}_k{k}"
                q = cached_embed(emb, [d.query for d in T["duels"]], f"{tag}_q",
                                 args.cache_dir, args.batch_size)
                c = cached_embed(emb, T["texts"], f"{tag}_c", args.cache_dir, args.batch_size)
                m = evaluate_duels(q, c, T["duels"], T["spans"], T["categories"], args.duel_sizes)
                by_lang[lang][str(k)] = m
                errs = "  ".join(f"d{D}={_fmt(m.by_duel_size[str(D)].duel_err)}"
                                 f"(n={m.by_duel_size[str(D)].n_used})" for D in args.duel_sizes)
                logger.info(f"  {lang} k={k}: {errs} detect={_fmt(m.detection_rate)}")
                if wandb_run is not None:
                    pref = f"{emb.name}/{lang}/k{k}"
                    log = {f"{pref}/detection_rate": m.detection_rate}
                    for D in args.duel_sizes:
                        bs = m.by_duel_size[str(D)]
                        log[f"{pref}/duel_err_d{D}"] = bs.duel_err
                        log[f"{pref}/coverage_d{D}"] = bs.coverage
                    wandb_run.log({k2: v for k2, v in log.items() if v is not None})

        result.models[emb.name] = ModelCells[DuelMetrics](
            spec=spec, by_lang=by_lang, mean_by_k=mean_by_k(by_lang))
        result.save(out_path)                       # incremental
        del emb
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plots = _plots(result, out_path.parent / "plots", out_path.stem, args.duel_sizes)
    if wandb_run is not None:
        import wandb
        for p in plots:
            wandb_run.log({f"plots/{p.stem}": wandb.Image(str(p))})
        wandb_run.finish()
    logger.info(f"\nResults -> {out_path}")


def _plots(result: DuelRunResult, plot_dir: Path, prefix: str, duel_sizes) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    encoders = list(result.models)

    def mean(enc) -> Dict[str, DuelMetrics]:
        return result.models[enc].mean_by_k

    def ks(enc):
        return sorted(mean(enc), key=int)

    categories = list(next(iter(mean(encoders[0]).values())).beat_by_category)

    def save(fig, name):
        p = plot_dir / f"{prefix}_{name}.png"
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # 1. headline — duel error against block length, one panel per duel size
    fig, axes = plt.subplots(1, len(duel_sizes), figsize=(4.6 * len(duel_sizes), 4.4),
                             squeeze=False, sharey=True)
    for ax, D in zip(axes[0], duel_sizes):
        for enc in encoders:
            mk = mean(enc)
            x = [int(k) for k in ks(enc)]
            y = [mk[k].by_duel_size[str(D)].duel_err for k in ks(enc)]
            if any(v is not None for v in y):
                ax.plot(x, y, marker="o", label=enc)
        ax.axhline(D / (D + 1), color="grey", ls=":", lw=1, label=f"chance ({D}/{D + 1})")
        ax.set_xlabel("block length k (sentences)")
        ax.set_title(f"1 gold vs {D} negatives", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("duel error (mean over all D-subsets)")
    axes[0][-1].legend(fontsize=7)
    save(fig, "duel_err_vs_length")

    # 2. coverage — fraction of blocks with ≥ D negatives (encoder-independent)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    enc0 = encoders[0]
    mk0 = mean(enc0)
    for D in duel_sizes:
        x = [int(k) for k in ks(enc0)]
        y = [mk0[k].by_duel_size[str(D)].coverage for k in ks(enc0)]
        ax.plot(x, y, marker="s", label=f"D = {D}")
    ax.set_xlabel("block length k (sentences)")
    ax.set_ylabel("fraction of blocks with ≥ D negatives")
    ax.set_ylim(0, 1.05)
    ax.set_title("Duel coverage vs block length (mean over languages)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    save(fig, "duel_coverage")

    # 3. which category the gold fails to beat, per k
    fig, axes = plt.subplots(1, len(encoders), figsize=(4.2 * len(encoders), 4),
                             squeeze=False, sharey=True)
    for ax, enc in zip(axes[0], encoders):
        mk = mean(enc)
        x = np.arange(len(ks(enc)))
        w = 0.8 / max(1, len(categories))
        for j, cat in enumerate(categories):
            y = [mk[k].beat_by_category.get(cat) for k in ks(enc)]
            ax.bar(x + j * w, [v if v is not None else 0 for v in y], width=w, label=cat)
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(ks(enc))
        ax.set_xlabel("k"); ax.set_title(enc, fontsize=9)
    axes[0][0].set_ylabel("P[gold beats the negative]")
    axes[0][-1].legend(fontsize=8)
    save(fig, "beat_by_category")

    for pth in paths:
        logger.info(f"  [plot] {pth}")
    return paths


if __name__ == "__main__":
    main()
