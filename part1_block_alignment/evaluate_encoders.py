"""Encoder cosine on the candidate pools: can an encoder still spot one
perturbed sentence once it is buried in a k-sentence block?

    python -m part1_block_alignment.evaluate_encoders \
        --encoders comet:Unbabel/wmt22-comet-da hf-mean:xlm-roberta-large labse e5 \
        --langs de es fr ru --k_list 2 3 4 5 --backend spacy --wandb_project comet-block-xsim

Reads data/pools_<backend>_<lang>_k<k>.json (perturb.py). Each (encoder, lang, k)
cell scores FOUR nested pools off one similarity matrix (`pool_ablation`):

    true_only           all true blocks                      classic xsim
    true+perturbed      all true blocks + all negatives      classic xsim++
    gold+all_perturbed  own gold + all negatives             distractors dropped
    gold+own_perturbed  own gold + own negatives only        pure dilution

Writes results/encoder_cosine.json (RunResult[PoolMetrics]) and five plots
results/plots/<output stem>_*.png.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.auth import init_wandb
from part1_block_alignment import DATA_DIR, RESULTS_DIR
from part1_block_alignment.models import (Candidate, CandidatePool, EncoderRunResult, ModelCells,
                                          PoolMetrics, mean_by_k)
from part1_block_alignment.perturb import load_pool, pool_categories

logger = logging.getLogger(__name__)


# ── retrieval + metrics (absolute margin = plain cosine, per xSIM++ footnote 6) ─
def _err_on_subset(sim: np.ndarray, cols: np.ndarray,
                   true_idx: np.ndarray) -> Tuple[float, np.ndarray]:
    """Error rate when retrieving over `cols` (pool positions). Returns
    (error_rate, predicted pool positions)."""
    pred = cols[sim[:, cols].argmax(1)]
    err = float((pred != true_idx).mean())
    return err, pred


def evaluate_blocks(q_emb: np.ndarray, p_emb: np.ndarray,
                    cands: List[Candidate], true_idx: List[int],
                    categories: Sequence[str]) -> PoolMetrics:
    """All xsim / xsim++ numbers for one (encoder, lang, k) cell.

    Four candidate pools are scored off the *same* similarity matrix, so the
    ablation costs nothing — no candidate is re-encoded:

      true_only           every true target block, no negatives    → classic xsim
      true+perturbed      everything                               → classic xsim++
      gold+all_perturbed  the query's own gold + every perturbed block in the
                          corpus; the *other* true target blocks — the classic
                          xsim distractors — are dropped
      gold+own_perturbed  the query's own gold + only its own single-edit hard
                          negatives; nothing but the injected perturbation is
                          left to separate, so this is the pure dilution number

    The last two are what "the pool should be the hard negatives" means. The
    gap `true+perturbed` − `gold+all_perturbed` is exactly what the classic
    distractors contribute; `gold+own_perturbed` removes cross-block confusion
    entirely. Caveat: `gold+own_perturbed` still has a pool that *grows* with k
    (up to 3·k·variants negatives) — evaluate_duel.py is the protocol that also
    pins the candidate count, of which this is the D = m (all negatives) case.
    """
    sim = q_emb @ p_emb.T                                  # (N_blocks, M_cands)
    n = sim.shape[0]
    rows = np.arange(n)
    true_arr = np.asarray(true_idx)
    cand_block = np.asarray([c.block_id for c in cands])
    cand_kind = np.asarray([c.kind for c in cands])
    cand_cat = np.asarray([c.category or "" for c in cands])
    cand_pos = np.asarray([-1 if c.position is None else c.position for c in cands])

    true_cols = np.asarray(true_idx)
    all_cols = np.arange(len(cands))
    is_pert = cand_kind == "perturbed"
    own = cand_block[None, :] == rows[:, None]               # (N, M)
    own_pert = own & is_pert[None, :]
    gold_oh = np.zeros_like(own)
    gold_oh[rows, true_arr] = True

    xsim_err, _ = _err_on_subset(sim, true_cols, true_arr)
    xsimpp_err, pred_all = _err_on_subset(sim, all_cols, true_arr)

    # ── pools that drop the classic xsim distractors ────────────────────────
    # The candidate set now differs per query, so mask columns row-wise instead
    # of slicing a shared column list.
    def _err_rowwise(colmask: np.ndarray, keep: Optional[np.ndarray] = None):
        """(error, predictions, n_rows_scored) over the rows in `keep`."""
        m = colmask if keep is None else colmask[keep]
        s = sim if keep is None else sim[keep]
        t = true_arr if keep is None else true_arr[keep]
        if s.shape[0] == 0:
            return None, np.zeros(0, dtype=int), 0
        pred = np.where(m, s, -np.inf).argmax(1)
        return float((pred != t).mean()), pred, int(s.shape[0])

    hard_err, _, _ = _err_rowwise(gold_oh | is_pert[None, :])

    # blocks with no negative of their own would score a free 0 — exclude them
    has_own = own_pert.any(1)
    own_err, pred_own, n_own = _err_rowwise(gold_oh | own_pert, has_own)

    own_breakdown: Dict[str, Optional[float]] = {}
    if n_own:
        ok = pred_own == true_arr[has_own]
        own_breakdown["correct"] = float(ok.mean())
        for c in categories:
            own_breakdown[c] = float(((~ok) & (cand_cat[pred_own] == c)).mean())

    # A bigger own-pool is a harder pool: a coin-flip encoder scores m/(m+1),
    # and m grows with k. Without this baseline the dilution curve is unreadable.
    n_own_neg = own_pert.sum(1)
    own_chance = (float((n_own_neg[has_own] / (n_own_neg[has_own] + 1)).mean())
                  if has_own.any() else None)

    own_by_cat: Dict[str, Optional[float]] = {}
    own_n_by_cat: Dict[str, int] = {}
    for c in categories:
        m_cat = own_pert & (cand_cat == c)[None, :]
        keep = m_cat.any(1)
        e, _, nk = _err_rowwise(gold_oh | m_cat, keep)
        own_by_cat[c], own_n_by_cat[c] = e, nk

    # error typology on the full pool
    correct = pred_all == true_arr
    pred_block = cand_block[pred_all]
    own_perturbed = (~correct) & (pred_block == rows)
    misaligned = (~correct) & (pred_block != rows)
    breakdown = {"correct": float(correct.mean()),
                 "misaligned": float(misaligned.mean())}
    for c in categories:
        breakdown[c] = float((own_perturbed & (cand_cat[pred_all] == c)).mean())

    # per-category pools (paper Table 4 rows) and every category combination
    def _pool_err(cats: Sequence[str]) -> float:
        mask = (cand_kind == "true") | np.isin(cand_cat, list(cats))
        return _err_on_subset(sim, all_cols[mask], true_arr)[0]

    per_category = {c: _pool_err([c]) for c in categories}
    combos: Dict[str, float] = {}
    for r in range(2, len(categories) + 1):
        for combo in combinations(categories, r):
            combos["+".join(combo)] = _pool_err(combo)

    # push-apart detection: P[cos(q, gold) > cos(q, negative)] over own negatives
    gold_sim = sim[rows, true_arr]
    det_by_cat: Dict[str, Optional[float]] = {}
    det_by_pos: Dict[str, Optional[float]] = {}
    cov: Dict[str, int] = {}
    for c in categories:
        m = own_pert & (cand_cat == c)[None, :]
        cov[c] = int(m.sum())
        det_by_cat[c] = float((gold_sim[:, None] > sim)[m].mean()) if m.any() else None
    for pos in sorted({int(p) for p in cand_pos if p >= 0}):
        m = own_pert & (cand_pos == pos)[None, :]
        det_by_pos[str(pos)] = float((gold_sim[:, None] > sim)[m].mean()) if m.any() else None
    detection = (float((gold_sim[:, None] > sim)[own_pert].mean())
                 if own_pert.any() else None)

    # margin between the gold block and the single best negative
    neg = sim.copy()
    neg[rows, true_arr] = -np.inf
    margin = float((gold_sim - neg.max(1)).mean())
    # margin against the *hardest own perturbation* only (isolates dilution)
    own_neg = np.where(own_pert, sim, -np.inf)
    margin_own = (float((gold_sim[has_own] - own_neg.max(1)[has_own]).mean())
                  if has_own.any() else None)

    return PoolMetrics(
        n_blocks=n, pool_size=len(cands),
        xsim_err=xsim_err, xsimpp_err=xsimpp_err,
        # pools with the classic xsim distractors dropped
        xsimpp_err_hard_pool=hard_err,
        xsimpp_err_own_pool=own_err,
        pool_ablation={"true_only": xsim_err,
                       "true+perturbed": xsimpp_err,
                       "gold+all_perturbed": hard_err,
                       "gold+own_perturbed": own_err},
        own_pool_breakdown=own_breakdown,
        own_pool_err_by_category=own_by_cat,
        own_pool_n_blocks=n_own,
        own_pool_n_blocks_by_category=own_n_by_cat,
        own_pool_chance_err=own_chance,
        own_pool_negatives_per_block=(float(n_own_neg[has_own].mean())
                                      if has_own.any() else 0.0),
        error_breakdown=breakdown,
        per_category_err=per_category, category_combos=combos,
        detection_rate=detection, detection_by_category=det_by_cat,
        detection_by_position=det_by_pos,
        n_negatives_by_category=cov,
        margin_vs_best_negative=margin,
        margin_vs_best_own_perturbation=margin_own)


# ── main ──────────────────────────────────────────────────────────────────────
def _fmt(v) -> str:
    return "  n/a " if v is None else f"{v:.4f}"


def load_pools(backend: str, langs: Sequence[str], k_list: Sequence[int]
               ) -> Dict[str, Dict[int, CandidatePool]]:
    pools: Dict[str, Dict[int, CandidatePool]] = {}
    for lang in langs:
        pools[lang] = {}
        for k in k_list:
            pool = load_pool(backend, lang, k)
            if pool.n_blocks < 2:
                logger.warning(f"  [{lang}] k={k}: {pool.n_blocks} blocks — skipping")
                continue
            pools[lang][k] = pool
            counts = ", ".join(f"{c}:{pool.variant_counts[c]}" for c in pool_categories(pool))
            logger.info(f"  [{lang}] k={k}: {pool.n_blocks} blocks, "
                        f"pool={len(pool.candidates)}, variants={{{counts}}}")
    return pools


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", default=["comet:Unbabel/wmt22-comet-da"],
                    help="comet:<id-or-ckpt> | hf-mean:<hf-id> | labse | e5[:<hf-id>]")
    ap.add_argument("--langs", nargs="+", default=["de", "es", "fr", "ru"])
    ap.add_argument("--k_list", nargs="+", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--backend", choices=["spacy", "heuristic"], default="spacy",
                    help="Which pools to read (the perturber that built them).")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache_dir", default=str(DATA_DIR / "emb_cache"))
    ap.add_argument("--output", default=str(RESULTS_DIR / "encoder_cosine.json"))
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--run_name", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from common.encoders import build_embedder, cached_embed, enc_tag, pick_device

    pools = load_pools(args.backend, args.langs, args.k_list)
    device = pick_device(args.device)
    logger.info(f"Device: {device}")

    run_name = args.run_name or f"encoder_cosine_{args.backend}"
    wandb_run = init_wandb(args.wandb_project, run_name, "evaluate_encoders",
                           tags=[f"backend:{args.backend}"] + [enc_tag(s) for s in args.encoders],
                           config=vars(args))

    result = EncoderRunResult(experiment="block_xsim++",
                              timestamp=datetime.now(timezone.utc).isoformat(),
                              config=vars(args))
    out_path = Path(args.output)

    for spec in args.encoders:
        logger.info(f"\n== {spec} ==")
        emb = build_embedder(spec, device)
        by_lang: Dict[str, Dict[str, PoolMetrics]] = {}
        for lang, by_k in pools.items():
            by_lang[lang] = {}
            for k, pool in by_k.items():
                tag = f"pool_{args.backend}_{lang}_k{k}"
                q = cached_embed(emb, pool.queries, f"{tag}_q", args.cache_dir, args.batch_size)
                c = cached_embed(emb, [x.text for x in pool.candidates], f"{tag}_c",
                                 args.cache_dir, args.batch_size)
                m = evaluate_blocks(q, c, pool.candidates, pool.true_index, pool_categories(pool))
                m.variant_stats = pool.variant_counts
                by_lang[lang][str(k)] = m
                pa = m.pool_ablation
                line = (f"  {lang} k={k}: own={_fmt(pa['gold+own_perturbed'])}"
                        f"/chance={_fmt(m.own_pool_chance_err)} "
                        f"hard={_fmt(pa['gold+all_perturbed'])}"
                        f"  [classic: xsim={m.xsim_err:.4f} xsim++={m.xsimpp_err:.4f}]")
                if m.detection_rate is not None:
                    line += (f" detect={m.detection_rate:.4f}"
                             f" margin_own={m.margin_vs_best_own_perturbation:.4f}")
                logger.info(line)
                if wandb_run is not None:
                    pref = f"{emb.name}/{lang}/k{k}"
                    log = {f"{pref}/xsim_err": m.xsim_err,
                           f"{pref}/xsimpp_err": m.xsimpp_err,
                           f"{pref}/margin": m.margin_vs_best_negative,
                           f"{pref}/err_hard_pool": pa["gold+all_perturbed"],
                           f"{pref}/err_own_pool": pa["gold+own_perturbed"],
                           f"{pref}/err_own_pool_chance": m.own_pool_chance_err,
                           f"{pref}/detection_rate": m.detection_rate,
                           f"{pref}/margin_own": m.margin_vs_best_own_perturbation}
                    log.update({f"{pref}/err_own_{cat}": v
                                for cat, v in m.own_pool_err_by_category.items()})
                    log.update({f"{pref}/err_{cat}": v for cat, v in m.per_category_err.items()})
                    wandb_run.log({key: v for key, v in log.items() if v is not None})

        result.models[emb.name] = ModelCells[PoolMetrics](
            spec=spec, by_lang=by_lang, mean_by_k=mean_by_k(by_lang))
        result.save(out_path)                       # incremental
        del emb
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plots = _plots(result, out_path.parent / "plots", out_path.stem)
    if wandb_run is not None:
        import wandb
        for p in plots:
            wandb_run.log({f"plots/{p.stem}": wandb.Image(str(p))})
        wandb_run.finish()
    logger.info(f"\nResults -> {out_path}")


def _plots(result: EncoderRunResult, plot_dir: Path, prefix: str) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    encoders = list(result.models)

    def mean(enc) -> Dict[str, PoolMetrics]:
        return result.models[enc].mean_by_k

    def ks(enc):
        return sorted(mean(enc), key=int)

    categories = list(next(iter(mean(encoders[0]).values())).per_category_err or {})

    def save(fig, name):
        p = plot_dir / f"{prefix}_{name}.png"
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # 1. headline — the hard-negative pools: no other block's true target is a
    #    candidate, so only the injected edit can separate gold from distractor
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, enc in enumerate(encoders):
        mk = mean(enc)
        x = [int(k) for k in ks(enc)]
        color = f"C{i}"
        y_all = [mk[k].pool_ablation["gold+all_perturbed"] for k in ks(enc)]
        y_own = [mk[k].pool_ablation["gold+own_perturbed"] for k in ks(enc)]
        if any(v is not None for v in y_all):
            ax.plot(x, y_all, marker="^", ls="--", color=color, alpha=0.5,
                    label=f"{enc} — gold + all negatives")
        if any(v is not None for v in y_own):
            ax.plot(x, y_own, marker="d", ls="-", color=color,
                    label=f"{enc} — gold + own negatives")
    # chance: the gold ranked at random among its own m negatives, m grows with k
    mk0 = mean(encoders[0])
    kk = ks(encoders[0])
    chance = [mk0[k].own_pool_chance_err for k in kk]
    if any(v is not None for v in chance):
        ax.plot([int(k) for k in kk], chance, color="grey", ls=":", lw=1.2,
                label="chance (gold ranked at random in its own pool)")
    ax.set_xticks(sorted({int(k) for enc in encoders for k in ks(enc)}))
    ax.set_xlabel("block length k (sentences)")
    ax.set_ylabel("retrieval error rate")
    ax.set_title("Hard-negative retrieval error vs block length (mean over languages)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    save(fig, "hard_negative_error")

    # 2. dilution — can the encoder still push the perturbed block away?
    fig, ax = plt.subplots(figsize=(8, 5))
    for enc in encoders:
        mk = mean(enc)
        x = [int(k) for k in ks(enc)]
        y = [mk[k].detection_rate for k in ks(enc)]
        if any(v is not None for v in y):
            ax.plot(x, y, marker="o", label=enc)
    ax.axhline(0.5, color="grey", ls=":", lw=1, label="chance")
    ax.set_xlabel("block length k (sentences)")
    ax.set_ylabel("P[cos(query, gold) > cos(query, perturbed)]")
    ax.set_title("Single-sentence error detection vs block length")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    save(fig, "detection_vs_length")

    # 3. per-category error, own pool (gold + this block's own negatives of one
    #    category) — the classic distractors are out of this one too
    fig, axes = plt.subplots(1, len(encoders), figsize=(4.2 * len(encoders), 4),
                             squeeze=False, sharey=True)
    for ax, enc in zip(axes[0], encoders):
        mk = mean(enc)
        x = np.arange(len(ks(enc)))
        w = 0.8 / max(1, len(categories))
        for j, cat in enumerate(categories):
            ax.bar(x + j * w,
                   [(mk[k].own_pool_err_by_category or {}).get(cat) or 0.0 for k in ks(enc)],
                   width=w, label=cat)
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(ks(enc))
        ax.set_xlabel("k"); ax.set_title(enc, fontsize=9)
    axes[0][0].set_ylabel("error (pool = gold + own negatives, one category)")
    axes[0][-1].legend(fontsize=8)
    save(fig, "error_by_category")

    # 4. is the encoder blind to errors late in the block?
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, enc in enumerate(encoders):
        mk = mean(enc)
        for k in ks(enc):
            d = mk[k].detection_by_position or {}
            pos = sorted((p for p, v in d.items() if v is not None), key=int)
            if pos:
                ax.plot([int(p) for p in pos], [d[p] for p in pos], marker="o",
                        color=f"C{i}", alpha=0.4 + 0.5 * int(k) / max(int(x) for x in ks(enc)),
                        label=f"{enc} k={k}")
    ax.set_xlabel("position of the perturbed sentence in the block")
    ax.set_ylabel("detection rate")
    ax.set_title("Position of the injected error vs detectability")
    ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=2)
    save(fig, "detection_by_position")

    # 5. pool ablation — what do the classic xsim distractors actually buy?
    pools = [("true_only", ":", "o", "true blocks only (xsim)"),
             ("true+perturbed", "-", "s", "true blocks + negatives (xsim++)"),
             ("gold+all_perturbed", "--", "^", "gold + all negatives"),
             ("gold+own_perturbed", "-.", "d", "gold + own negatives")]
    fig, axes = plt.subplots(1, len(encoders), figsize=(4.2 * len(encoders), 4),
                             squeeze=False, sharey=True)
    for ax, enc in zip(axes[0], encoders):
        mk = mean(enc)
        x = [int(k) for k in ks(enc)]
        for pool, ls, mk_, lab in pools:
            y = [(mk[k].pool_ablation or {}).get(pool) for k in ks(enc)]
            if any(v is not None for v in y):
                xs = [xi for xi, yi in zip(x, y) if yi is not None]
                ax.plot(xs, [v for v in y if v is not None], ls=ls, marker=mk_, label=lab)
        ax.set_xticks(x)
        ax.set_xlabel("block length k"); ax.set_title(enc, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("retrieval error rate")
    axes[0][-1].legend(fontsize=7)
    save(fig, "pool_ablation")

    for pth in paths:
        logger.info(f"  [plot] {pth}")
    return paths


if __name__ == "__main__":
    main()
