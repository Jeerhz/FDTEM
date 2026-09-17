"""Align translations with the COMET SCORE instead of the encoder cosine, on the
same pools.

    python -m part1_block_alignment.evaluate_comet_score \
        --scorers comet-score:Unbabel/wmt22-cometkiwi-da encoder-cos:Unbabel/wmt22-cometkiwi-da \
                  encoder-cos:Unbabel/wmt22-comet-da --langs de es fr ru --k_list 1 2 3 4 5
    python -m part1_block_alignment.evaluate_comet_score --dry_run --langs de   # task sizes only

Two decision rules over identical candidates:

    encoder-cos:<ckpt>    argmax_c  cos(embed(src block), embed(c))
    comet-score:<ckpt>    argmax_c  COMET(src = src block, mt = c)   (reference-free only)

Run both from the same checkpoint and the contrast is "the head versus the
representation it sits on". `comet-score:` refuses a reference-based checkpoint:
at alignment time the reference is the thing being retrieved.

Candidates = the reference block and ITS OWN perturbed variants; the other
blocks' true targets are excluded (telling articles apart is another, easy skill).
Two fixed-size protocols, so the candidate count never grows with k:

  duel        gold vs ONE of its own negatives; chance 0.5; every sampled negative
              is used (coverage-complete), broken down per category and position.
  shortlist   gold vs exactly size-1 of its own negatives; chance 1/size. A block
              short of size-1 negatives is skipped, never padded; the coverage is
              reported per cell and hollowed under 80 % in the plot. Default size
              3 (needing two negatives), because at 5 the k=1 coverage was 20-46 %.

Reads data/pools_<backend>_<lang>_k<k>.json; writes results/comet_score.json
(RunResult[ScoreMetrics]) and results/plots/<output stem>.png.
"""
from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.auth import init_wandb
from part1_block_alignment import DATA_DIR, RESULTS_DIR
from part1_block_alignment.evaluate_encoders import load_pools
from part1_block_alignment.models import Candidate, ModelCells, ScoreMetrics, ScoreRunResult, mean_by_k
from part1_block_alignment.perturb import pool_categories

logger = logging.getLogger(__name__)


# ── candidate sets: fixed size, deterministic ─────────────────────────────────
def sample_negatives(neg: List[int], cands: List[Candidate], n: int,
                     rng: random.Random) -> List[int]:
    """Up to `n` negatives per block, spread over the categories present.

    Taking the first n would over-sample whichever category `build_pool`
    happened to emit first and whichever sentence position it started from, and
    the per-category breakdown would then measure the sampler.
    """
    if len(neg) <= n:
        return list(neg)
    by_cat: Dict[str, List[int]] = defaultdict(list)
    for i in neg:
        by_cat[cands[i].category].append(i)
    for v in by_cat.values():
        rng.shuffle(v)
    picked, cats = [], sorted(by_cat)
    while len(picked) < n and any(by_cat[c] for c in cats):
        for c in cats:
            if by_cat[c] and len(picked) < n:
                picked.append(by_cat[c].pop())
    return sorted(picked)


def shortlist(gold: int, negs: Sequence[int], size: int) -> Optional[List[int]]:
    """gold + exactly (size-1) of the block's OWN single-edit negatives, or None
    when the block cannot supply them — padding it with something easier would
    make that decision cheaper than the others and quietly bias the average."""
    if len(negs) < size - 1:
        return None
    return [gold] + list(negs[:size - 1])


# ── scorers: one interface over "cosine" and "ask the metric" ─────────────────
class Scorer:
    """score(queries, candidates, pairs) -> array of len(pairs), higher = better
    match. `pairs` are (query index, candidate index)."""

    name = "scorer"
    kind = "scorer"

    def score(self, queries: List[str], candidates: List[str],
              pairs: List[Tuple[int, int]], tag: str) -> np.ndarray:
        raise NotImplementedError


class CosineScorer(Scorer):
    """The rule evaluate_encoders already uses: cosine between encoder embeddings."""

    kind = "encoder-cos"

    def __init__(self, ref: str, device: str, batch_size: int, cache_dir: str | None):
        from common.encoders import build_embedder, enc_tag
        self.emb = build_embedder(f"comet:{ref}", device)
        self.name = f"encoder-cos:{enc_tag('comet:' + ref)}"
        self.batch_size, self.cache_dir = batch_size, cache_dir

    def score(self, queries, candidates, pairs, tag):
        from common.encoders import cached_embed
        q = cached_embed(self.emb, queries, f"{tag}_q", self.cache_dir, self.batch_size)
        c = cached_embed(self.emb, candidates, f"{tag}_c", self.cache_dir, self.batch_size)
        qi = np.fromiter((p[0] for p in pairs), dtype=int, count=len(pairs))
        ci = np.fromiter((p[1] for p in pairs), dtype=int, count=len(pairs))
        return np.einsum("ij,ij->i", q[qi], c[ci])   # embeddings are L2-normalised


class CometScorer(Scorer):
    """Ask the metric itself, reference-free."""

    kind = "comet-score"

    def __init__(self, ref: str, batch_size: int, gpus: int):
        from common.comet_models import load_comet, uses_reference
        from common.encoders import enc_tag
        self.model = load_comet(ref)
        if uses_reference(self.model):
            raise SystemExit(
                f"comet-score:{ref} is a REFERENCE-BASED metric. Alignment has no "
                "reference — the reference is the translation being retrieved, so "
                "passing it would hand the model the answer. Use a reference-free "
                "checkpoint (cometkiwi or a QE arm) for comet-score:, and compare "
                "reference-based checkpoints through encoder-cos:.")
        self.name = f"comet-score:{enc_tag('comet:' + ref)}"
        self.batch_size, self.gpus = batch_size, gpus
        self._cache: Dict[Tuple[str, str], float] = {}

    def score(self, queries, candidates, pairs, tag):
        # The same (src, mt) pair turns up in the duel and in the shortlist, and
        # a gold candidate is shared by every protocol — score each pair once.
        want = {(queries[q], candidates[c]) for q, c in pairs} - self._cache.keys()
        todo = sorted(want)
        if todo:
            logger.info(f"    {self.name}: scoring {len(todo):,} new (src, mt) pairs")
            out = self.model.predict([{"src": s, "mt": m} for s, m in todo],
                                     batch_size=self.batch_size, gpus=self.gpus,
                                     progress_bar=True)
            self._cache.update(zip(todo, (float(x) for x in out["scores"])))
        return np.array([self._cache[(queries[q], candidates[c])] for q, c in pairs])


def build_scorer(spec: str, device: str, batch_size: int, gpus: int,
                 cache_dir: str | None) -> Scorer:
    kind, _, ref = spec.partition(":")
    if kind == "encoder-cos":
        return CosineScorer(ref, device, batch_size, cache_dir)
    if kind == "comet-score":
        return CometScorer(ref, batch_size, gpus)
    raise SystemExit(f"unknown scorer {spec!r}; use encoder-cos:<ckpt> or comet-score:<ckpt>")


# ── metrics ───────────────────────────────────────────────────────────────────
def _fmt(v) -> str:
    return " n/a  " if v is None else f"{v:.4f}"


def evaluate(sc: Dict[Tuple[int, int], float], duels: List[Tuple[int, int, int]],
             lists: List[Tuple[int, List[int]]], cands: List[Candidate],
             categories: Sequence[str]) -> ScoreMetrics:
    """duels: (block, gold pool idx, negative pool idx).
       lists: (block, [gold pool idx, other pool idx, ...])."""
    wins, ties = [], 0
    by_cat: Dict[str, List[int]] = defaultdict(list)
    by_pos: Dict[int, List[int]] = defaultdict(list)
    for b, g, n in duels:
        d = sc[(b, g)] - sc[(b, n)]
        w = int(d > 0)
        ties += int(d == 0)
        wins.append(w)
        by_cat[cands[n].category].append(w)
        by_pos[cands[n].position].append(w)

    hits, ranks, beaten = [], [], []
    for b, cols in lists:
        vals = np.array([sc[(b, c)] for c in cols])
        gold = vals[0]
        hits.append(int(gold > vals[1:].max()) if len(vals) > 1 else 1)
        ranks.append(int(1 + (vals[1:] > gold).sum()))
        beaten.append(float((gold > vals[1:]).mean()) if len(vals) > 1 else 1.0)

    return ScoreMetrics(
        n_duels=len(duels),
        duel_accuracy=float(np.mean(wins)) if wins else None,
        duel_ties=ties,
        duel_accuracy_by_category={
            c: (float(np.mean(by_cat[c])) if by_cat.get(c) else None) for c in categories},
        duel_n_by_category={c: len(by_cat.get(c, [])) for c in categories},
        duel_accuracy_by_position={
            str(p): float(np.mean(v)) for p, v in sorted(by_pos.items()) if v},
        n_shortlists=len(lists),
        shortlist_accuracy=float(np.mean(hits)) if hits else None,
        shortlist_mean_rank=float(np.mean(ranks)) if ranks else None,
        frac_negatives_beaten=float(np.mean(beaten)) if beaten else None)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorers", nargs="+",
                    default=["comet-score:Unbabel/wmt22-cometkiwi-da",
                             "encoder-cos:Unbabel/wmt22-cometkiwi-da",
                             "encoder-cos:Unbabel/wmt22-comet-da"],
                    help="encoder-cos:<ckpt> | comet-score:<ckpt> (ref-free only)")
    ap.add_argument("--langs", nargs="+", default=["de", "es", "fr", "ru"])
    ap.add_argument("--k_list", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                    help="k=1 is the single-sentence reference point of the dilution curve.")
    ap.add_argument("--backend", choices=["spacy", "heuristic"], default="spacy",
                    help="Which pools to read (the perturber that built them).")
    ap.add_argument("--negatives_per_block", type=int, default=6,
                    help="Hard negatives kept per block, spread over the categories. "
                         "Caps the number of COMET forward passes.")
    ap.add_argument("--shortlist_size", type=int, default=3,
                    help="Candidates per retrieval decision: the gold plus size-1 of ITS "
                         "OWN negatives, fixed across k. Must be <= negatives_per_block + 1.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42, help="Seed of the negative sampler.")
    ap.add_argument("--emb_cache_dir", default=str(DATA_DIR / "emb_cache"))
    ap.add_argument("--output", default=str(RESULTS_DIR / "comet_score.json"))
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.shortlist_size > args.negatives_per_block + 1:
        raise SystemExit(
            f"--shortlist_size {args.shortlist_size} needs {args.shortlist_size - 1} own "
            f"negatives per block, but --negatives_per_block is {args.negatives_per_block}. "
            f"Every block would be skipped.")

    # ── build every (lang, k) task once; the scorers only differ in the rule ──
    tasks: Dict[str, Dict[int, dict]] = {}
    total_pairs = 0
    for lang, by_k in load_pools(args.backend, args.langs, args.k_list).items():
        tasks[lang] = {}
        for k, pool in by_k.items():
            cands, true_idx, n_blocks = pool.candidates, pool.true_index, pool.n_blocks
            rng = random.Random(f"{args.seed}|{lang}|{k}")
            negs = [sample_negatives(n, cands, args.negatives_per_block, rng)
                    for n in pool.own_negatives()]
            duels = [(b, true_idx[b], n) for b in range(n_blocks) for n in negs[b]]
            lists, short = [], 0
            for b in range(n_blocks):
                cols = shortlist(true_idx[b], negs[b], args.shortlist_size)
                if cols is None:
                    short += 1
                    continue
                lists.append((b, cols))
            pairs = sorted({(b, c) for b, g, n in duels for c in (g, n)} |
                           {(b, c) for b, cols in lists for c in cols})
            used = sorted({c for _b, c in pairs})
            slot = {c: i for i, c in enumerate(used)}
            tasks[lang][k] = {"pool": pool, "duels": duels, "lists": lists, "pairs": pairs,
                              "texts": [cands[c].text for c in used],
                              "compact": [(b, slot[c]) for b, c in pairs],
                              "shortlist_skipped": short}
            total_pairs += len(pairs)
            skip_note = (f" ({short} block(s) short of {args.shortlist_size - 1} own "
                         f"negatives — skipped)" if short else "")
            logger.info(f"  [{lang}] k={k}: {n_blocks} blocks, {len(duels):,} duels, "
                        f"{len(lists):,} shortlists of {args.shortlist_size}{skip_note} -> "
                        f"{len(pairs):,} (src, mt) pairs over {len(used):,} of the pool's "
                        f"{len(cands):,} candidates")

    logger.info(f"\n{total_pairs:,} (src, mt) pairs per scorer (a comet-score scorer runs "
                f"that many forward passes; an encoder-cos scorer only embeds the unique texts).")
    if args.dry_run:
        return

    import torch
    from common.encoders import pick_device
    gpus = args.gpus if args.gpus is not None else (1 if torch.cuda.is_available() else 0)
    device = pick_device(args.device)
    out_path = Path(args.output)
    result = ScoreRunResult(
        experiment="comet_align",
        question="does the COMET SCORE align translations better than the cosine similarity "
                 "of the encoder it is built on, and how does the answer move with block length k",
        timestamp=datetime.now(timezone.utc).isoformat(),
        config=vars(args),
        protocols={"duel": "gold vs one of its own single-error negatives; chance 0.5",
                   "shortlist": f"gold vs {args.shortlist_size - 1} of ITS OWN single-error "
                                f"negatives; chance {1 / args.shortlist_size:.3f}"},
        candidate_set="reference block + its own perturbed variants only — the classic xsim "
                      "distractors (other blocks' true targets) are deliberately excluded")
    wandb_run = init_wandb(args.wandb_project, "comet_score", "evaluate_comet_score",
                           config=vars(args))

    for spec in args.scorers:
        logger.info(f"\n== {spec} ==")
        scorer = build_scorer(spec, device, args.batch_size, gpus, args.emb_cache_dir)
        by_lang: Dict[str, Dict[str, ScoreMetrics]] = {}
        for lang, by_k in tasks.items():
            by_lang[lang] = {}
            for k, T in by_k.items():
                pool = T["pool"]
                tag = f"pool_{args.backend}_{lang}_k{k}"
                vals = scorer.score(pool.queries, T["texts"], T["compact"], tag)
                sc = dict(zip(T["pairs"], (float(v) for v in vals)))
                m = evaluate(sc, T["duels"], T["lists"], pool.candidates, pool_categories(pool))
                m.variant_stats = pool.variant_counts
                # Coverage travels with the number it qualifies: a shortlist accuracy
                # measured on 20 % of the blocks is not the quantity measured on 100 %.
                m.n_blocks = pool.n_blocks
                m.shortlist_skipped = T["shortlist_skipped"]
                m.shortlist_coverage = len(T["lists"]) / pool.n_blocks if pool.n_blocks else None
                by_lang[lang][str(k)] = m
                cov = m.shortlist_coverage
                logger.info(f"  {lang} k={k}: duel={_fmt(m.duel_accuracy)} "
                            f"shortlist={_fmt(m.shortlist_accuracy)} "
                            f"(cov {cov:.0%} of {m.n_blocks:,} blocks) "
                            f"beaten={_fmt(m.frac_negatives_beaten)}"
                            + ("  << low coverage" if cov < 0.8 else ""))
                if wandb_run is not None:
                    pref = f"{scorer.name}/{lang}/k{k}"
                    wandb_run.log({f"{pref}/{key}": v for key, v in
                                   (("duel_accuracy", m.duel_accuracy),
                                    ("shortlist_accuracy", m.shortlist_accuracy),
                                    ("frac_negatives_beaten", m.frac_negatives_beaten))
                                   if v is not None})

        cells = ModelCells[ScoreMetrics](kind=scorer.kind, spec=spec, by_lang=by_lang,
                                         mean_by_k=mean_by_k(by_lang))
        for m in cells.mean_by_k.values():          # coverage is pooled, not a mean of ratios
            m.shortlist_coverage = m.n_shortlists / m.n_blocks if m.n_blocks else None
        result.models[scorer.name] = cells
        result.save(out_path)                       # incremental
        del scorer

    _plot(result, out_path.parent / "plots" / f"{out_path.stem}.png", args.k_list)
    if wandb_run is not None:
        wandb_run.finish()
    logger.info(f"\nResults -> {out_path}")


def _plot(result: ScoreRunResult, out: Path, k_list: List[int]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.parent.mkdir(parents=True, exist_ok=True)

    size = result.config.get("shortlist_size", 5)
    panels = [("duel_accuracy", "duel: gold vs 1 of its own negatives", 0.5),
              ("shortlist_accuracy",
               f"shortlist: gold vs {size - 1} of its own negatives", 1 / size),
              ("frac_negatives_beaten", "fraction of own negatives beaten", None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.8))
    for ax, (key, title, chance) in zip(axes, panels):
        for name, sc in result.models.items():
            mb = sc.mean_by_k
            xs = [k for k in k_list if str(k) in mb and getattr(mb[str(k)], key) is not None]
            ys = [getattr(mb[str(k)], key) for k in xs]
            if not xs:
                continue
            line, = ax.plot(xs, ys, marker="o", ls="-" if sc.kind == "comet-score" else "--",
                            label=name)
            if key.startswith("shortlist"):
                # Hollow out the points measured on a biased subset of blocks:
                # at small k many blocks cannot supply size-1 negatives, and the
                # ones that can are the perturbation-rich ones.
                low = [(k, y) for k, y in zip(xs, ys)
                       if (mb[str(k)].shortlist_coverage or 1.0) < 0.8]
                if low:
                    ax.plot([k for k, _ in low], [y for _, y in low], "o",
                            mfc="white", mec=line.get_color(), mew=1.4, ms=7,
                            ls="none", zorder=5)
        if chance is not None:
            ax.axhline(chance, color="grey", ls=":", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("block length k (sentences)")
        ax.set_xticks(k_list)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=7)
    axes[1].text(0.02, 0.02, "hollow = under 80 % of blocks could supply a full\n"
                             "shortlist at that k; not comparable to solid points",
                 transform=axes[1].transAxes, fontsize=6.5, color="grey", va="bottom")
    fig.suptitle("Aligning with the COMET score (solid) vs with encoder cosine "
                 "(dashed)\ncandidates = the reference block and its own "
                 "perturbed variants, no other-block distractors", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    logger.info(f"  [plot] {out}")
    return out


if __name__ == "__main__":
    main()
