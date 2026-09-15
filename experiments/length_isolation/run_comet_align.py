#!/usr/bin/env python3
"""
run_comet_align.py — align translations with the COMET SCORE instead of with
encoder similarity, on the same blocks `run_xsim.py` uses.

The block-xSIM++ experiment asks whether COMET's *encoder* still separates a
correct translation from one with a single perturbed sentence, by cosine
similarity in embedding space. But the encoder is not what anybody reports:
the metric is the encoder plus a regression head trained on human judgements.
This script swaps the decision rule and leaves everything else — the same
FLORES+ articles, the same non-overlapping blocks, the same xSIM++ hard
negatives — exactly where it was:

    encoder-cos:<ckpt>    pick argmax_c  cos(embed(src block), embed(c))
    comet-score:<ckpt>    pick argmax_c  COMET(src = src block, mt = c)

Running both from the SAME checkpoint makes the contrast "the head versus the
representation it sits on". Run them from a base metric and from a length-trained
arm and the contrast becomes "what did long-text training move — the representation,
the head, or both".

Reference-free only
------------------------------------------
The metric-score rule needs to score a candidate WITHOUT a reference: at
alignment time the reference is the thing being retrieved :)

Candidate set: the reference block and its own perturbed variants
----------------------------------------------------------------
The candidates for a query are that block's gold translation plus its own
single-edit negatives. The *other* blocks' true targets — the classic xsim
distractors — are deliberately NOT in the set: telling one article from another
is a different skill, an easy one, and including it lets a model look sensitive
to the injected edit when it is only recognising the topic. Dropped, there is
nothing left to separate but the perturbation, which is the quantity of
interest. (`blocks.evaluate_blocks` reports the same ablation for the cosine
rule, as `pool_ablation`.)

The number of candidates must also be a parameter, never a by-product of k: a
k=5 block admits more possible perturbations than a k=2 block, so a pool that
grew with k would confound length with task difficulty. Both protocols hold the
candidate count fixed across k.

  duel        gold vs ONE of its own hard negatives (one perturbed sentence).
              Accuracy = P[score(gold) > score(negative)], chance 0.5. Every
              sampled negative is used, and the breakdown is reported per
              perturbation category and per position of the perturbed sentence
              inside the block — the dilution question, per phenomenon.
  shortlist   gold vs exactly P-1 of its OWN negatives. Accuracy =
              P[argmax = gold], chance 1/P. A block that cannot supply P-1
              negatives is skipped rather than padded with something easier, so
              every scored decision is equally hard.

Coverage: read the duel curve first
-----------------------------------
The duel is coverage-complete — every block with at least one negative takes
part — so its curve against k is unbiased. The shortlist is not: a single
sentence often admits fewer perturbations than a five-sentence block, so at
k=1 a large share of blocks cannot supply P-1 negatives and are skipped, and
the survivors are the perturbation-rich ones (longer, with numbers and named
entities). Measured at P=5 on FLORES+ dev+devtest, coverage ran 20-46 % at k=1
against 53-100 % at k=2 — enough to move a curve on its own.

So: `shortlist_coverage` is reported per (scorer, language, k) and pooled in
`_mean_by_k`, the plot hollows out any point below 80 %, and the default P is
3 rather than 5 because needing two negatives instead of four is far more often
satisfiable at k=1. A shortlist point drawn hollow is not comparable to a solid
one; the duel panel beside it is.

Scores from a cosine and scores from a regression head are not on one scale, so
margins are not comparable across the two rules. `frac_negatives_beaten` and the
gold's rank in the shortlist are, and both are reported.

Usage
-----
  python experiments/length_isolation/run_comet_align.py \
      --scorers "comet-score:Unbabel/wmt22-cometkiwi-da" \
                "encoder-cos:Unbabel/wmt22-cometkiwi-da" \
                "encoder-cos:Unbabel/wmt22-comet-da" \
                "comet-score:$HOME/scratch/checkpoints/retrain-wmt-v3/kiwi-mix-frac000nat/*/*/checkpoints/last.ckpt" \
      --langs de es fr ru --k_list 1 2 3 4 5 \
      --output results/comet_align/comet_align.json

  # what would be scored, without touching a GPU
  python experiments/length_isolation/run_comet_align.py --dry_run --langs de
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from blocks import (CATEGORIES, Candidate, Perturber, block_text,
                    build_blocks, build_pool)
from nlp_perturb import build_perturber
from fdtem.comet_io import load_comet, uses_reference
from fdtem.encoders import build_embedder, cached_embed, enc_tag, pick_device
from fdtem.flores import load_flores_source

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ════════════════════════════════════════════════════════════════════════════
# Candidate sets — fixed size, deterministic
# ════════════════════════════════════════════════════════════════════════════
def own_negatives(cands: List[Candidate], n_blocks: int) -> List[List[int]]:
    """Pool positions of each block's own perturbed candidates."""
    out: List[List[int]] = [[] for _ in range(n_blocks)]
    for i, c in enumerate(cands):
        if c.kind == "perturbed":
            out[c.block_id].append(i)
    return out


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
    """gold + exactly (size-1) of the block's OWN single-edit negatives.

    No other-block distractors: separating articles is a different and much
    easier skill, and mixing it in inflates the apparent sensitivity to the
    injected edit. Returns None when the block cannot supply size-1 negatives —
    padding it with something easier would make that decision cheaper than the
    others and quietly bias the average.
    """
    if len(negs) < size - 1:
        return None
    return [gold] + list(negs[:size - 1])


# ════════════════════════════════════════════════════════════════════════════
# Scorers — one interface over "cosine" and "ask the metric"
# ════════════════════════════════════════════════════════════════════════════
class Scorer:
    """score(queries, candidates, pairs) -> array of len(pairs), higher = better
    match. `pairs` are (query index, candidate index)."""

    name = "scorer"
    kind = "scorer"

    def score(self, queries: List[str], candidates: List[str],
              pairs: List[Tuple[int, int]], tag: str) -> np.ndarray:
        raise NotImplementedError


class CosineScorer(Scorer):
    """The rule block-xSIM++ already uses: cosine between encoder embeddings."""

    kind = "encoder-cos"

    def __init__(self, ref: str, device: str, batch_size: int, cache_dir: str | None):
        self.emb = build_embedder(f"comet:{ref}", device)
        self.name = f"encoder-cos:{enc_tag('comet:' + ref)}"
        self.batch_size, self.cache_dir = batch_size, cache_dir

    def score(self, queries, candidates, pairs, tag):
        q = cached_embed(self.emb, queries, f"{tag}_q", self.cache_dir, self.batch_size)
        c = cached_embed(self.emb, candidates, f"{tag}_c", self.cache_dir, self.batch_size)
        qi = np.fromiter((p[0] for p in pairs), dtype=int, count=len(pairs))
        ci = np.fromiter((p[1] for p in pairs), dtype=int, count=len(pairs))
        return np.einsum("ij,ij->i", q[qi], c[ci])   # embeddings are L2-normalised


class CometScorer(Scorer):
    """The rule nobody has tried here: ask the metric itself, reference-free."""

    kind = "comet-score"

    def __init__(self, ref: str, batch_size: int, gpus: int):
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
    raise SystemExit(f"unknown scorer {spec!r}; use encoder-cos:<ckpt> or "
                     f"comet-score:<ckpt>")


# ════════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════════
def _fmt(v) -> str:
    return " n/a  " if v is None else f"{v:.4f}"


def _nanmean(vals):
    """Mean over the cells that produced a number. A (lang, k) cell can be
    empty — k=1 blocks admit the fewest perturbations, so some cannot supply a
    full shortlist and are skipped — and a plain np.mean would raise on the
    None rather than say so."""
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None
def evaluate(sc: Dict[Tuple[int, int], float], duels: List[Tuple[int, int, int]],
             lists: List[Tuple[int, List[int]]], cands: List[Candidate],
             categories: Sequence[str]) -> dict:
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

    return {
        "n_duels": len(duels),
        "duel_accuracy": float(np.mean(wins)) if wins else None,
        "duel_ties": ties,
        "duel_accuracy_by_category": {
            c: (float(np.mean(by_cat[c])) if by_cat.get(c) else None)
            for c in categories},
        "duel_n_by_category": {c: len(by_cat.get(c, [])) for c in categories},
        "duel_accuracy_by_position": {
            str(p): float(np.mean(v)) for p, v in sorted(by_pos.items()) if v},
        "n_shortlists": len(lists),
        "shortlist_accuracy": float(np.mean(hits)) if hits else None,
        "shortlist_mean_rank": float(np.mean(ranks)) if ranks else None,
        "frac_negatives_beaten": float(np.mean(beaten)) if beaten else None,
    }


# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorers", nargs="+",
                    default=["comet-score:Unbabel/wmt22-cometkiwi-da",
                             "encoder-cos:Unbabel/wmt22-cometkiwi-da",
                             "encoder-cos:Unbabel/wmt22-comet-da"],
                    help="encoder-cos:<ckpt> | comet-score:<ckpt> (ref-free only)")
    ap.add_argument("--langs", nargs="+", default=["de", "es", "fr", "ru"],
                    help="Non-pivot languages. Cased scripts only — the entity "
                         "perturbation needs letter case.")
    ap.add_argument("--pivot", default="en")
    ap.add_argument("--k_list", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                    help="k=1 is the single-sentence reference point: the "
                         "dilution curve is only readable against it.")
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES),
                    choices=list(CATEGORIES))
    ap.add_argument("--variants_per_position", type=int, default=2)
    ap.add_argument("--negatives_per_block", type=int, default=6,
                    help="Hard negatives kept per block, spread over the "
                         "categories. Caps the number of COMET forward passes, "
                         "which is what makes this affordable at all.")
    ap.add_argument("--shortlist_size", type=int, default=3,
                    help="Candidates per retrieval decision — the gold plus "
                         "size-1 of ITS OWN negatives, FIXED across k. Must be "
                         "<= negatives_per_block + 1, and small enough that k=1 "
                         "blocks (which admit the fewest perturbations) can "
                         "still supply it: at 5 the k=1 coverage fell to 20-46 %, "
                         "which biases the curve towards perturbation-rich "
                         "sentences. 3 needs only two negatives.")
    ap.add_argument("--flores_source", choices=["plus", "raw"], default="plus")
    ap.add_argument("--splits", nargs="+", default=["dev", "devtest"])
    ap.add_argument("--max_sents", type=int, default=None)
    ap.add_argument("--max_blocks", type=int, default=None,
                    help="Truncate the block list per k and split (smoke tests).")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perturb_backend", choices=["heuristic", "spacy", "auto"],
                    default="spacy",
                    help="Negative generator — see run_xsim.py. `spacy` needs "
                         "`python -m spacy download <lang>_core_*_sm`.")
    ap.add_argument("--wordnet_langs", nargs="*", default=["en"],
                    help="Languages allowed to draw antonyms from WordNet "
                         "(spacy backend only).")
    ap.add_argument("--emb_cache_dir", default="results/comet_align/emb_cache")
    ap.add_argument("--output", default="results/comet_align/comet_align.json")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.shortlist_size > args.negatives_per_block + 1:
        raise SystemExit(
            f"--shortlist_size {args.shortlist_size} needs "
            f"{args.shortlist_size - 1} own negatives per block, but "
            f"--negatives_per_block is {args.negatives_per_block}. Every block "
            f"would be skipped.")

    import torch
    gpus = args.gpus if args.gpus is not None else (1 if torch.cuda.is_available() else 0)
    langs = [args.pivot] + [l for l in args.langs if l != args.pivot]
    loaded = [(sp, load_flores_source(args.flores_source, langs, sp, args.max_sents))
              for sp in args.splits]

    # ── build every (lang, k) task once; the scorers only differ in the rule ──
    tasks: Dict[str, Dict[int, dict]] = {}
    total_pairs = 0
    for lang in args.langs:
        if lang == args.pivot:
            continue
        corpus = [s for _sp, d in loaded for s in d.sentences[lang]]
        pert = build_perturber(corpus, lang, args.seed,
                               args.perturb_backend, args.wordnet_langs)
        tasks[lang] = {}
        for k in args.k_list:
            per_split = [(sp, d, build_blocks(d, k, split=sp)) for sp, d in loaded]
            if args.max_blocks:
                per_split = [(sp, d, b[:args.max_blocks]) for sp, d, b in per_split]
            n_blocks = sum(len(b) for _, _, b in per_split)
            if n_blocks < 2:
                logger.warning(f"  [{lang}] k={k}: {n_blocks} blocks — skipped")
                continue
            queries = [block_text(d, blk, args.pivot)
                       for _sp, d, bs in per_split for blk in bs]
            cands, true_idx, stats = build_pool(per_split, lang, args.categories,
                                                args.variants_per_position, pert)
            rng = random.Random(f"{args.seed}|{lang}|{k}")
            negs = own_negatives(cands, n_blocks)
            negs = [sample_negatives(n, cands, args.negatives_per_block, rng)
                    for n in negs]
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
            tasks[lang][k] = {"queries": queries, "cands": cands,
                              "true_idx": true_idx, "duels": duels,
                              "lists": lists, "pairs": pairs,
                              "texts": [cands[c].text for c in used],
                              "compact": [(b, slot[c]) for b, c in pairs],
                              "n_blocks": n_blocks, "n_shortlists": len(lists),
                              "shortlist_skipped": short, "stats": stats}
            total_pairs += len(pairs)
            skip_note = (f" ({short} block(s) short of "
                         f"{args.shortlist_size - 1} own negatives — skipped)"
                         if short else "")
            logger.info(f"  [{lang}] k={k}: {n_blocks} blocks, "
                        f"{len(duels):,} duels, {len(lists):,} shortlists of "
                        f"{args.shortlist_size}{skip_note} → {len(pairs):,} "
                        f"(src, mt) pairs over {len(used):,} of the pool's "
                        f"{len(cands):,} candidates")

    logger.info(f"\n{total_pairs:,} (src, mt) pairs per scorer "
                f"(a comet-score scorer runs that many forward passes; an "
                f"encoder-cos scorer only embeds the unique texts).")
    if args.dry_run:
        return

    device = pick_device(args.device)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "comet_align",
        "question": "does the COMET SCORE align translations better than the "
                    "cosine similarity of the encoder it is built on, and how "
                    "does the answer move with block length k",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {k: v for k, v in vars(args).items()},
        "protocols": {
            "duel": "gold vs one of its own single-error negatives; chance 0.5",
            "shortlist": f"gold vs {args.shortlist_size - 1} of ITS OWN "
                         f"single-error negatives; chance "
                         f"{1 / args.shortlist_size:.3f}",
        },
        "candidate_set": "reference block + its own perturbed variants only — "
                         "the classic xsim distractors (other blocks' true "
                         "targets) are deliberately excluded",
        "scorers": {},
    }

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, job_type="comet_align",
                                   name="comet_align", config=vars(args))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"W&B init failed: {exc}")

    for spec in args.scorers:
        logger.info(f"\n══ {spec} ══")
        scorer = build_scorer(spec, device, args.batch_size, gpus, args.emb_cache_dir)
        per_lang: Dict[str, dict] = {}
        for lang, by_k in tasks.items():
            per_lang[lang] = {}
            for k, T in by_k.items():
                tag = f"align_{lang}_k{k}_v{args.variants_per_position}"
                vals = scorer.score(T["queries"], T["texts"], T["compact"], tag)
                sc = dict(zip(T["pairs"], (float(v) for v in vals)))
                m = evaluate(sc, T["duels"], T["lists"], T["cands"], args.categories)
                m["variant_stats"] = T["stats"]
                # Coverage travels with the number it qualifies: a shortlist
                # accuracy measured on 20 % of the blocks is not the same
                # quantity as one measured on 100 %, and the difference is
                # systematic (the survivors are the perturbation-rich blocks).
                m["n_blocks"] = T["n_blocks"]
                m["shortlist_skipped"] = T["shortlist_skipped"]
                m["shortlist_coverage"] = (T["n_shortlists"] / T["n_blocks"]
                                           if T["n_blocks"] else None)
                per_lang[lang][str(k)] = m
                cov = m["shortlist_coverage"]
                logger.info(f"  {lang} k={k}: duel={_fmt(m['duel_accuracy'])} "
                            f"shortlist={_fmt(m['shortlist_accuracy'])} "
                            f"(cov {cov:.0%} of {m['n_blocks']:,} blocks) "
                            f"beaten={_fmt(m['frac_negatives_beaten'])}"
                            + ("  << low coverage" if cov < 0.8 else ""))
                if wandb_run is not None:
                    pref = f"{scorer.name}/{lang}/k{k}"
                    wandb_run.log({f"{pref}/{key}": m[key] for key in
                                   ("duel_accuracy", "shortlist_accuracy",
                                    "frac_negatives_beaten")
                                   if m[key] is not None})

        mean_by_k = {}
        for k in args.k_list:
            cells = [per_lang[l][str(k)] for l in per_lang if str(k) in per_lang[l]]
            if not cells:
                continue
            mean_by_k[str(k)] = {
                key: _nanmean([c[key] for c in cells])
                for key in ("duel_accuracy", "shortlist_accuracy",
                            "frac_negatives_beaten", "shortlist_mean_rank")}
            n_blk = int(sum(c["n_blocks"] for c in cells))
            n_sl = int(sum(c["n_shortlists"] for c in cells))
            mean_by_k[str(k)].update({
                "n_duels": int(sum(c["n_duels"] for c in cells)),
                "n_blocks": n_blk, "n_shortlists": n_sl,
                "shortlist_coverage": (n_sl / n_blk) if n_blk else None,
                "duel_accuracy_by_category": {
                    cat: _nanmean([c["duel_accuracy_by_category"][cat] for c in cells])
                    for cat in args.categories},
            })
        per_lang["_mean_by_k"] = mean_by_k
        results["scorers"][scorer.name] = {"kind": scorer.kind, "spec": spec,
                                           **per_lang}
        json.dump(results, open(out_path, "w"), indent=2)   # incremental
        del scorer

    _plot(results, out_path.parent / "plots", args.k_list)
    if wandb_run is not None:
        wandb_run.finish()
    logger.info(f"\nResults → {out_path}")


def _plot(results: dict, plot_dir: Path, k_list: List[int]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir.mkdir(parents=True, exist_ok=True)

    size = results["config"].get("shortlist_size", 5)
    panels = [("duel_accuracy", "duel: gold vs 1 of its own negatives", 0.5),
              ("shortlist_accuracy",
               f"shortlist: gold vs {size - 1} of its own negatives", 1 / size),
              ("frac_negatives_beaten", "fraction of own negatives beaten", None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.8))
    for ax, (key, title, chance) in zip(axes, panels):
        for name, sc in results["scorers"].items():
            mb = sc.get("_mean_by_k", {})
            xs = [k for k in k_list if mb.get(str(k), {}).get(key) is not None]
            ys = [mb[str(k)][key] for k in xs]
            if not xs:
                continue
            line, = ax.plot(xs, ys, marker="o",
                            ls="-" if sc["kind"] == "comet-score" else "--",
                            label=name)
            if key.startswith("shortlist"):
                # Hollow out the points measured on a biased subset of blocks:
                # at small k many blocks cannot supply size-1 negatives, and the
                # ones that can are the perturbation-rich ones.
                low = [(k, y) for k, y in zip(xs, ys)
                       if (mb[str(k)].get("shortlist_coverage") or 1.0) < 0.8]
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
                 transform=axes[1].transAxes, fontsize=6.5, color="grey",
                 va="bottom")
    fig.suptitle("Aligning with the COMET score (solid) vs with encoder cosine "
                 "(dashed)\ncandidates = the reference block and its own "
                 "perturbed variants, no other-block distractors", fontsize=11)
    out = plot_dir / "comet_align.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    logger.info(f"  [plot] {out}")
    return out


if __name__ == "__main__":
    main()
