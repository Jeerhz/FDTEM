"""Find the reference block among D distractors perturbed in EVERY sentence
(perturb_every_sentence.py), for every D in one pass.

    python -m part1_block_alignment.evaluate_every_sentence \
        --scorers comet-score:Unbabel/wmt22-cometkiwi-da comet:Unbabel/wmt22-comet-da labse e5 \
        --langs de es fr ru --k_list 1 2 3 4 5 --D_list 1 2 3 4 5 6

A scorer is `comet-score:<ckpt>`, COMET(src = English block, mt = candidate) for a
reference-free checkpoint, or any encoder spec of common.encoders (`comet:<ckpt>`,
`hf-mean:<id>`, `labse`, `e5`), scored by the cosine to the English block.

Two conditions on the same windows, distractors and k:

    every_sentence   every sentence of a distractor is perturbed: one error per sentence
    first_sentence   only the first one is, the other k-1 are the reference's: the
                     single-edit protocol, whose error share falls as 1/k (the control)

They coincide at k = 1. With the error share held, what is left of the fall with k is
the model's own; the gap between the conditions is the dilution.

A window with m distractors enters the size-D decision when m >= D and is averaged
over all C(m, D) subsets in closed form: P[win] = C(w, D) / C(m, D), rank
1 + D (m - w) / m, w = distractors scored strictly below the reference (a tie counts
against it). Chance is 1 / (D + 1). Intervals are 95 % percentile bootstraps over
articles, jointly across languages for the language mean. Writes
results/every_sentence.json (EverySentenceRunResult) and results/plots/<stem>.png.
"""
from __future__ import annotations

import argparse
import gc
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from math import comb
from pathlib import Path

import numpy as np

from common.auth import init_wandb
from part1_block_alignment import DATA_DIR, RESULTS_DIR
from part1_block_alignment.evaluate_comet_score import CometScorer, CosineScorer
from part1_block_alignment.models import (ConditionCells, DecisionMetrics, EverySentenceCells,
                                          EverySentenceMetrics, EverySentenceRunResult,
                                          PerturbedBlock, mean_by_k)
from part1_block_alignment.perturb_every_sentence import load_rows

logger = logging.getLogger(__name__)

CONDITIONS = ("every_sentence", "first_sentence")


def first_sentence_only(block: PerturbedBlock, first: PerturbedBlock) -> list[str]:
    """`block`'s distractors with only their first sentence perturbed; `first` is its k = 1 row."""
    rest = block.reference[len(first.reference):]
    return [d + rest for d in first.distractors]


class EncoderCosine(CosineScorer):
    """CosineScorer over any encoder of the zoo, not only COMET checkpoints."""

    def __init__(self, spec: str, device: str, batch_size: int, cache_dir: str | None):
        from common.encoders import build_embedder
        self.emb = build_embedder(spec, device)
        self.name = f"encoder-cos:{self.emb.name}"
        self.batch_size, self.cache_dir = batch_size, cache_dir


def build_scorer(spec: str, device: str, batch_size: int, gpus: int, cache_dir: str | None):
    if spec.startswith("comet-score:"):
        return CometScorer(spec.partition(":")[2], batch_size, gpus)
    return EncoderCosine(spec, device, batch_size, cache_dir)


def wins(scores: np.ndarray, sizes: Sequence[int]) -> tuple[np.ndarray, np.ndarray, int]:
    """(w, m, ties): `scores` holds, window after window, the reference then its m distractors."""
    w, s, ties = [], 0, 0
    for m in sizes:
        gold, neg = scores[s], scores[s + 1:s + 1 + m]
        w.append(int((neg < gold).sum()))
        ties += int((neg == gold).sum())
        s += 1 + m
    return np.array(w), np.array(sizes), ties


def success(w: np.ndarray, m: np.ndarray, D: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(eligible windows, P[win], rank) of the size-D decision."""
    ok = m >= D
    p = np.array([comb(int(a), D) / comb(int(b), D) for a, b in zip(w[ok], m[ok])])
    return ok, p, 1 + D * (m[ok] - w[ok]) / m[ok]


def bootstrap_ci(groups: Sequence[tuple[np.ndarray, np.ndarray]], n_boot: int = 1000,
                 seed: int = 0) -> list[float] | None:
    """95 % interval of the mean over groups (languages) of each group's mean, resampling
    the articles of all groups jointly. A group is (article of each value, values)."""
    groups = [(a, v) for a, v in groups if len(v)]
    if not groups:
        return None
    articles = np.unique(np.concatenate([a for a, _v in groups]))
    counts = np.random.default_rng(seed).multinomial(
        len(articles), np.full(len(articles), 1 / len(articles)), size=n_boot)
    means = []
    for a, v in groups:
        c = counts[:, np.searchsorted(articles, a)]
        n = c.sum(1)
        means.append(np.where(n > 0, c @ v / np.maximum(n, 1), np.nan))
    lo, hi = np.nanpercentile(np.nanmean(np.vstack(means), 0), [2.5, 97.5])
    return [float(lo), float(hi)]


def cell_metrics(w: np.ndarray, m: np.ndarray, ties: int, articles: np.ndarray,
                 tokens: float, D_list: Sequence[int],
                 n_boot: int = 1000) -> tuple[EverySentenceMetrics, dict]:
    """The cell, plus D -> (articles, P[win]) of its eligible windows for the language mean."""
    by_D, per_window = {}, {}
    for D in D_list:
        ok, p, rank = success(w, m, D)
        per_window[D] = (articles[ok], p)
        by_D[str(D)] = DecisionMetrics(
            n_used=int(ok.sum()), coverage=float(ok.mean()),
            success=float(p.mean()) if len(p) else None,
            success_ci=bootstrap_ci([per_window[D]], n_boot),
            mean_rank=float(rank.mean()) if len(rank) else None)
    return EverySentenceMetrics(n_windows=len(w), pairwise_win=float(np.mean(w / m)), ties=ties,
                                mean_source_tokens=tokens, by_D=by_D), per_window


def _fmt(v) -> str:
    return " n/a " if v is None else f"{v:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorers", nargs="+",
                    default=["comet-score:Unbabel/wmt22-cometkiwi-da", "comet:Unbabel/wmt22-comet-da",
                             "hf-mean:xlm-roberta-large", "labse", "e5"],
                    help="comet-score:<ckpt> (reference-free) | an encoder spec, scored by cosine")
    ap.add_argument("--langs", nargs="+", default=["de", "es", "fr", "ru"])
    ap.add_argument("--k_list", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--D_list", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--backend", choices=["spacy", "heuristic"], default="spacy",
                    help="Which dataset to read (the perturber that built it).")
    ap.add_argument("--max_windows", type=int, default=None, help="Per language (smoke tests).")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--emb_cache_dir", default=str(DATA_DIR / "emb_cache"))
    ap.add_argument("--output", default=str(RESULTS_DIR / "every_sentence.json"))
    ap.add_argument("--wandb_project", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if 1 not in args.k_list:
        raise SystemExit("--k_list must contain 1: the first_sentence control is built from the k = 1 rows")

    # (lang, k) -> the rows of that block length, one per window, in window order
    blocks: dict[str, dict[int, list[PerturbedBlock]]] = {}
    for lang in args.langs:
        rows = load_rows(args.backend, lang)
        keep = sorted({r.window for r in rows})[:args.max_windows]
        blocks[lang] = {k: sorted((r for r in rows if r.k == k and r.window in keep),
                                  key=lambda r: r.window) for k in args.k_list}
        n_pairs = sum(len(r.distractors) + 1 for k in args.k_list for r in blocks[lang][k])
        logger.info(f"  [{lang}] {len(keep)} windows, {n_pairs:,} (source, candidate) pairs "
                    f"per condition")

    import torch
    from common.encoders import pick_device
    gpus = args.gpus if args.gpus is not None else int(torch.cuda.is_available())
    device = pick_device(args.device)
    out_path = Path(args.output)
    result = EverySentenceRunResult(
        experiment="every_sentence",
        question="with the error share held constant (one perturbation per sentence), does "
                 "finding the reference among D distractors still get harder as the block grows",
        timestamp=datetime.now(timezone.utc).isoformat(), config=vars(args),
        protocols={"every_sentence": "every sentence of each distractor perturbed",
                   "first_sentence": "only the first sentence perturbed (single-edit control)"},
        candidate_set="the reference block and D of its own distractors; chance 1/(D+1)")
    wandb_run = init_wandb(args.wandb_project, out_path.stem, "evaluate_every_sentence",
                           config=vars(args))

    for spec in args.scorers:
        logger.info(f"\n== {spec} ==")
        scorer = build_scorer(spec, device, args.batch_size, gpus, args.emb_cache_dir)
        cells = EverySentenceCells(kind=scorer.kind, spec=spec)
        for cond in CONDITIONS:
            by_lang: dict[str, dict[str, EverySentenceMetrics]] = {}
            per_window: dict[tuple[str, str, int], tuple] = {}
            for lang, by_k in blocks.items():
                by_lang[lang] = {}
                firsts = {r.window: r for r in by_k[1]}
                for k, rows in by_k.items():
                    texts, pairs, sizes = [], [], []
                    for i, r in enumerate(rows):
                        cands = (r.distractors if cond == "every_sentence"
                                 else first_sentence_only(r, firsts[r.window]))
                        pairs += [(i, len(texts) + c) for c in range(len(cands) + 1)]
                        texts += [r.reference] + cands
                        sizes.append(len(cands))
                    scores = scorer.score([r.source for r in rows], texts, pairs,
                                          f"every_{args.backend}_{cond}_{lang}_k{k}")
                    w, m, ties = wins(np.asarray(scores), sizes)
                    cell, pw = cell_metrics(w, m, ties, np.array([r.url for r in rows]),
                                            float(np.mean([r.source_n_tokens for r in rows])),
                                            args.D_list, args.n_boot)
                    by_lang[lang][str(k)] = cell
                    per_window.update({(str(k), str(D), lang): v for D, v in pw.items()})
                    logger.info(f"  {cond:14s} {lang} k={k}: pairwise={_fmt(cell.pairwise_win)} "
                                + " ".join(f"A({D})={_fmt(cell.by_D[str(D)].success)}"
                                           for D in args.D_list))
                    if wandb_run is not None:
                        wandb_run.log({f"{scorer.name}/{cond}/{lang}/k{k}/success_D{D}": v.success
                                       for D, v in ((D, cell.by_D[str(D)]) for D in args.D_list)
                                       if v.success is not None})
            cc = ConditionCells(by_lang=by_lang, mean_by_k=mean_by_k(by_lang))
            for k, mc in cc.mean_by_k.items():           # coverage pooled, interval joint
                for D, dm in mc.by_D.items():
                    dm.coverage = dm.n_used / mc.n_windows if mc.n_windows else None
                    dm.success_ci = bootstrap_ci([per_window[(k, D, lang)] for lang in by_lang],
                                                 args.n_boot)
            cells.conditions[cond] = cc
        result.models[scorer.name] = cells
        result.save(out_path)                            # incremental
        del scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _plot(result, out_path.parent / "plots" / f"{out_path.stem}.png", args.D_list)
    if wandb_run is not None:
        wandb_run.finish()
    logger.info(f"\nResults -> {out_path}")


def _plot(result: EverySentenceRunResult, out: Path, D_list: Sequence[int]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.parent.mkdir(parents=True, exist_ok=True)
    shown = [D for D in (1, 2, 4, 6) if D in D_list] or list(D_list)[:4]
    fig, axes = plt.subplots(1, len(shown), figsize=(4.4 * len(shown), 3.9), squeeze=False,
                             sharey=True)
    for ax, D in zip(axes[0], shown):
        for i, (name, cells) in enumerate(result.models.items()):
            for cond, ls in zip(CONDITIONS, ("-", "--")):
                mb = cells.conditions[cond].mean_by_k
                ks = sorted(mb, key=int)
                ys = [np.nan if mb[k].by_D[str(D)].success is None else mb[k].by_D[str(D)].success
                      for k in ks]
                ax.plot([int(k) for k in ks], ys, ls, color=f"C{i}", marker="o", ms=3.5,
                        label=name if cond == CONDITIONS[0] else None)
                if cond == CONDITIONS[0]:
                    ci = [mb[k].by_D[str(D)].success_ci or [np.nan, np.nan] for k in ks]
                    ax.fill_between([int(k) for k in ks], [c[0] for c in ci], [c[1] for c in ci],
                                    color=f"C{i}", alpha=0.12, lw=0)
        ax.axhline(1 / (D + 1), color="grey", ls=":", lw=1)
        ax.set_xticks([int(k) for k in ks])
        ax.set_title(f"reference vs D = {D} distractors (chance {1 / (D + 1):.2f})", fontsize=9)
        ax.set_xlabel("block length k (sentences)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("success rate A(D)")
    axes[0][-1].legend(fontsize=6.5)
    fig.suptitle("Every sentence perturbed (solid, 95 % CI) vs first sentence only (dashed); "
                 "mean over languages", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    logger.info(f"  [plot] {out}")
    return out


if __name__ == "__main__":
    main()
