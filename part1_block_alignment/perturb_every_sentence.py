"""FLORES+ blocks whose distractors carry one perturbation in EVERY sentence.

    python -m part1_block_alignment.perturb_every_sentence --langs de es fr ru --k_list 1 2 3 4 5
    python -m part1_block_alignment.perturb_every_sentence --langs de --dry_run   # coverage, CPU

perturb.py puts one error in a block, so the perturbed share of the text falls as 1/k and
the dilution of the error is confounded with the model's own handling of length. Here
every sentence of a distractor is perturbed: one error per sentence at every k.

Windows are non-overlapping runs of max(k_list) sentences of one article (build_blocks);
the k-block of a window is its first k sentences, so every k is measured on the same
windows. Distractor d perturbs sentence j with category CATEGORIES[(d + j) % 3], so
neighbouring sentences carry different types; when the sentence does not admit it, the
next category in that order, and never a perturbation that an earlier distractor already
took at that sentence. Nothing depends on k: the distractor at k+1 is the distractor at k
plus one perturbed sentence, so a longer block carries exactly the errors of the shorter.

A window keeps its first m <= --n_distractors distractors, those whose first sentence
differs from every earlier one (they then differ at every k); a window with a sentence
that admits no perturbation at all is dropped. Writes
data/every_sentence_<backend>_<lang>.jsonl, one PerturbedBlock per (window, k), with the
token count of the English source block (XLM-R tokenizer, the one COMET reads with).
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from common.flores import FloresCorpus, joiner
from part1_block_alignment import DATA_DIR
from part1_block_alignment.build_blocks import build_blocks
from part1_block_alignment.load_flores import load_corpus
from part1_block_alignment.models import CATEGORIES, Block, PerturbedBlock
from part1_block_alignment.perturb import backend_name, build_perturber

logger = logging.getLogger(__name__)


def perturb_window(sentences: Sequence[str], perturber, n: int) -> list[list[tuple[str, str]]]:
    """Up to n distractors of one window, each a (category, perturbed sentence) per sentence.

    Empty when a sentence admits no perturbation at all."""
    options = [{c: perturber.variants(s, c, n) for c in CATEGORIES} for s in sentences]
    if not all(any(o.values()) for o in options):
        return []
    distractors: list[list[tuple[str, str]]] = [[] for _ in range(n)]
    for j, opts in enumerate(options):
        taken: set[str] = set()
        for d in range(n):
            order = [CATEGORIES[(d + j + t) % len(CATEGORIES)] for t in range(len(CATEGORIES))]
            fresh = [(c, v) for c in order for v in opts[c] if v not in taken]
            # nothing fresh left: share an edit with another distractor (still one error here)
            cat, text = fresh[0] if fresh else next((c, opts[c][0]) for c in order if opts[c])
            taken.add(text)
            distractors[d].append((cat, text))
    firsts = [d[0][1] for d in distractors]
    return [d for i, d in enumerate(distractors) if firsts[i] not in firsts[:i]]


def build_rows(corpus: FloresCorpus, windows: Sequence[Block], first_window: int, lang: str,
               pivot: str, k_list: Sequence[int], perturber, n: int,
               n_tokens: Callable[[str], int]) -> list[PerturbedBlock]:
    """Rows of one split; window ids start at `first_window`."""
    rows: list[PerturbedBlock] = []
    jn = joiner(lang)
    for w, win in enumerate(windows, start=first_window):
        dist = perturb_window([corpus.sentences[lang][r] for r in win.rows], perturber, n)
        if not dist:
            continue
        for k in k_list:
            source = corpus.text(pivot, win.rows[:k])
            rows.append(PerturbedBlock(
                id=f"{lang}-w{w:03d}-k{k}", lang_pair=f"{pivot}-{lang}", window=w, k=k,
                split=corpus.split, url=win.url, rows=win.rows[:k], source=source,
                source_n_tokens=n_tokens(source), reference=corpus.text(lang, win.rows[:k]),
                distractors=[jn.join(t for _c, t in d[:k]) for d in dist],
                categories=[[c for c, _t in d[:k]] for d in dist]))
    return rows


def token_counter(name: str) -> Callable[[str], int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    return lambda text: len(tok(text, add_special_tokens=False)["input_ids"])


def dataset_path(backend: str, lang: str) -> Path:
    return DATA_DIR / f"every_sentence_{backend}_{lang}.jsonl"


def save_rows(rows: Sequence[PerturbedBlock], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")


def load_rows(backend: str, lang: str) -> list[PerturbedBlock]:
    path = dataset_path(backend, lang)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found — run `python -m part1_block_alignment.perturb_every_sentence "
            f"--backend {backend} --langs {lang}` first")
    return [PerturbedBlock.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()]


def _log_stats(rows: Sequence[PerturbedBlock], n_windows: int, n: int) -> None:
    by_window = {r.window: len(r.distractors) for r in rows}
    ms = list(by_window.values())
    logger.info(f"  windows: {len(ms)} kept of {n_windows}, distractors per window "
                f"{dict(sorted(Counter(ms).items()))}")
    logger.info("  coverage (windows with >= D distractors): " + "  ".join(
        f"D={D}: {sum(m >= D for m in ms) / max(1, len(ms)):.2f}" for D in range(1, n + 1)))
    for k in sorted({r.k for r in rows}):
        rk = [r for r in rows if r.k == k]
        cats = Counter(c for r in rk for d in r.categories for c in d)
        tot = sum(cats.values())
        toks = np.array([r.source_n_tokens for r in rk])
        logger.info(f"  k={k}: {len(rk)} rows, source tokens mean {toks.mean():.0f} max {toks.max()}, "
                    "categories " + " ".join(f"{c}={cats[c] / tot:.2f}" for c in CATEGORIES))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["plus", "raw"], default="plus")
    ap.add_argument("--splits", nargs="+", default=["dev", "devtest"])
    ap.add_argument("--langs", nargs="+", default=["de", "es", "fr", "ru"])
    ap.add_argument("--pivot", default="en")
    ap.add_argument("--k_list", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                    help="Block lengths; windows are max(k_list) sentences long.")
    ap.add_argument("--n_distractors", type=int, default=6, help="Distractors per window, at most.")
    ap.add_argument("--backend", choices=["spacy", "heuristic", "auto"], default="spacy")
    ap.add_argument("--tokenizer", default="xlm-roberta-large",
                    help="Counts the tokens of the English source block.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_windows", type=int, default=None, help="Per split (smoke tests).")
    ap.add_argument("--dry_run", action="store_true", help="Print coverage and an example only.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    corpora = [load_corpus(args.source, sp) for sp in args.splits]
    windows = [build_blocks(c, max(args.k_list))[:args.max_windows] for c in corpora]
    n_tokens = token_counter(args.tokenizer)
    for lang in args.langs:
        pert = build_perturber([s for c in corpora for s in c.sentences[lang]], lang,
                               args.seed, args.backend)
        logger.info(f"\n== {args.pivot}-{lang} ({backend_name(pert)}) ==")
        rows: list[PerturbedBlock] = []
        start = 0
        for corpus, wins in zip(corpora, windows):
            rows += build_rows(corpus, wins, start, lang, args.pivot, args.k_list, pert,
                               args.n_distractors, n_tokens)
            start += len(wins)
        _log_stats(rows, start, args.n_distractors)
        if args.dry_run:
            r = rows[-1]
            logger.info(f"  example {r.id}\n    source:    {r.source}\n    reference: {r.reference}")
            for d, cats in zip(r.distractors[:3], r.categories):
                logger.info(f"    {'/'.join(cats)}: {d}")
        else:
            path = dataset_path(backend_name(pert), lang)
            save_rows(rows, path)
            logger.info(f"  -> {path}")


if __name__ == "__main__":
    main()
