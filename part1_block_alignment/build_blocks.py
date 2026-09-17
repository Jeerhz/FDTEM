"""Cut a FLORES corpus into blocks of k consecutive sentences of one article.

    python -m part1_block_alignment.build_blocks --source plus --splits dev devtest --k_list 1 2 3 4 5

Writes data/blocks_<source>_<split>_k<k>.json (a JSON list of Block). Blocks do
not overlap (stride = k) and never cross an article boundary.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import TypeAdapter

from common.flores import FloresCorpus
from part1_block_alignment import DATA_DIR
from part1_block_alignment.load_flores import load_corpus
from part1_block_alignment.models import Block

_BLOCKS = TypeAdapter(list[Block])


def build_blocks(corpus: FloresCorpus, k: int, stride: int | None = None) -> list[Block]:
    """Non-overlapping (stride=k) windows of k consecutive same-article rows.

    Overlapping windows would put near-duplicate blocks in the pool, which is a
    different (and confounded) retrieval problem — hence stride defaults to k.
    """
    stride = stride or k
    blocks: list[Block] = []
    i, n = 0, corpus.n
    while i < n:
        j = i
        while j < n and j - i < k and corpus.urls[j] == corpus.urls[i]:
            j += 1
        if j - i == k:
            blocks.append(Block(rows=list(range(i, j)), url=corpus.urls[i], split=corpus.split))
            i += stride
        else:
            i = j if j > i else i + 1
    return blocks


def block_text(corpus: FloresCorpus, block: Block, lang: str) -> str:
    return corpus.text(lang, block.rows)


def blocks_path(source: str, split: str, k: int) -> Path:
    return DATA_DIR / f"blocks_{source}_{split}_k{k}.json"


def load_blocks(source: str, split: str, k: int) -> list[Block]:
    path = blocks_path(source, split, k)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found — run `python -m part1_block_alignment.build_blocks "
            f"--source {source} --splits {split} --k_list {k}` first")
    return _BLOCKS.validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["plus", "raw"], default="plus")
    ap.add_argument("--splits", nargs="+", default=["dev", "devtest"])
    ap.add_argument("--k_list", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    for split in args.splits:
        corpus = load_corpus(args.source, split)
        for k in args.k_list:
            blocks = build_blocks(corpus, k)
            path = blocks_path(args.source, split, k)
            path.write_bytes(_BLOCKS.dump_json(blocks, indent=2))
            print(f"{split} k={k}: {len(blocks)} blocks -> {path}")


if __name__ == "__main__":
    main()
