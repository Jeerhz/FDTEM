"""Fetch FLORES once and cache it under data/ as data/flores_<source>_<split>.json.

    python -m part1_block_alignment.load_flores --source plus --langs en de es fr ru --splits dev devtest

`plus` is the gated FLORES+ on the HF Hub (log in and accept the terms once);
`raw` is a local flores200_dataset dump. The other scripts read the cached files
through `load_corpus`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common.flores import FloresCorpus, load_flores
from part1_block_alignment import DATA_DIR


def corpus_path(source: str, split: str) -> Path:
    return DATA_DIR / f"flores_{source}_{split}.json"


def load_corpus(source: str, split: str) -> FloresCorpus:
    path = corpus_path(source, split)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found — run `python -m part1_block_alignment.load_flores "
            f"--source {source} --splits {split}` first")
    return FloresCorpus.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["plus", "raw"], default="plus")
    ap.add_argument("--langs", nargs="+", default=["en", "de", "es", "fr", "ru"])
    ap.add_argument("--splits", nargs="+", default=["dev", "devtest"])
    ap.add_argument("--max_sents", type=int, default=None)
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        corpus = load_flores(args.source, args.langs, split, args.max_sents)
        path = corpus_path(args.source, split)
        path.write_text(corpus.model_dump_json(indent=2), encoding="utf-8")
        print(f"{split}: {corpus.n} rows, {len(set(corpus.urls))} articles, "
              f"langs={corpus.langs} -> {path}")


if __name__ == "__main__":
    main()
