"""FLORES loading, row-aligned across languages.

`load_flores("plus", langs, split)` returns a `FloresCorpus` whose sentences are
parallel row by row; `urls` groups rows into source articles, which is what makes
document-level blocks possible.

FLORES+ is gated: `pip install datasets`, `hf auth login`, and accept the terms once at
https://huggingface.co/datasets/openlanguagedata/flores_plus.
"""
from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel

from common.auth import hf_token

logger = logging.getLogger(__name__)

# ISO-639-1 -> FLORES+ languoid config (OLDI codes; Mandarin is cmn_Hans there).
FLORES_PLUS_CODE = {
    "en": "eng_Latn", "de": "deu_Latn", "es": "spa_Latn", "fr": "fra_Latn",
    "ru": "rus_Cyrl", "zh": "cmn_Hans", "ar": "arb_Arab", "hi": "hin_Deva",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "tr": "tur_Latn", "vi": "vie_Latn",
    "sw": "swh_Latn", "el": "ell_Grek", "th": "tha_Thai", "bg": "bul_Cyrl",
    "ur": "urd_Arab",
}
# The legacy flores200_dataset text dump uses zho_Hans for Mandarin.
FLORES_CODE = {**FLORES_PLUS_CODE, "zh": "zho_Hans"}
FLORES_PLUS_REPO = "openlanguagedata/flores_plus"

# No whitespace word segmentation: perturbations work on characters, joins use "".
NO_SPACE_LANGS = {"zh", "ja", "th"}


def joiner(lang: str) -> str:
    return "" if lang in NO_SPACE_LANGS else " "


class FloresCorpus(BaseModel):
    """Row-aligned sentences for several languages; `urls[i]` is row i's article."""

    source: Literal["plus", "raw"]
    split: str
    langs: list[str]
    sentences: dict[str, list[str]]
    urls: list[str]

    @property
    def n(self) -> int:
        return len(self.urls)

    def text(self, lang: str, rows: Sequence[int]) -> str:
        return joiner(lang).join(self.sentences[lang][r] for r in rows)


def load_flores(source: str, langs: Sequence[str], split: str = "devtest",
                max_sents: int | None = None) -> FloresCorpus:
    """`source="plus"`: official FLORES+ from the HF Hub. `source="raw"`: local flores200 dump."""
    if source == "plus":
        return load_flores_plus(langs, split, max_sents)
    if source == "raw":
        return load_flores_raw(langs, split, max_sents)
    raise ValueError(f"unknown flores source {source!r} (use 'plus' or 'raw')")


def load_flores_plus(langs: Sequence[str], split: str = "devtest",
                     max_sents: int | None = None, repo: str = FLORES_PLUS_REPO) -> FloresCorpus:
    """One `load_dataset(repo, "<iso639_3>_<iso15924>")` per language, aligned on `id`."""
    from datasets import load_dataset

    token = hf_token()
    rows_by_lang: dict[str, dict[str, str]] = {}
    url_of: dict[str, str] = {}
    for lang in langs:
        code = FLORES_PLUS_CODE.get(lang)
        if code is None:
            logger.warning("no FLORES+ config for %r, skipping", lang)
            continue
        try:
            ds = load_dataset(repo, code, split=split, token=token)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"could not load FLORES+ {code!r}/{split!r}: log in to the HF Hub and accept "
                f"the terms at https://huggingface.co/datasets/{repo}") from exc
        rows_by_lang[lang] = {str(r["id"]): r["text"] for r in ds}
        for r in ds:
            url_of.setdefault(str(r["id"]), r.get("url") or "")
        logger.info("FLORES+ %s %s: %d rows", code, split, len(rows_by_lang[lang]))
    if not rows_by_lang:
        raise RuntimeError("no FLORES+ language loaded")

    shared = set.intersection(*(set(r) for r in rows_by_lang.values()))
    ids = sorted(shared, key=lambda i: (0, int(i)) if i.isdigit() else (1, i))[:max_sents]
    if not ids:
        raise RuntimeError("FLORES+: no ids shared across the requested languages")
    return FloresCorpus(
        source="plus", split=split, langs=list(rows_by_lang),
        sentences={lang: [rows[i] for i in ids] for lang, rows in rows_by_lang.items()},
        urls=[url_of.get(i) or i for i in ids])


def load_flores_raw(langs: Sequence[str], split: str = "devtest",
                    max_sents: int | None = None, flores_dir: str | None = None) -> FloresCorpus:
    """The legacy flores200_dataset text dump: <dir>/<split>/<code>.<split>, one sentence per line."""
    root = Path(flores_dir) if flores_dir else _find_flores_dir()
    if root is None or not (root / split).is_dir():
        raise FileNotFoundError("no flores200_dataset directory found; set $FLORES_DIR")

    meta = root / f"metadata_{split}.tsv"
    urls = [line.split("\t")[0] for line in meta.read_text(encoding="utf-8").splitlines()[1:]] \
        if meta.is_file() else []

    sentences: dict[str, list[str]] = {}
    for lang in langs:
        path = root / split / f"{FLORES_CODE[lang]}.{split}"
        if not path.is_file():
            logger.warning("missing %s, skipping %s", path, lang)
            continue
        sentences[lang] = path.read_text(encoding="utf-8").splitlines()[:max_sents]
    if not sentences:
        raise RuntimeError("no FLORES language loaded")
    n = {len(s) for s in sentences.values()}
    if len(n) != 1:
        raise RuntimeError(f"FLORES files disagree on length: {n}")
    n = n.pop()
    if len(urls) < n:
        urls = [str(i) for i in range(n)]  # every row its own article
    return FloresCorpus(source="raw", split=split, langs=list(sentences),
                        sentences=sentences, urls=urls[:n])


def _find_flores_dir() -> Path | None:
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/scratch/hf_cache"))
    candidates = [os.environ.get("FLORES_DIR"), os.path.expanduser("~/scratch/flores200_dataset")]
    candidates += glob.glob(os.path.join(hf_home, "datasets/downloads/extracted/*/flores200_dataset"))
    candidates += glob.glob(os.path.expanduser(
        "~/.cache/huggingface/datasets/downloads/extracted/*/flores200_dataset"))
    for c in candidates:
        if c and (Path(c) / "devtest").is_dir():
            return Path(c)
    return None
