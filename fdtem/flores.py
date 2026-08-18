"""FLORES / FLORES+ loading, row-aligned across languages.

`load_flores_source("plus"|"raw", ...)` returns a `FloresData` whose sentences
are parallel row by row; `urls` groups rows into source articles, which is what
makes document-level blocks possible.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .languages import FLORES_CODE, NO_SPACE_LANGS

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# Data — FLORES-200 (multi-parallel) + pseudo-paragraph builder
# ════════════════════════════════════════════════════════════════════════════
# We read the *raw* FLORES-200 distribution (plain aligned text files), not the
# HF dataset scripts — those hardcode a `/datasets` path that is not writable on
# the cluster. The raw layout is:
#     <flores_dir>/{dev,devtest}/<code>.<split>     one sentence per line
#     <flores_dir>/metadata_<split>.tsv             URL/domain/... per line
# Line i is parallel across every language. To download once:
#     wget https://tinyurl.com/flores200dataset -O flores200_dataset.tar.gz
#     tar -xzf flores200_dataset.tar.gz             # -> flores200_dataset/
# Point at it with $FLORES_DIR; otherwise we auto-discover the copy HF already
# extracted into the datasets cache.


def _find_flores_dir() -> Optional[Path]:
    """Locate an extracted flores200_dataset directory."""
    import glob
    env = os.environ.get("FLORES_DIR")
    cands: List[str] = [env] if env else []
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/scratch/hf_cache"))
    cands += [
        os.path.expanduser("~/scratch/flores200_dataset"),
        "data/flores200_dataset",
    ]
    cands += glob.glob(os.path.join(hf_home, "datasets", "downloads", "extracted",
                                    "*", "flores200_dataset"))
    cands += glob.glob(os.path.expanduser(
        "~/.cache/huggingface/datasets/downloads/extracted/*/flores200_dataset"))
    for c in cands:
        if c and (Path(c) / "devtest").is_dir():
            return Path(c)
    return None


@dataclass
class FloresData:
    langs: List[str]                  # ISO codes actually loaded
    sentences: Dict[str, List[str]]   # lang -> N aligned sentences
    urls: List[str]                   # per-row article URL (for grouping)

    @property
    def n(self) -> int:
        return len(next(iter(self.sentences.values())))


def load_flores(langs: Sequence[str], split: str = "devtest",
                max_sents: Optional[int] = None,
                flores_dir: Optional[str] = None) -> FloresData:
    """Load FLORES-200 for several languages from the raw text files, row-aligned."""
    root = Path(flores_dir) if flores_dir else _find_flores_dir()
    if root is None or not (root / split).is_dir():
        raise FileNotFoundError(
            "Could not find an extracted flores200_dataset directory. Set "
            "$FLORES_DIR to it (see representation_analysis_common.py header for the download "
            "command).")
    logger.info(f"  FLORES root: {root}")

    # article URLs for pseudo-paragraph grouping (one per sentence row)
    urls: List[str] = []
    meta = root / f"metadata_{split}.tsv"
    if meta.is_file():
        with open(meta, encoding="utf-8") as fh:
            rows = fh.read().splitlines()
        urls = [r.split("\t")[0] for r in rows[1:]]  # skip header; col0 = URL

    sentences: Dict[str, List[str]] = {}
    kept: List[str] = []
    ref_n = None
    for lang in langs:
        code = FLORES_CODE.get(lang)
        if code is None:
            logger.warning(f"  no FLORES code for {lang!r}, skipping")
            continue
        fp = root / split / f"{code}.{split}"
        if not fp.is_file():
            logger.warning(f"  missing FLORES file {fp}, skipping {lang}")
            continue
        with open(fp, encoding="utf-8") as fh:
            sents = fh.read().splitlines()
        if max_sents:
            sents = sents[:max_sents]
        if ref_n is None:
            ref_n = len(sents)
        elif len(sents) != ref_n:
            raise RuntimeError(f"FLORES length mismatch for {lang}: "
                               f"{len(sents)} vs {ref_n}")
        sentences[lang] = sents
        kept.append(lang)
    if not kept:
        raise RuntimeError("No FLORES languages loaded.")
    if not urls or len(urls) < ref_n:
        urls = [str(i) for i in range(ref_n)]      # fallback: every row its own "article"
    urls = urls[:ref_n]
    logger.info(f"  FLORES {split}: {len(kept)} langs × {ref_n} sentences")
    return FloresData(langs=kept, sentences=sentences, urls=urls)


# ────────────────────────────────────────────────────────────────────────────
# Official FLORES+ (OLDI) loader — the access path documented on the dataset card
# ────────────────────────────────────────────────────────────────────────────
# Unlike load_flores() above (which reads the legacy flores200_dataset text dump),
# this follows the *official* FLORES+ guidelines verbatim:
#
#     from datasets import load_dataset
#     ds = load_dataset("openlanguagedata/flores_plus", "fra_Latn", split="devtest")
#
# FLORES+ is gated: you must (1) `pip install datasets`, (2) log in to the HF Hub
# (`hf auth login` or huggingface_hub.login()), and (3) accept the terms of use at
# https://huggingface.co/datasets/openlanguagedata/flores_plus once.
#
# Each languoid config is "<iso_639_3>_<iso_15924>" and rows carry a shared `id`
# field; rows with the same id+split are mutual translations, so we align across
# languages by id to recover the multi-parallel matrix. NB the codes differ from
# the legacy FLORES-200 ones in places — most notably Mandarin is cmn_Hans (not
# zho_Hans). Names/codes follow the dataset card's language-coverage table.
FLORES_PLUS_REPO = "openlanguagedata/flores_plus"
FLORES_PLUS_CODE = {
    "en": "eng_Latn", "de": "deu_Latn", "es": "spa_Latn", "fr": "fra_Latn",
    "ru": "rus_Cyrl", "zh": "cmn_Hans", "ar": "arb_Arab", "hi": "hin_Deva",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "tr": "tur_Latn", "vi": "vie_Latn",
    "sw": "swh_Latn", "el": "ell_Grek", "th": "tha_Thai", "bg": "bul_Cyrl",
    "ur": "urd_Arab",
}


def load_flores_plus(langs: Sequence[str], split: str = "devtest",
                     max_sents: Optional[int] = None,
                     repo: str = FLORES_PLUS_REPO) -> FloresData:
    """Load the official FLORES+ benchmark (OLDI), row-aligned across languages.

    Follows the dataset card's recommended access pattern
    ``load_dataset(repo, "<iso_639_3>_<iso_15924>", split=split)`` per languoid,
    then intersects the shared ``id`` field so every loaded language is parallel.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError("FLORES+ needs the `datasets` package (pip install datasets).") from exc

    # Resolve the stored HF token explicitly: FLORES+ is gated, and the datasets /
    # hf:// download path only authenticates when HF_TOKEN is in the environment.
    # NOTE huggingface_hub.get_token() reads $HF_HOME/token, so when HF_HOME is
    # redirected (e.g. to scratch) it misses the default ~/.cache/huggingface/token
    # — hence the explicit default-file fallback below.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:  # noqa: BLE001
            token = None
    if not token:
        default_token = Path.home() / ".cache" / "huggingface" / "token"
        if default_token.is_file():
            token = default_token.read_text(encoding="utf-8").strip() or None
    if token:
        # make the resolved token visible to the fsspec/datasets download path
        os.environ.setdefault("HF_TOKEN", token)

    per_lang_rows: Dict[str, Dict[str, str]] = {}   # lang -> {id: text}
    meta_url: Dict[str, str] = {}                    # id -> article url (for E3 grouping)
    kept: List[str] = []
    for lang in langs:
        code = FLORES_PLUS_CODE.get(lang)
        if code is None:
            logger.warning(f"  no FLORES+ config for {lang!r}, skipping")
            continue
        try:
            ds = load_dataset(repo, code, split=split, token=token)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load FLORES+ config {code!r} (split={split!r}) from {repo}. "
                "Make sure you are logged in to the HF Hub (`hf auth login` or "
                "huggingface_hub.login()) and have accepted the dataset terms at "
                f"https://huggingface.co/datasets/{repo}.") from exc
        rows: Dict[str, str] = {}
        for r in ds:
            rid = str(r["id"])
            rows[rid] = r["text"]
            meta_url.setdefault(rid, r.get("url", "") or "")
        per_lang_rows[lang] = rows
        kept.append(lang)
        logger.info(f"  FLORES+ {code} {split}: {len(rows)} rows")
    if not kept:
        raise RuntimeError("No FLORES+ languages loaded.")

    # keep only ids present in every language → guaranteed multi-parallel; sort numerically
    common = set.intersection(*(set(per_lang_rows[l]) for l in kept))

    def _key(i: str):
        try:
            return (0, int(i))
        except ValueError:
            return (1, i)

    ids = sorted(common, key=_key)
    if max_sents:
        ids = ids[:max_sents]
    if not ids:
        raise RuntimeError("FLORES+: no ids shared across the requested languages.")

    sentences = {l: [per_lang_rows[l][i] for i in ids] for l in kept}
    urls = [meta_url.get(i, "") or i for i in ids]
    logger.info(f"  FLORES+ {split}: {len(kept)} langs × {len(ids)} aligned sentences")
    return FloresData(langs=kept, sentences=sentences, urls=urls)


def load_flores_source(source: str, langs: Sequence[str], split: str = "devtest",
                       max_sents: Optional[int] = None) -> FloresData:
    """Dispatch FLORES loading by source.

    ``source="plus"`` → official OLDI FLORES+ via the HF Hub (gated; recommended).
    ``source="raw"``  → legacy flores200_dataset local text files.
    """
    if source == "plus":
        return load_flores_plus(langs, split, max_sents)
    if source == "raw":
        return load_flores(langs, split, max_sents)
    raise ValueError(f"Unknown flores source {source!r} (use 'plus' or 'raw').")


def build_pseudo_docs(data: FloresData, k: int) -> Dict[str, List[str]]:
    """Concatenate k consecutive same-article sentences into pseudo-paragraphs.

    FLORES rows are in document order and the URL column marks article
    boundaries, so windows stay semantically coherent — and stay parallel across
    languages because we apply identical row windows to every language.
    k=1 returns the original sentences.
    """
    if k <= 1:
        return {l: list(s) for l, s in data.sentences.items()}
    # contiguous windows that never cross an article boundary
    windows: List[List[int]] = []
    i, n = 0, data.n
    while i < n:
        j = i
        while j < n and j - i < k and data.urls[j] == data.urls[i]:
            j += 1
        if j - i == k:               # only keep full-length windows
            windows.append(list(range(i, j)))
        i = j if j > i else i + 1
    joiner = {l: ("" if l in NO_SPACE_LANGS else " ") for l in data.langs}
    out: Dict[str, List[str]] = {}
    for lang, sents in data.sentences.items():
        out[lang] = [joiner[lang].join(sents[w] for w in win) for win in windows]
    return out


