#!/usr/bin/env python3
"""
data.py — aligned sentence units and nested k_max-unit families.

Sources
-------
  wmt24pp   : google/wmt24pp (HF). Human post-edits = the error-free, high-quality
              baseline (Deutsch et al. 2025). Units are the dataset segments
              (aligned by construction) or bertalign-realigned sentences.
  flores    : FLORES-200 raw files (robustness set, longer paragraphs). Gold
              translations, docs grouped by article URL.
  synthetic : deterministic fake parallel docs — logic smoke test only.

Paired design: the sampling unit is a FAMILY — a contiguous window of k_max
units. Every scale k ∈ k_list is the k-prefix of the same family, so the
candidate pool is identical at all scales (no 'short blocks are plentiful,
long blocks are rare' selection confound) and only length varies across scales.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from common import iso_of_lp, joiner_for, md5_short

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Doc:
    doc_id: str
    src_units: List[str]
    tgt_units: List[str]


@dataclass
class Family:
    """A contiguous window of k_max aligned units; every scale is a prefix."""

    lang: str                 # target ISO code (e.g. "de")
    lp: str                   # full language-pair tag (e.g. "en-de_DE")
    k_max: int
    family_idx: int           # index within lang (after sampling)
    doc_id: str
    start: int                # unit offset inside the document
    src_units: List[str]      # length k_max
    tgt_units: List[str]      # length k_max
    prefix_tokens: Optional[dict] = None  # k -> (n_tokens_src, n_tokens_tgt)
    truncated: bool = False   # k_max prefix exceeds the token budget

    @property
    def family_id(self) -> str:
        return f"{self.lang}_{self.doc_id}_{self.start}"

    def join_src(self, k: Optional[int] = None) -> str:
        return " ".join(self.src_units[:k])      # source is always English here

    def join_tgt(self, units: Optional[List[str]] = None,
                 k: Optional[int] = None) -> str:
        u = units if units is not None else self.tgt_units
        return joiner_for(self.lang).join(u[:k])


# ════════════════════════════════════════════════════════════════════════════
# Token counting (only used to enforce the ≤500-token encoder budget)
# ════════════════════════════════════════════════════════════════════════════
class TokenCounter:
    def __init__(self, tokenizer_ref: Optional[str]):
        self.tok = None
        if tokenizer_ref:
            try:
                from transformers import AutoTokenizer  # noqa: PLC0415
                self.tok = AutoTokenizer.from_pretrained(tokenizer_ref)
                logger.info(f"  token counter: {tokenizer_ref}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"  token counter unavailable ({exc}); "
                               "falling back to whitespace*1.3 estimate")

    def count(self, text: str) -> int:
        if self.tok is not None:
            return len(self.tok(text, add_special_tokens=True)["input_ids"])
        return int(len(text.split()) * 1.3) + 2


# ════════════════════════════════════════════════════════════════════════════
# Loaders → Dict[lang_iso, List[Doc]]
# ════════════════════════════════════════════════════════════════════════════
def load_wmt24pp(cfg: dict) -> Dict[str, List[Doc]]:
    from datasets import load_dataset  # noqa: PLC0415
    d = cfg["data"]
    out: Dict[str, List[Doc]] = {}
    lp_of: Dict[str, str] = {}
    for lp in d["wmt24pp_lps"]:
        lang = iso_of_lp(lp)
        logger.info(f"  loading google/wmt24pp:{lp}")
        ds = load_dataset("google/wmt24pp", lp, split="train")
        docs: List[Doc] = []
        cur_id, srcs, tgts = None, [], []

        def flush():
            if cur_id is not None and srcs:
                docs.append(Doc(doc_id=md5_short(f"{lp}|{cur_id}"), src_units=list(srcs),
                                tgt_units=list(tgts)))

        for row in ds:
            if d.get("drop_bad_source", True) and row.get("is_bad_source"):
                continue
            src, tgt = (row["source"] or "").strip(), (row["target"] or "").strip()
            if len(src) < d["min_unit_chars"] or len(tgt) < d["min_unit_chars"]:
                continue
            if row["document_id"] != cur_id:
                flush()
                cur_id, srcs, tgts = row["document_id"], [], []
            srcs.append(src)
            tgts.append(tgt)
        flush()
        if d.get("resegment", "segment") == "bertalign":
            docs = [_bertalign_doc(doc, lang) for doc in docs]
            docs = [doc for doc in docs if doc is not None]
        out[lang] = docs
        lp_of[lang] = lp
        logger.info(f"    {lang}: {len(docs)} docs, "
                    f"{sum(len(x.src_units) for x in docs)} units")
    cfg["_lp_of"] = lp_of
    return out


def _bertalign_doc(doc: Doc, lang: str) -> Optional[Doc]:
    """Re-align a document into sentence beads with bertalign (optional dep)."""
    try:
        from bertalign import Bertalign  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "resegment: bertalign requires `pip install "
            "git+https://github.com/bfsujason/bertalign`") from exc
    src_text = " ".join(doc.src_units)
    tgt_text = joiner_for(lang).join(doc.tgt_units)
    try:
        aligner = Bertalign(src_text, tgt_text, is_split=False)
        aligner.align_sents()
        srcs, tgts = [], []
        for bead in aligner.result:
            s = " ".join(aligner.src_sents[i] for i in bead[0]).strip()
            t = " ".join(aligner.tgt_sents[i] for i in bead[1]).strip()
            if s and t:
                srcs.append(s)
                tgts.append(t)
        if len(srcs) < 2:
            return None
        return Doc(doc_id=doc.doc_id, src_units=srcs, tgt_units=tgts)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"    bertalign failed on {doc.doc_id}: {exc}")
        return None


# FLORES raw layout (see scripts/representation_analysis_common.py header).
_FLORES_CODE = {
    "en": "eng_Latn", "de": "deu_Latn", "es": "spa_Latn", "fr": "fra_Latn",
    "ru": "rus_Cyrl", "zh": "zho_Hans", "ar": "arb_Arab", "hi": "hin_Deva",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "pt": "por_Latn",
}


def load_flores(cfg: dict) -> Dict[str, List[Doc]]:
    d = cfg["data"]
    root = Path(d["flores_dir"] or "").expanduser()
    split = d.get("flores_split", "devtest")
    if not (root / split).is_dir():
        raise FileNotFoundError(
            f"FLORES dir not found at {root!s}. Set $FLORES_DIR to the extracted "
            "flores200_dataset directory.")

    def read(lang: str) -> List[str]:
        fp = root / split / f"{_FLORES_CODE[lang]}.{split}"
        return fp.read_text(encoding="utf-8").splitlines()

    en = read("en")
    meta = root / f"metadata_{split}.tsv"
    if meta.is_file():
        urls = [r.split("\t")[0] for r in meta.read_text(encoding="utf-8").splitlines()[1:]]
    else:
        urls = [str(i) for i in range(len(en))]

    out: Dict[str, List[Doc]] = {}
    lp_of: Dict[str, str] = {}
    for lang in d["flores_langs"]:
        tgt = read(lang)
        assert len(tgt) == len(en), f"FLORES row mismatch for {lang}"
        docs: List[Doc] = []
        i = 0
        while i < len(en):
            j = i
            while j < len(en) and urls[j] == urls[i]:
                j += 1
            if j - i >= 2:
                docs.append(Doc(doc_id=md5_short(f"flores|{lang}|{urls[i]}|{i}"),
                                src_units=en[i:j], tgt_units=tgt[i:j]))
            i = j
        out[lang] = docs
        lp_of[lang] = f"en-{lang}(flores)"
        logger.info(f"  flores {lang}: {len(docs)} docs")
    cfg["_lp_of"] = lp_of
    return out


_SYN_TOPICS = ["the budget", "water policy", "border control", "rail transport",
               "school reform", "energy prices", "public housing", "food safety"]
_SYN_VERBS = ["approved", "rejected", "postponed", "amended", "reviewed", "debated"]


def load_synthetic(cfg: dict, langs: List[str], n_docs: int = 40,
                   units_per_doc: int = 10, seed: int = 0) -> Dict[str, List[Doc]]:
    """Deterministic fake parallel docs — enough structure for rule perturbations."""
    out: Dict[str, List[Doc]] = {}
    lp_of: Dict[str, str] = {}
    for lang in langs:
        rng = random.Random(f"{seed}|{lang}")
        docs = []
        for di in range(n_docs):
            srcs, tgts = [], []
            for si in range(units_per_doc):
                topic = rng.choice(_SYN_TOPICS)
                verb = rng.choice(_SYN_VERBS)
                year = rng.randint(1998, 2026)
                n = rng.randint(3, 97)
                srcs.append(f"The committee {verb} measure {n} on {topic} in {year} "
                            f"after a long debate in session {si + 1}.")
                tgts.append(f"Der Ausschuss hat Vorlage {n} zu {topic} im Jahr {year} "
                            f"nach langer Debatte in Sitzung {si + 1} {verb}.")
            docs.append(Doc(doc_id=md5_short(f"syn|{lang}|{di}"),
                            src_units=srcs, tgt_units=tgts))
        out[lang] = docs
        lp_of[lang] = f"en-{lang}(synthetic)"
    cfg["_lp_of"] = lp_of
    return out


def load_docs(cfg: dict) -> Dict[str, List[Doc]]:
    src = cfg["data"]["source"]
    if src == "wmt24pp":
        return load_wmt24pp(cfg)
    if src == "flores":
        return load_flores(cfg)
    if src == "synthetic":
        langs = cfg["data"].get("synthetic_langs") or [iso_of_lp(x) for x in
                                                       cfg["data"]["wmt24pp_lps"]]
        return load_synthetic(cfg, langs, seed=cfg["seed"])
    raise ValueError(f"unknown data.source {src!r}")


# ════════════════════════════════════════════════════════════════════════════
# Families (paired across scales)
# ════════════════════════════════════════════════════════════════════════════
def build_families(cfg: dict, docs_by_lang: Dict[str, List[Doc]]) -> List[Family]:
    """Non-overlapping contiguous k_max-unit windows, token-filtered on the FULL
    k_max prefix (so every scale k ≤ k_max fits the encoder budget), sampled to
    n_families per language.

    Only documents long enough for k_max ever enter the pool, and every scale is
    a prefix of the same window — the candidate pool is IDENTICAL at all scales."""
    des = cfg["design"]
    k_list = sorted(des["k_list"])
    k_max = max(k_list)
    counter = TokenCounter(des.get("token_counter"))
    rng = random.Random(cfg["seed"])
    families: List[Family] = []
    for lang, docs in docs_by_lang.items():
        lp = cfg.get("_lp_of", {}).get(lang, lang)
        n_long = sum(1 for d in docs
                     if min(len(d.src_units), len(d.tgt_units)) >= k_max)
        cand: List[Family] = []
        for doc in docs:
            n = min(len(doc.src_units), len(doc.tgt_units))
            for start in range(0, n - k_max + 1, k_max):
                cand.append(Family(lang=lang, lp=lp, k_max=k_max, family_idx=-1,
                                   doc_id=doc.doc_id, start=start,
                                   src_units=doc.src_units[start:start + k_max],
                                   tgt_units=doc.tgt_units[start:start + k_max]))
        rng.shuffle(cand)
        kept: List[Family] = []
        for f in cand:
            if len(kept) >= des["n_families"]:
                break
            f.prefix_tokens = {k: (counter.count(f.join_src(k)),
                                   counter.count(f.join_tgt(k=k)))
                               for k in k_list}
            ns, nt = f.prefix_tokens[k_max]
            f.truncated = max(ns, nt) > des["max_tokens"]
            if f.truncated and des.get("drop_truncated", True):
                continue
            kept.append(f)
        for i, f in enumerate(kept):
            f.family_idx = i
        families.extend(kept)
        logger.info(f"  families[{lang}]: {len(kept)} kept of {len(cand)} candidate "
                    f"k_max={k_max} windows ({n_long}/{len(docs)} docs long enough)")
    return families
