#!/usr/bin/env python3
"""
matched_core_common.py — the matched-core length-sensitivity design.

Every evaluation input is built as

    x(L, φ, π) = FILLER_pre ⊕ CORE ⊕ FILLER_post        |x| = L reference tokens

where CORE is a fixed FLORES+ sentence (the signal carrier: its clean target
vs a single-perturbation negative), and the filler carries ALL of the length
variation. The same core appears in every cell of the (L, φ, π) grid — an
intra-item design: any performance change is attributable to length, not
content difficulty.

Filler types φ (the dilution gradient):
  inert       one lexically neutral phrase repeated — purely mechanical effect
              of sequence length (softmax dispersion, pooling dilution); no
              competing content.
  neutral     in-domain FLORES sentences from OTHER articles — adds the cost
              of separating signal from in-distribution non-signal.
  distractor  sentences from the SAME article (topically close to the core,
              competing for the query's similarity mass) — active interference.
  natural     the contiguous corpus stream around the core — the ecological
              condition; confounds length with semantic drift (upper bound).

Positions π: fraction of filler tokens placed BEFORE the core (0 = core at
start, 0.5 = middle, 1 = end).

All lengths are counted in REFERENCE tokens (XLM-R sentencepiece), the shared
backbone vocabulary of COMET/XLM-R, and are capped so that no encoder in the
zoo (all max_len = 512) ever truncates the input — otherwise we would measure
truncation, not length. The core is by construction never truncated.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from block_xsim_common import Perturber

PHIS = ("inert", "neutral", "distractor", "natural")

NEUTRAL_PHRASE = {
    "de": "und so weiter",
    "es": "y así sucesivamente",
    "fr": "et ainsi de suite",
    "ru": "и так далее",
    "en": "and so on",
}

_REF_TOK = None


def ref_tokenizer():
    global _REF_TOK
    if _REF_TOK is None:
        from transformers import AutoTokenizer
        _REF_TOK = AutoTokenizer.from_pretrained("xlm-roberta-large")
    return _REF_TOK


class TokenCounter:
    """text → reference-token ids, memoised (fillers are re-used heavily)."""

    def __init__(self):
        self.tok = ref_tokenizer()
        self._cache: Dict[str, List[int]] = {}

    def ids(self, s: str) -> List[int]:
        if s not in self._cache:
            self._cache[s] = self.tok(s, add_special_tokens=False)["input_ids"]
        return self._cache[s]

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids))


@dataclass
class Item:
    idx: int                 # row in the concatenated corpus (all splits)
    url: str
    src: str                 # pivot-language core sentence (the query)
    core: str                # target-language core, clean
    core_ids: List[int]      # reference tokens of the clean core
    negs: Dict[str, str]     # category -> single-perturbation core


def pick_items(rows: List[Tuple[int, str, str, str]], pert: Perturber,
               tc: TokenCounter, n_items: int, min_tok: int, max_tok: int,
               seed: int, categories: Sequence[str]) -> List[Item]:
    """Sample cores: token length in [min_tok, max_tok] and at least two
    perturbation categories applicable (so each item carries ≥2 negatives)."""
    rng = random.Random(f"mc|{seed}")
    order = list(range(len(rows)))
    rng.shuffle(order)
    items: List[Item] = []
    for i in order:
        idx, url, src, tgt = rows[i]
        ids = tc.ids(tgt)
        if not (min_tok <= len(ids) <= max_tok):
            continue
        negs = {}
        for cat in categories:
            vs = pert.variants(tgt, cat, 1)
            if vs:
                negs[cat] = vs[0]
        if len(negs) < 2:
            continue
        items.append(Item(idx=idx, url=url, src=src, core=tgt,
                          core_ids=ids, negs=negs))
        if len(items) >= n_items:
            break
    return items


class FillerBank:
    """Builds the φ filler token streams for one target language."""

    def __init__(self, sentences: Sequence[str], urls: Sequence[str],
                 tc: TokenCounter, lang: str, seed: int = 42):
        self.sentences = list(sentences)
        self.urls = list(urls)
        self.tc = tc
        self.lang = lang
        self.seed = seed
        phrase = NEUTRAL_PHRASE.get(lang, NEUTRAL_PHRASE["en"])
        self._inert_unit = tc.ids(phrase + ",")

    def _rng(self, item: Item, phi: str) -> random.Random:
        return random.Random(f"{self.seed}|{self.lang}|{item.idx}|{phi}")

    def _stream(self, indices: Sequence[int], need: int) -> List[int]:
        out: List[int] = []
        i = 0
        while len(out) < need and indices:
            out += self.tc.ids(self.sentences[indices[i % len(indices)]])
            i += 1
        return out[:need]

    def filler_ids(self, item: Item, phi: str, n_pre: int, n_post: int
                   ) -> Tuple[List[int], List[int]]:
        n = len(self.sentences)
        if phi == "inert":
            reps = (n_pre + n_post) // max(1, len(self._inert_unit)) + 2
            stream = (self._inert_unit * reps)
            return stream[:n_pre], stream[n_pre:n_pre + n_post]
        if phi == "neutral":
            cand = [i for i in range(n) if self.urls[i] != item.url]
            self._rng(item, phi).shuffle(cand)
            s = self._stream(cand, n_pre + n_post)
            return s[:n_pre], s[n_pre:]
        if phi == "distractor":
            same = [i for i in range(n)
                    if self.urls[i] == item.url and i != item.idx]
            rest = [i for i in range(n) if self.urls[i] != item.url]
            rng = self._rng(item, phi)
            rng.shuffle(same)
            rng.shuffle(rest)
            s = self._stream(same + rest, n_pre + n_post)
            return s[:n_pre], s[n_pre:]
        if phi == "natural":
            # prefix = corpus stream just BEFORE the core (natural order, take
            # its tail); suffix = stream just AFTER (take its head). Crossing
            # article boundaries is intentional: that IS the semantic drift of
            # real long documents.
            pre: List[int] = []
            j = item.idx
            while len(pre) < n_pre:
                j = (j - 1) % n
                pre = self.tc.ids(self.sentences[j]) + pre
            post: List[int] = []
            j = item.idx
            while len(post) < n_post:
                j = (j + 1) % n
                post += self.tc.ids(self.sentences[j])
            return pre[len(pre) - n_pre:], post[:n_post]
        raise ValueError(f"unknown filler type {phi!r}")


def build_input(bank: FillerBank, item: Item, core_text: str,
                L: int, phi: str, pi: float) -> Optional[str]:
    """Compose x(L, φ, π) with `core_text` (clean core or one negative) in
    place of the core. The filler is sized from the CLEAN core's token count,
    so the clean input and its negatives share byte-identical filler."""
    n_fill = L - len(item.core_ids)
    if n_fill < 0:
        return None
    n_pre = int(round(pi * n_fill))
    n_post = n_fill - n_pre
    pre_ids, post_ids = bank.filler_ids(item, phi, n_pre, n_post)
    parts = []
    if pre_ids:
        parts.append(bank.tc.decode(pre_ids))
    parts.append(core_text)
    if post_ids:
        parts.append(bank.tc.decode(post_ids))
    return " ".join(parts)
