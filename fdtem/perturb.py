"""Sentence-level perturbations used to build hard negatives."""
from __future__ import annotations

import random as _random
from typing import List, Optional, Tuple

from .languages import NO_SPACE_LANGS

import random as _random

_DIGITS = "0123456789"


def _tokens(text: str, lang: str) -> Tuple[List[str], bool]:
    """Return (tokens, is_char_level). Char-level for no-space languages."""
    if lang in NO_SPACE_LANGS:
        return list(text), True
    return text.split(" "), False


def _detok(tokens: List[str], char_level: bool) -> str:
    return ("" if char_level else " ").join(tokens)


def perturb(text: str, lang: str, kind: str, rng: _random.Random) -> Optional[str]:
    """Apply one error-like perturbation. Returns None if not applicable.

    kinds
    -----
      number   replace one digit with a different digit  (number error)
      delete   drop one content token                    (omission)
      swap     swap two adjacent content tokens          (reordering)
      replace  overwrite one token with another from the same text (mistranslation)
    """
    toks, char_level = _tokens(text, lang)
    if len(toks) < 3:
        return None
    idxs = list(range(len(toks)))

    if kind == "number":
        digit_pos = [(i, j) for i in idxs for j, c in enumerate(toks[i]) if c in _DIGITS]
        if not digit_pos:
            return None
        i, j = rng.choice(digit_pos)
        old = toks[i][j]
        new = rng.choice([d for d in _DIGITS if d != old])
        toks[i] = toks[i][:j] + new + toks[i][j + 1:]
        return _detok(toks, char_level)

    if kind == "delete":
        i = rng.randrange(len(toks))
        del toks[i]
        return _detok(toks, char_level)

    if kind == "swap":
        if len(toks) < 4:
            return None
        i = rng.randrange(len(toks) - 1)
        toks[i], toks[i + 1] = toks[i + 1], toks[i]
        out = _detok(toks, char_level)
        return out if out != text else None

    if kind == "replace":
        if len(set(toks)) < 2:
            return None
        i = rng.randrange(len(toks))
        choices = [t for t in toks if t != toks[i]]
        if not choices:
            return None
        toks[i] = rng.choice(choices)
        return _detok(toks, char_level)

    raise ValueError(f"unknown perturbation kind {kind!r}")


PERTURBATIONS = ["number", "delete", "swap", "replace"]

