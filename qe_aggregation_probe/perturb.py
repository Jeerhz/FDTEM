#!/usr/bin/env python3
"""
perturb.py — controlled error injection into single target sentences.

Backends
--------
  llm  : the published error-injection prompts of Zhang et al. 2026
         (arXiv:2510.22028, App. A.4) — base template + one task description per
         {minor,major} × {accuracy,fluency} — sent to any OpenAI-compatible chat
         endpoint. Responses are disk-cached so re-runs are free.
  rule : deterministic, length-preserving proxy edits (no API, smoke tests and
         lower-bound sanity checks). Adapted from the xsim++-style perturbations
         in scripts/representation_analysis_common.py.

Invariants enforced here (and asserted again at item-assembly time):
  * exactly ONE perturbation per sentence per condition;
  * the edit must not change sentence length materially (QC below), so
    paragraph length stays ~constant across m and quality is dissociated from
    length;
  * a failed/QC-rejected perturbation returns None — the caller must NEVER
    substitute the clean sentence for a missing perturbed one.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Dict, Optional

from common import NO_SPACE_LANGS, md5_short

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# Prompts (Zhang et al. 2026, App. A.4 — Figures 8–12)
# ════════════════════════════════════════════════════════════════════════════
BASE_PROMPT = """\
You are an expert in {src_lang} and {tgt_lang}. You are presented with the \
following {src_lang} source text and its {tgt_lang} translation.

{task_description}

Source text: {source}

Translation: {translation}

IMPORTANT NOTE: Only insert one such error in the translation. Do not insert \
any other error. Do not significantly increase or decrease the length of the \
translation. Do not bold any words or number the lines. Return ONLY the \
modified translation text, with no preamble, labels or quotation marks."""

TASK_DESCRIPTIONS: Dict[str, str] = {
    "major_accuracy": (
        "Your task is to revise the provided translation to introduce one major "
        "accuracy error. This error should be a significant mistranslation, "
        "omission, or addition that alters a key piece of information from the "
        "source text. The error must be noticeable and disrupt the intended "
        "meaning, but the sentence should remain grammatically correct and "
        "generally understandable."),
    "major_fluency": (
        "Your task is to revise the provided translation to introduce one major "
        "fluency error. This error should be a significant grammatical, spelling, "
        "or punctuation mistake that makes the translated text sound unnatural "
        "and difficult to read. The error must disrupt the flow of the text, but "
        "a reader should still be able to understand the intended message."),
    "minor_accuracy": (
        "Your task is to revise the provided translation to introduce one minor "
        "accuracy error. This error should be a subtle mistranslation (such as "
        "using a word with a slightly incorrect meaning) or a minor omission "
        "that is technically incorrect. The error must not significantly change "
        "the overall meaning of the sentence or hinder a reader's comprehension."),
    "minor_fluency": (
        "Your task is to revise the provided translation to introduce one minor "
        "fluency error. This error should be a small grammatical, spelling, or "
        "punctuation mistake. The error must be subtle and should not disrupt "
        "the natural flow of the translation or prevent a reader from easily "
        "understanding it."),
}

LANG_NAMES = {"en": "English", "de": "German", "zh": "Chinese", "es": "Spanish",
              "fr": "French", "ja": "Japanese", "ko": "Korean", "hi": "Hindi",
              "ar": "Arabic", "ru": "Russian", "pt": "Portuguese"}


# ════════════════════════════════════════════════════════════════════════════
# Quality control
# ════════════════════════════════════════════════════════════════════════════
_WRAP_RE = re.compile(r'^\s*(?:translation\s*:?\s*)?["\'«»„“”]?\s*', re.IGNORECASE)


def _strip_wrappers(text: str) -> str:
    t = text.strip()
    t = _WRAP_RE.sub("", t)
    t = re.sub(r'\s*["\'«»„“”]?\s*$', "", t)
    return t.strip()


def qc_check(orig: str, pert: Optional[str], qc_cfg: dict) -> Optional[str]:
    """Return the cleaned perturbation if it passes QC, else None."""
    if not pert:
        return None
    if qc_cfg.get("strip_wrappers", True):
        pert = _strip_wrappers(pert)
    if not pert or "\n" in pert.strip("\n") and pert.count("\n") > 1:
        return None
    pert = pert.replace("\n", " ").strip()
    if qc_cfg.get("reject_identical", True) and pert == orig.strip():
        return None
    tol = qc_cfg.get("max_char_ratio_delta", 0.30)
    if abs(len(pert) / max(1, len(orig)) - 1.0) > tol:
        return None
    return pert


# ════════════════════════════════════════════════════════════════════════════
# Rule backend — deterministic, length-preserving proxies
# ════════════════════════════════════════════════════════════════════════════
def _tokens(text: str, lang: str):
    if lang in NO_SPACE_LANGS:
        return list(text), True
    return text.split(" "), False


def _detok(toks, char_level: bool) -> str:
    return ("" if char_level else " ").join(toks)


def rule_perturb(tgt: str, lang: str, condition: str, seed_key: str) -> Optional[str]:
    """One deterministic edit per condition. Token/char count is preserved
    (substitution and reordering only — no deletions), so length never covaries
    with the injected error."""
    rng = random.Random(f"{seed_key}|{condition}")
    toks, char_level = _tokens(tgt, lang)
    n = len(toks)
    if n < 4:
        return None

    if condition == "minor_fluency":                 # adjacent swap
        i = rng.randrange(1, n - 2)
        toks[i], toks[i + 1] = toks[i + 1], toks[i]
        out = _detok(toks, char_level)
        return out if out != tgt else None

    if condition == "major_fluency":                 # shuffle a ~40% window
        w = max(3, int(round(n * 0.4)))
        i = rng.randrange(0, max(1, n - w))
        window = toks[i:i + w]
        for _ in range(10):
            rng.shuffle(window)
            if window != toks[i:i + w]:
                break
        toks[i:i + w] = window
        out = _detok(toks, char_level)
        return out if out != tgt else None

    if condition == "minor_accuracy":                # digit flip, else 1-token replace
        digit_pos = [(i, j) for i in range(n) for j, c in enumerate(toks[i])
                     if c.isdigit()]
        if digit_pos:
            i, j = rng.choice(digit_pos)
            old = toks[i][j]
            new = rng.choice([d for d in "0123456789" if d != old])
            toks[i] = toks[i][:j] + new + toks[i][j + 1:]
            return _detok(toks, char_level)
        i = rng.randrange(n)
        choices = [t for t in toks if t != toks[i]]
        if not choices:
            return None
        toks[i] = rng.choice(choices)
        out = _detok(toks, char_level)
        return out if out != tgt else None

    if condition == "major_accuracy":                # overwrite a ~25% span with
        w = max(2, int(round(n * 0.25)))             # tokens from elsewhere (length-
        if n < 2 * w + 1:                            # preserving "mistranslation")
            w = max(1, n // 3)
        i = rng.randrange(0, n - w)
        donor_idx = [j for j in range(n) if j < i or j >= i + w]
        repl = [toks[rng.choice(donor_idx)] for _ in range(w)]
        toks[i:i + w] = repl
        out = _detok(toks, char_level)
        return out if out != tgt else None

    raise ValueError(f"unknown condition {condition!r}")


# ════════════════════════════════════════════════════════════════════════════
# LLM backend (OpenAI-compatible), with a disk cache
# ════════════════════════════════════════════════════════════════════════════
class LLMPerturber:
    def __init__(self, llm_cfg: dict):
        from openai import OpenAI  # noqa: PLC0415
        kwargs = {}
        if llm_cfg.get("base_url"):
            kwargs["base_url"] = llm_cfg["base_url"]
        api_key = os.environ.get(llm_cfg.get("api_key_env", "OPENAI_API_KEY"))
        if not api_key:
            raise RuntimeError(
                f"perturb.backend=llm but ${llm_cfg.get('api_key_env')} is not set")
        self.client = OpenAI(api_key=api_key, timeout=llm_cfg.get("request_timeout", 60),
                             max_retries=llm_cfg.get("max_retries", 3), **kwargs)
        self.model = llm_cfg["model"]
        self.temperature = llm_cfg.get("temperature", 0.3)

    def __call__(self, src: str, tgt: str, lang: str, condition: str) -> Optional[str]:
        prompt = BASE_PROMPT.format(
            src_lang=LANG_NAMES.get("en", "English"),
            tgt_lang=LANG_NAMES.get(lang, lang),
            task_description=TASK_DESCRIPTIONS[condition],
            source=src, translation=tgt)
        try:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}])
            return resp.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"    LLM call failed ({condition}): {exc}")
            return None


class Perturber:
    """Unified interface with per-(lang,condition) JSON caches.

    perturb(src, tgt, lang, condition, seed_key) -> cleaned text or None.
    A None return means NO valid perturbation exists for this sentence — the
    caller must drop the whole cell, never fall back to the clean sentence.
    """

    def __init__(self, cfg: dict):
        p = cfg["perturb"]
        self.backend = p["backend"]
        self.qc = p.get("qc", {})
        self.cache_dir = Path(cfg["paths"]["perturb_cache"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._caches: Dict[str, Dict[str, Optional[str]]] = {}
        self._dirty: Dict[str, int] = {}
        self._llm = None
        if self.backend == "llm":
            self._llm = LLMPerturber(p["llm"])
            self.model_tag = p["llm"]["model"]
        else:
            self.model_tag = "rule"
        self.n_calls = self.n_cache_hits = self.n_rejected = 0

    def _cache_for(self, lang: str, condition: str) -> Dict[str, Optional[str]]:
        key = f"{lang}_{condition}"
        if key not in self._caches:
            fp = self.cache_dir / f"{key}.json"
            self._caches[key] = (json.loads(fp.read_text(encoding="utf-8"))
                                 if fp.exists() else {})
            self._dirty[key] = 0
        return self._caches[key]

    def _save(self, lang: str, condition: str) -> None:
        key = f"{lang}_{condition}"
        fp = self.cache_dir / f"{key}.json"
        fp.write_text(json.dumps(self._caches[key], ensure_ascii=False, indent=0),
                      encoding="utf-8")
        self._dirty[key] = 0

    def perturb(self, src: str, tgt: str, lang: str, condition: str,
                seed_key: str) -> Optional[str]:
        cache = self._cache_for(lang, condition)
        h = md5_short(f"{self.model_tag}|{src}|{tgt}|{condition}")
        if h in cache:
            self.n_cache_hits += 1
            raw = cache[h]
        else:
            self.n_calls += 1
            if self.backend == "llm":
                raw = self._llm(src, tgt, lang, condition)
            else:
                raw = rule_perturb(tgt, lang, condition, seed_key)
            cache[h] = raw
            self._dirty[f"{lang}_{condition}"] += 1
            if self._dirty[f"{lang}_{condition}"] >= 50:
                self._save(lang, condition)
        out = qc_check(tgt, raw, self.qc)
        if out is None:
            self.n_rejected += 1
        return out

    def flush(self) -> None:
        for key, n in list(self._dirty.items()):
            if n:
                lang, condition = key.split("_", 1)
                self._save(lang, condition)
        logger.info(f"  perturb: {self.n_calls} generated, {self.n_cache_hits} cache "
                    f"hits, {self.n_rejected} QC-rejected")
