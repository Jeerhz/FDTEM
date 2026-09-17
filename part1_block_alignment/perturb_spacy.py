"""The xSIM++ perturbations driven by spaCy (+ optional WordNet) — `--backend spacy`.

The heuristic `perturb.Perturber` (a) sees no entities in uncased scripts, so
zh/ja/th get zero entity negatives, (b) misses spelled-out numerals and
(c) guesses where a negation particle attaches. spaCy fixes those three:

  entity     `doc.ents` — real, typed NER: a LOC is swapped for a LOC.
  number     `like_num` / `pos_ == NUM` / `NumType=Ord` — digits and words.
  causality  `Polarity=Neg` / `dep_ in {neg, ng}` gives the negation particles
             and their verb, so French "ne … pas" is removed as a unit and an
             inserted negation lands on the finite verb.

Antonyms and modal boosting stay lexical (`perturb.ANTONYMS` / `MODAL_BOOST`);
WordNet is a per-language opt-in because Open Multilingual WordNet has no de/ru
and is sense-ambiguous elsewhere.

    pip install spacy nltk
    python -m spacy download de_core_news_sm   # es/fr/ru/zh/en likewise
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-2.0')"

Determinism is unchanged: the RNG is seeded from (seed, lang, category, sentence, variant).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from common.flores import joiner
from part1_block_alignment.perturb import (ANTONYMS, MODAL_BOOST, ORDINALS, _apply_pair_map,
                                           _clean_spaces, _match_case, _rng)

logger = logging.getLogger(__name__)

# Small models are enough: NER + POS + morph, and they are ~15 MB each.
SPACY_MODELS: Dict[str, str] = {
    "en": "en_core_web_sm", "de": "de_core_news_sm", "es": "es_core_news_sm",
    "fr": "fr_core_news_sm", "ru": "ru_core_news_sm", "zh": "zh_core_web_sm",
    "ja": "ja_core_news_sm", "it": "it_core_news_sm", "pt": "pt_core_news_sm",
    "nl": "nl_core_news_sm", "pl": "pl_core_news_sm", "el": "el_core_news_sm",
    "ro": "ro_core_news_sm", "da": "da_core_news_sm", "sv": "sv_core_news_sm",
    "nb": "nb_core_news_sm", "fi": "fi_core_news_sm", "lt": "lt_core_news_sm",
    "uk": "uk_core_news_sm", "hr": "hr_core_news_sm", "ca": "ca_core_news_sm",
    "ko": "ko_core_news_sm", "mk": "mk_core_news_sm", "sl": "sl_core_news_sm",
}

# Entity labels worth swapping. DATE/CARDINAL/PERCENT/QUANTITY/ORDINAL/MONEY are
# deliberately excluded — those belong to the `number` category, and letting both
# categories edit the same span would make the per-category comparison mush.
ENTITY_LABELS = {"PER", "PERSON", "LOC", "GPE", "ORG", "FAC", "NORP", "EVENT",
                 "PRODUCT", "WORK_OF_ART", "LANGUAGE", "MISC"}
NUMBER_LABELS = {"CARDINAL", "ORDINAL", "PERCENT", "QUANTITY", "MONEY", "DATE", "TIME"}

# Negation particles, to catch what the morphologiser misses. Split in two:
#   CORE    unambiguously negative on their own — safe to trigger on
#   PAIRED  only negative when they complete a CORE particle (French "ne … plus":
#           bare "plus" means *more*, so it is dropped only alongside its "ne")
NEG_LEMMAS_CORE: Dict[str, Set[str]] = {
    "en": {"not", "n't", "never"},
    "de": {"nicht", "kein", "nie", "niemals"},
    "es": {"no", "nunca", "jamás"},
    "fr": {"ne"},
    "ru": {"не", "нет", "никогда"},
    "zh": {"不", "没", "没有", "未", "非"},
}
NEG_LEMMAS_PAIRED: Dict[str, Set[str]] = {
    "fr": {"pas", "plus", "jamais", "rien", "personne", "guère", "aucun", "nul"},
    "en": {"no", "any"},
}
# How to insert a negation once spaCy has pointed at the finite verb:
# (particle before it, particle after it). German is V2 — "nicht" goes *after*
# the finite verb ("wurde nicht veröffentlicht"), not before it.
NEG_INSERT: Dict[str, Tuple[str, str]] = {
    "en": ("", "not"), "de": ("", "nicht"), "es": ("no", ""),
    "fr": ("ne", "pas"), "ru": ("не", ""), "zh": ("不", ""),
}
# English has no bare-verb negation ("said not …"), so only negate an auxiliary.
NEG_AUX_ONLY = {"en"}
_VOWELS = "aeiouyáàâéèêíìîóòôúùûhAEIOUY"

# UPOS → WordNet POS.
_UPOS2WN = {"ADJ": ("a", "s"), "ADV": ("r",), "VERB": ("v",), "NOUN": ("n",)}
# ISO-639-1 → the 3-letter code NLTK's OMW uses.
OMW_CODE = {"en": "eng", "es": "spa", "fr": "fra", "it": "ita", "pt": "por",
            "nl": "nld", "pl": "pol", "el": "ell", "ca": "cat", "da": "dan",
            "fi": "fin", "sv": "swe", "th": "tha", "id": "ind", "ja": "jpn",
            "zh": "cmn", "he": "heb", "bg": "bul", "hr": "hrv", "sl": "slv"}


# ════════════════════════════════════════════════════════════════════════════
# Loading
# ════════════════════════════════════════════════════════════════════════════
def load_pipeline(lang: str):
    """spaCy pipeline for `lang`, with an actionable error if it is missing."""
    import spacy
    name = SPACY_MODELS.get(lang)
    if name is None:
        raise RuntimeError(f"no spaCy model registered for {lang!r}; add one to "
                           "SPACY_MODELS or use --backend heuristic")
    try:
        return spacy.load(name)
    except OSError as exc:                                     # model not installed
        raise RuntimeError(f"spaCy model {name!r} is not installed — run "
                           f"`python -m spacy download {name}`") from exc


def _wordnet():
    from nltk.corpus import wordnet as wn
    wn.synsets("test")                                          # force the lookup
    return wn


# ════════════════════════════════════════════════════════════════════════════
# Rendering: rebuild a sentence from an edited token sequence
# ════════════════════════════════════════════════════════════════════════════
def _render(doc, replace: Dict[int, str], drop: Set[int],
            before: Dict[int, str] = None, after: Dict[int, str] = None) -> str:
    """Detokenise `doc` with edits applied. `whitespace_` keeps spacing right —
    and keeps it *absent* for zh/ja, which is why this is not `" ".join`.
    `before`/`after` are spliced verbatim, separator included, so the caller
    controls elision ("n'a" vs "ne a")."""
    before, after = before or {}, after or {}
    parts: List[str] = []
    for t in doc:
        if t.i in before:
            parts.append(before[t.i])
        if t.i not in drop:
            parts.append(replace.get(t.i, t.text))
            if t.i in after:
                parts.append(after[t.i])
            parts.append(t.whitespace_)
    return _clean_spaces("".join(parts))


# ════════════════════════════════════════════════════════════════════════════
# The perturber
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class SpacyPerturber:
    """Same `variants()` interface as `perturb.Perturber`."""
    lang: str
    seed: int = 42
    nlp: object = None
    docs: Dict[str, object] = field(default_factory=dict)
    ent_bank: Dict[str, List[str]] = field(default_factory=dict)
    card_bank: List[str] = field(default_factory=list)
    ord_bank: List[str] = field(default_factory=list)
    wordnet_langs: Tuple[str, ...] = ("en",)
    max_senses: int = 2
    _wn: object = None

    @property
    def _ordinals(self) -> Set[str]:
        return {w.lower() for w in ORDINALS.get(self.lang, [])}

    @property
    def _sep(self) -> str:
        """" " for space-delimited languages, "" for zh/ja/th."""
        return joiner(self.lang)

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def for_corpus(cls, sentences: Sequence[str], lang: str, seed: int = 42,
                   wordnet_langs: Sequence[str] = ("en",),
                   batch_size: int = 128) -> "SpacyPerturber":
        nlp = load_pipeline(lang)
        uniq = list(dict.fromkeys(sentences))
        docs = dict(zip(uniq, nlp.pipe(uniq, batch_size=batch_size)))
        self = cls(lang=lang, seed=seed, nlp=nlp, docs=docs,
                   wordnet_langs=tuple(wordnet_langs))
        self._harvest()
        if lang in self.wordnet_langs:
            code = OMW_CODE.get(lang)
            if code is None:
                logger.warning(f"  [{lang}] no Open Multilingual WordNet code — "
                               "antonyms fall back to the curated lexicon")
            else:
                try:
                    wn = _wordnet()
                    wn.synsets("test", lang=code)               # probe coverage
                    self._wn = wn
                except Exception as exc:                        # noqa: BLE001
                    logger.warning(f"  [{lang}] WordNet unusable ({exc.__class__.__name__}) "
                                   "— antonyms fall back to the curated lexicon")
        return self

    def _harvest(self) -> None:
        """Corpus-wide entity and numeral banks, from the parses."""
        ents: Dict[str, Set[str]] = {}
        cards: Set[str] = set()
        ords_: Set[str] = set()
        for doc in self.docs.values():
            for e in doc.ents:
                if e.label_ in ENTITY_LABELS:
                    ents.setdefault(e.label_, set()).add(e.text)
            for t in doc:
                if any(ch.isdigit() for ch in t.text):
                    continue
                if self._is_ordinal(t):
                    ords_.add(t.text)
                elif t.like_num and t.pos_ == "NUM":
                    cards.add(t.text)
        self.ent_bank = {k: sorted(v) for k, v in ents.items()}
        self.card_bank = sorted(cards)
        # not every model sets NumType=Ord (the German one does not), so the
        # curated list seeds the bank as well
        self.ord_bank = sorted(ords_ | set(ORDINALS.get(self.lang, [])))

    def _is_ordinal(self, tok) -> bool:
        return ("Ord" in tok.morph.get("NumType")
                or tok.text.lower() in self._ordinals)

    @property
    def bank(self) -> List[str]:
        """Flat entity list — same meaning as `perturb.Perturber.bank`."""
        return sorted({e for v in self.ent_bank.values() for e in v})

    def coverage(self) -> Dict[str, int]:
        return {"sentences": len(self.docs), "entities": len(self.bank),
                "entity_labels": len(self.ent_bank),
                "cardinal_words": len(self.card_bank),
                "ordinal_words": len(self.ord_bank),
                "wordnet": int(self._wn is not None)}

    def _doc(self, sent: str):
        d = self.docs.get(sent)
        if d is None:
            d = self.nlp(sent)
            self.docs[sent] = d
        return d

    # ── the public API, identical to perturb.Perturber ───────────────────────
    def variants(self, sent: str, category: str, n: int) -> List[str]:
        out: List[str] = []
        for v in range(n * 6):
            if len(out) >= n:
                break
            rng = _rng(self.seed, self.lang, category, sent, v)
            doc = self._doc(sent)
            if category == "number":
                cand = self._number(doc, rng)
            elif category == "entity":
                cand = self._entity(doc, rng)
            elif category == "causality":
                cand = self._causality(doc, sent, rng)
            else:
                raise ValueError(f"unknown category {category!r}")
            if cand and cand != sent and cand not in out:
                out.append(cand)
        return out

    # ── entity ──────────────────────────────────────────────────────────────
    def _entity(self, doc, rng) -> Optional[str]:
        # French/Italian elision: "l'université Stanford" → "l'Berlin" would be
        # ungrammatical, so a span behind an apostrophe is left alone.
        spans = [e for e in doc.ents if e.label_ in ENTITY_LABELS
                 and not (e.start and doc[e.start - 1].text.endswith("'"))]
        if not spans or not self.ent_bank:
            return None
        span = rng.choice(spans)
        pool = [e for e in self.ent_bank.get(span.label_, []) if e != span.text]
        if not pool:                                   # no same-type entity: any type
            pool = [e for e in self.bank if e != span.text]
        if not pool:
            return None
        new = rng.choice(pool)
        return _render(doc, {span.start: new},
                       set(range(span.start + 1, span.end)))

    # ── number ──────────────────────────────────────────────────────────────
    def _number(self, doc, rng) -> Optional[str]:
        digits, words = [], []
        for t in doc:
            if any(ch.isdigit() for ch in t.text):
                digits.append(t)
            elif (t.pos_ == "NUM" and t.like_num) or self._is_ordinal(t):
                words.append(t)
        choices = digits + words
        if not choices:
            return None
        t = rng.choice(choices)
        if t in digits:
            new = _scramble_digits(t.text, rng)
        else:
            bank = (self.ord_bank if self._is_ordinal(t) else self.card_bank) or self.card_bank
            pool = [w for w in bank if w.lower() != t.text.lower()]
            new = _match_case(rng.choice(pool), t.text) if pool else None
        if not new:
            return None
        return _render(doc, {t.i: new}, set())

    # ── causality ───────────────────────────────────────────────────────────
    def _causality(self, doc, sent: str, rng) -> Optional[str]:
        ops = ["antonym", "modal", "negate"]
        rng.shuffle(ops)
        for op in ops:
            if op == "antonym":
                out = (_apply_pair_map(sent, self.lang, ANTONYMS.get(self.lang, []), rng)
                       or self._wordnet_antonym(doc, rng))
            elif op == "modal":
                out = _apply_pair_map(sent, self.lang, MODAL_BOOST.get(self.lang, []),
                                      rng, one_way=True)
            else:
                out = self._negate(doc, rng)
            if out and out != sent:
                return out
        return None

    def _negation_tokens(self, doc) -> List[object]:
        core = NEG_LEMMAS_CORE.get(self.lang, set())
        return [t for t in doc
                if "Neg" in t.morph.get("Polarity") or t.dep_ in ("neg", "ng")
                or t.lemma_.lower() in core or t.text.lower() in core]

    def _negate(self, doc, rng) -> Optional[str]:
        negs = self._negation_tokens(doc)
        if negs:
            # Drop every particle attached to one head, so French "ne … pas"
            # goes as a unit — dropping half of it does not flip anything.
            head = rng.choice(negs).head
            paired = NEG_LEMMAS_PAIRED.get(self.lang, set())
            drop = {t.i for t in negs if t.head == head} or {rng.choice(negs).i}
            drop |= {t.i for t in doc
                     if t.head == head and t.lemma_.lower() in paired}
            return _render(doc, {}, drop)
        pre, post = NEG_INSERT.get(self.lang, ("", ""))
        if not (pre or post):
            return None
        verbs = [t for t in doc if t.pos_ in ("VERB", "AUX")
                 and (not t.morph.get("VerbForm") or "Fin" in t.morph.get("VerbForm"))]
        verbs = verbs or [t for t in doc if t.pos_ in ("VERB", "AUX")]
        if self.lang in NEG_AUX_ONLY:
            verbs = [t for t in verbs if t.pos_ == "AUX"]
        if not verbs:
            return None
        v = rng.choice(verbs)
        # Romance clitics sit between the negation and the verb: "no se anunció",
        # not "se no anunció" — so anchor on the leftmost clitic of this verb.
        anchor = v.i
        while (anchor > 0 and doc[anchor - 1].pos_ == "PRON"
               and doc[anchor - 1].head.i == v.i):
            anchor -= 1
        sep = self._sep
        if self.lang == "fr" and pre == "ne" and doc[anchor].text[:1] in _VOWELS:
            pre, sep = "n'", ""                                 # elision
        return _render(doc, {}, set(),
                       {anchor: pre + sep} if pre else None,
                       {v.i: self._sep + post} if post else None)

    def _wordnet_antonym(self, doc, rng) -> Optional[str]:
        """WordNet antonym of one token — narrow on purpose (see module docstring)."""
        if self._wn is None:
            return None
        code = OMW_CODE[self.lang]
        cands: List[Tuple[int, str]] = []
        for t in doc:
            wn_pos = _UPOS2WN.get(t.pos_)
            # only substitute where the surface *is* the lemma: swapping a lemma
            # into an inflected slot yields an ungrammatical negative
            if not wn_pos or t.text.lower() != t.lemma_.lower():
                continue
            for a in _antonyms(self._wn, t.lemma_.lower(), code, wn_pos,
                               self.max_senses):
                if " " in a and self.lang != "en":              # multi-word lemma
                    continue
                if a.lower() != t.text.lower():
                    cands.append((t.i, a))
        if not cands:
            return None
        i, new = rng.choice(cands)
        return _render(doc, {i: _match_case(new, doc[i].text)}, set())


def _antonyms(wn, lemma: str, code: str, wn_pos: Tuple[str, ...],
              max_senses: int) -> List[str]:
    out: List[str] = []
    for pos in wn_pos:
        try:
            synsets = wn.synsets(lemma, pos=pos, lang=code)[:max_senses]
        except Exception:                                      # noqa: BLE001
            return out
        for ss in synsets:
            for lem in ss.lemmas():
                for ant in lem.antonyms():
                    names = (ant.synset().lemma_names(code) if code != "eng"
                             else [ant.name()])
                    for nm in names:
                        nm = nm.replace("_", " ")
                        if nm not in out:
                            out.append(nm)
    return out


def _scramble_digits(surface: str, rng) -> Optional[str]:
    """Replace the digit run in `surface` with a different one of equal width."""
    import re
    m = re.search(r"\d+", surface)
    if not m:
        return None
    old = m.group()
    n = len(old)
    lo = 10 ** (n - 1) if n > 1 and old[0] != "0" else 0
    for _ in range(20):
        cand = str(rng.randint(lo, 10 ** n - 1)).zfill(n)
        if cand != old:
            return surface[:m.start()] + cand + surface[m.end():]
    return None
