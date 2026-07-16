#!/usr/bin/env python3
"""
block_xsim_common.py — xSIM++ on *blocks* of concatenated sentences.

The task (cf. Chen et al. 2023, "xSIM++", arXiv:2306.12907)
------------------------------------------------------------
Take k consecutive sentences of one FLORES+ article and their translations.
The query is the source block (the k source sentences, concatenated in order).
The candidate pool contains

  * every true target block in the corpus (the classic xsim distractors), and
  * for every true target block, *hard negatives* obtained by applying ONE
    xSIM++ perturbation to ONE of its k sentences, leaving the other k−1 intact.

The encoder must retrieve the block that is the concatenation of all k
translations, in order, with no perturbation. A hard negative differs from the
gold block by a single localised semantic edit inside one of k sentences — so
the task gets strictly harder as k grows and the edit is diluted. That dilution
curve is the object of interest.

Perturbation categories (the three of xSIM++ §2.2)
--------------------------------------------------
  causality  antonym substitution, negation insertion/removal, and negation
             strengthening (modal boosting, Tan et al. 2021)
  entity     replace a named-entity-like span with one sampled from a
             corpus-wide bank of the same language
  number     replace digits / ordinals with different values

Deviation from the paper: xSIM++ perturbs *English* with NLTK NER, spaCy and
WordNet. Here the perturbed side is whichever side the pool is built from —
by default the **translations** (non-English) — so the perturbations are
implemented self-contained and multilingually (no NLTK/spaCy/WordNet). Entity
detection is a casing heuristic and is therefore unavailable for languages
without case (zh, ja, th); `variant_stats` reports per-category coverage so any
gap is explicit rather than silent.

All perturbations are deterministic: the RNG is seeded from
(seed, lang, category, sentence, variant index).
"""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from representation_analysis_common import NO_SPACE_LANGS, FloresData

logger = logging.getLogger(__name__)

CATEGORIES = ("causality", "entity", "number")


# ════════════════════════════════════════════════════════════════════════════
# Blocks
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Block:
    """k consecutive rows of one article; row indices into a FloresData."""
    rows: List[int]
    url: str
    split: str = ""

    @property
    def k(self) -> int:
        return len(self.rows)


def joiner(lang: str) -> str:
    return "" if lang in NO_SPACE_LANGS else " "


def build_blocks(data: FloresData, k: int, stride: Optional[int] = None,
                 split: str = "") -> List[Block]:
    """Non-overlapping (stride=k) windows of k consecutive same-article rows.

    Overlapping windows would put near-duplicate blocks in the pool, which is a
    different (and confounded) retrieval problem — hence stride defaults to k.
    """
    stride = stride or k
    blocks: List[Block] = []
    i, n = 0, data.n
    while i < n:
        j = i
        while j < n and j - i < k and data.urls[j] == data.urls[i]:
            j += 1
        if j - i == k:
            blocks.append(Block(rows=list(range(i, j)), url=data.urls[i], split=split))
            i += stride
        else:
            i = j if j > i else i + 1
    return blocks


def block_text(data: FloresData, block: Block, lang: str) -> str:
    return joiner(lang).join(data.sentences[lang][r] for r in block.rows)


# ════════════════════════════════════════════════════════════════════════════
# Lexicons  (compact, per-language; enough for the core FLORES languages)
# ════════════════════════════════════════════════════════════════════════════
ANTONYMS: Dict[str, List[Tuple[str, str]]] = {
    "en": [("good", "bad"), ("high", "low"), ("large", "small"), ("big", "small"),
           ("more", "less"), ("most", "fewest"), ("increase", "decrease"),
           ("increased", "decreased"), ("rise", "fall"), ("rising", "falling"),
           ("positive", "negative"), ("possible", "impossible"), ("open", "closed"),
           ("early", "late"), ("first", "last"), ("strong", "weak"), ("new", "old"),
           ("young", "old"), ("long", "short"), ("fast", "slow"), ("hot", "cold"),
           ("safe", "dangerous"), ("healthy", "sick"), ("above", "below"),
           ("before", "after"), ("success", "failure"), ("win", "lose"),
           ("accept", "reject"), ("begin", "end"), ("best", "worst")],
    "de": [("gut", "schlecht"), ("hoch", "niedrig"), ("groß", "klein"),
           ("mehr", "weniger"), ("steigen", "fallen"), ("positiv", "negativ"),
           ("möglich", "unmöglich"), ("offen", "geschlossen"), ("früh", "spät"),
           ("erste", "letzte"), ("stark", "schwach"), ("neu", "alt"),
           ("jung", "alt"), ("lang", "kurz"), ("schnell", "langsam"),
           ("heiß", "kalt"), ("sicher", "gefährlich"), ("gesund", "krank"),
           ("über", "unter"), ("vor", "nach"), ("Erfolg", "Misserfolg")],
    "es": [("bueno", "malo"), ("buena", "mala"), ("alto", "bajo"), ("alta", "baja"),
           ("grande", "pequeño"), ("más", "menos"), ("aumentar", "disminuir"),
           ("positivo", "negativo"), ("posible", "imposible"),
           ("abierto", "cerrado"), ("temprano", "tarde"), ("primero", "último"),
           ("fuerte", "débil"), ("nuevo", "viejo"), ("joven", "viejo"),
           ("largo", "corto"), ("rápido", "lento"), ("caliente", "frío"),
           ("seguro", "peligroso"), ("sano", "enfermo"), ("antes", "después"),
           ("éxito", "fracaso")],
    "fr": [("bon", "mauvais"), ("bonne", "mauvaise"), ("haut", "bas"),
           ("grand", "petit"), ("plus", "moins"), ("augmenter", "diminuer"),
           ("positif", "négatif"), ("possible", "impossible"),
           ("ouvert", "fermé"), ("tôt", "tard"), ("premier", "dernier"),
           ("fort", "faible"), ("nouveau", "ancien"), ("jeune", "vieux"),
           ("long", "court"), ("rapide", "lent"), ("chaud", "froid"),
           ("sûr", "dangereux"), ("sain", "malade"), ("avant", "après"),
           ("succès", "échec")],
    "ru": [("хороший", "плохой"), ("высокий", "низкий"), ("большой", "маленький"),
           ("больше", "меньше"), ("увеличить", "уменьшить"),
           ("положительный", "отрицательный"), ("возможно", "невозможно"),
           ("открытый", "закрытый"), ("рано", "поздно"), ("первый", "последний"),
           ("сильный", "слабый"), ("новый", "старый"), ("молодой", "старый"),
           ("длинный", "короткий"), ("быстрый", "медленный"),
           ("горячий", "холодный"), ("безопасный", "опасный"),
           ("здоровый", "больной"), ("до", "после"), ("успех", "провал")],
    # Chinese pairs are all ≥2 characters on purpose: a single-character antonym
    # (大/小) matches *inside* compounds — 大学 "university" → 小学 "elementary
    # school" — which is a lexical swap, not a causality alternation.
    "zh": [("增加", "减少"), ("上升", "下降"), ("积极", "消极"),
           ("第一", "最后"), ("年轻", "年老"), ("安全", "危险"),
           ("健康", "生病"), ("成功", "失败"), ("支持", "反对"),
           ("同意", "反对"), ("提高", "降低"), ("接受", "拒绝")],
}

# Negation strengthening (Tan et al. 2021): hedged modal → assertive modal.
MODAL_BOOST: Dict[str, List[Tuple[str, str]]] = {
    "en": [("may", "will"), ("might", "will"), ("could", "will"),
           ("possibly", "certainly"), ("perhaps", "certainly"),
           ("probably", "certainly"), ("should", "must"), ("can", "must")],
    "de": [("könnte", "wird"), ("kann", "muss"), ("vielleicht", "sicherlich"),
           ("möglicherweise", "sicherlich"), ("sollte", "muss")],
    "es": [("podría", "va a"), ("puede", "debe"), ("quizás", "ciertamente"),
           ("tal vez", "ciertamente"), ("probablemente", "ciertamente")],
    "fr": [("pourrait", "va"), ("peut", "doit"), ("peut-être", "certainement"),
           ("probablement", "certainement"), ("devrait", "doit")],
    "ru": [("может", "будет"), ("возможно", "непременно"),
           ("вероятно", "непременно"), ("следует", "должен")],
    "zh": [("可能", "一定"), ("也许", "必然"), ("大概", "必然"), ("应该", "必须")],
}

# Discontinuous negation: both particles are dropped, the enclosed material
# (group 1) is kept, so the meaning flips cleanly.
NEG_REMOVE_PAIR: Dict[str, List[str]] = {
    "fr": [r"\bne\s+(.+?)\s+(?:pas|plus|jamais|rien|personne|guère)\b",
           r"\bn'(.+?)\s+(?:pas|plus|jamais|rien|personne|guère)\b"],
}
# Single-particle negation removal patterns per language.
NEG_REMOVE: Dict[str, List[str]] = {
    "en": [r"\bdid not\b", r"\bdoes not\b", r"\bdo not\b", r"\bis not\b",
           r"\bare not\b", r"\bwas not\b", r"\bwere not\b", r"\bhas not\b",
           r"\bhave not\b", r"\bcannot\b", r"\bnot\b", r"n't\b"],
    "de": [r"\bnicht\b", r"\bkein\b", r"\bkeine\b", r"\bkeinen\b"],
    "es": [r"\bno\b", r"\bnunca\b"],
    "fr": [r"\bjamais\b"],
    "ru": [r"\bне\b", r"\bнет\b"],
    "zh": [r"不", r"没有", r"未"],
}
# Insert a negation right after one of these auxiliaries.
NEG_INSERT_AFTER: Dict[str, Tuple[List[str], str]] = {
    "en": (["is", "are", "was", "were", "has", "have", "had", "can", "will",
            "would", "could", "should", "does", "did", "do"], "not"),
    "es": (["es", "son", "era", "fue", "ha", "han", "puede", "pueden"], "no"),
    "ru": (["будет", "было", "может", "могут", "есть"], "не"),
}
# Sentence-final negation insertion (before terminal punctuation).
NEG_INSERT_FINAL: Dict[str, str] = {"de": "nicht"}

STOPWORDS_CAP: Dict[str, set] = {
    "en": {"the", "a", "an", "this", "that", "these", "those", "it", "he", "she",
           "they", "we", "you", "i", "in", "on", "at", "of", "and", "but", "for",
           "to", "as", "if", "when", "while", "after", "before", "there", "his",
           "her", "their", "our", "its", "one", "two", "some", "many", "most",
           "however", "although", "according", "during", "since", "because"},
    "de": {"der", "die", "das", "ein", "eine", "einen", "dieser", "diese",
           "dieses", "er", "sie", "es", "wir", "ihr", "und", "aber", "oder",
           "in", "an", "auf", "zu", "von", "mit", "für", "als", "wenn", "nach",
           "vor", "während", "weil", "obwohl", "laut", "nachdem", "seit"},
    "es": {"el", "la", "los", "las", "un", "una", "este", "esta", "ese", "esa",
           "él", "ella", "ellos", "y", "pero", "o", "en", "de", "a", "con",
           "para", "por", "como", "si", "cuando", "después", "antes", "porque",
           "aunque", "según", "durante", "desde"},
    "fr": {"le", "la", "les", "un", "une", "ce", "cette", "ces", "il", "elle",
           "ils", "elles", "et", "mais", "ou", "en", "de", "à", "avec", "pour",
           "par", "comme", "si", "quand", "après", "avant", "parce", "bien",
           "selon", "pendant", "depuis", "dans", "sur"},
    "ru": {"этот", "эта", "это", "эти", "он", "она", "оно", "они", "мы", "вы",
           "и", "но", "или", "в", "на", "с", "для", "по", "как", "если",
           "когда", "после", "до", "потому", "хотя", "согласно", "во", "не"},
}

ORDINALS: Dict[str, List[str]] = {
    "en": ["first", "second", "third", "fourth", "fifth", "sixth", "seventh",
           "eighth", "ninth", "tenth"],
    "de": ["erste", "zweite", "dritte", "vierte", "fünfte", "sechste", "siebte",
           "achte", "neunte", "zehnte"],
    "es": ["primero", "segundo", "tercero", "cuarto", "quinto", "sexto",
           "séptimo", "octavo", "noveno", "décimo"],
    "fr": ["premier", "deuxième", "troisième", "quatrième", "cinquième",
           "sixième", "septième", "huitième", "neuvième", "dixième"],
    "ru": ["первый", "второй", "третий", "четвёртый", "пятый", "шестой",
           "седьмой", "восьмой", "девятый", "десятый"],
    "zh": ["第一", "第二", "第三", "第四", "第五", "第六", "第七", "第八",
           "第九", "第十"],
}

_DIGITS_RE = re.compile(r"\d+")
_WORD_STRIP = "\"'“”«»„().,;:!?—–-…[]{}"


def _is_upper_initial(tok: str) -> bool:
    for ch in tok:
        if ch.isalpha():
            return ch.isupper()
        if ch.isdigit():
            return False
    return False


def _match_case(new: str, old: str) -> str:
    if old[:1].isupper() and not old.isupper():
        return new[:1].upper() + new[1:]
    if old.isupper() and len(old) > 1:
        return new.upper()
    return new


def _rng(seed: int, *parts) -> random.Random:
    return random.Random(f"{seed}|" + "|".join(str(p) for p in parts))


# ════════════════════════════════════════════════════════════════════════════
# Entity bank (corpus-wide, per language)
# ════════════════════════════════════════════════════════════════════════════
def lowercase_freq(sentences: Sequence[str]) -> Dict[str, int]:
    """How often each surface form occurs *lowercased* in the corpus."""
    freq: Dict[str, int] = {}
    for s in sentences:
        for t in s.split(" "):
            core = t.strip(_WORD_STRIP)
            if core and core[:1].islower():
                freq[core] = freq.get(core, 0) + 1
    return freq


def _entity_token(tok: str, lang: str, lower_freq: Dict[str, int]) -> bool:
    """Is this token a plausible named entity?

    Capitalised, ≥2 characters, not a known capitalisable function word, and —
    the load-bearing test — *never seen lowercased in the corpus*. That last
    condition is what separates 'Stanford' from a sentence-initial 'Des'/'The',
    without needing a POS tagger or an NER model. Note German capitalises every
    noun, so for `de` the bank is noun-like rather than strictly entity-like;
    the perturbation is still a minimal, meaning-changing surface edit.
    """
    core = tok.strip(_WORD_STRIP)
    return (len(core) >= 2 and _is_upper_initial(core)
            and core.lower() not in STOPWORDS_CAP.get(lang, set())
            and lower_freq.get(core.lower(), 0) == 0)


def _spans_in(tokens: List[str], lang: str, lower_freq: Dict[str, int]
              ) -> List[Tuple[int, int]]:
    """Maximal runs of entity-like tokens."""
    spans: List[Tuple[int, int]] = []
    i = 0
    while i < len(tokens):
        if _entity_token(tokens[i], lang, lower_freq):
            j = i + 1
            while j < len(tokens) and _entity_token(tokens[j], lang, lower_freq):
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def build_entity_bank(sentences: Sequence[str], lang: str,
                      lower_freq: Dict[str, int]) -> List[str]:
    """Entity-like surfaces harvested from the whole corpus."""
    if lang in NO_SPACE_LANGS:
        return []
    bank: set = set()
    for s in sentences:
        toks = s.split(" ")
        for a, b in _spans_in(toks, lang, lower_freq):
            surface = " ".join(t.strip(_WORD_STRIP) for t in toks[a:b])
            if surface:
                bank.add(surface)
    return sorted(bank)


# ════════════════════════════════════════════════════════════════════════════
# The three xSIM++ perturbation categories
# ════════════════════════════════════════════════════════════════════════════
def _perturb_number(sent: str, lang: str, rng: random.Random) -> Optional[str]:
    digit_spans = [(m.start(), m.end(), m.group()) for m in _DIGITS_RE.finditer(sent)]
    ords = ORDINALS.get(lang, [])
    ord_spans: List[Tuple[int, int, str]] = []
    for w in ords:
        pattern = re.escape(w) if lang in NO_SPACE_LANGS else rf"\b{re.escape(w)}\b"
        for m in re.finditer(pattern, sent, flags=re.IGNORECASE):
            ord_spans.append((m.start(), m.end(), m.group()))
    choices = digit_spans + ord_spans
    if not choices:
        return None
    a, b, surface = rng.choice(choices)
    if surface.isdigit():
        n = len(surface)
        for _ in range(20):
            lo = 10 ** (n - 1) if n > 1 and surface[0] != "0" else 0
            hi = 10 ** n - 1
            cand = str(rng.randint(lo, hi)).zfill(n)
            if cand != surface:
                break
        else:
            return None
        new = cand
    else:
        pool = [w for w in ords if w.lower() != surface.lower()]
        if not pool:
            return None
        new = _match_case(rng.choice(pool), surface)
    out = sent[:a] + new + sent[b:]
    return out if out != sent else None


def _perturb_entity(sent: str, lang: str, rng: random.Random,
                    bank: Sequence[str], lower_freq: Dict[str, int]) -> Optional[str]:
    if lang in NO_SPACE_LANGS or not bank:
        return None
    toks = sent.split(" ")
    spans = _spans_in(toks, lang, lower_freq)
    if not spans:
        return None
    a, b = rng.choice(spans)
    surface = " ".join(t.strip(_WORD_STRIP) for t in toks[a:b])
    width = b - a
    same_width = [e for e in bank if e.count(" ") + 1 == width and e != surface]
    pool = same_width or [e for e in bank if e != surface]
    if not pool:
        return None
    new = rng.choice(pool)
    # keep any punctuation that trailed the original span
    tail = toks[b - 1][len(toks[b - 1].rstrip(_WORD_STRIP)):]
    lead = toks[a][:len(toks[a]) - len(toks[a].lstrip(_WORD_STRIP))]
    out = " ".join(toks[:a] + [lead + new + tail] + toks[b:])
    return out if out != sent else None


def _perturb_causality(sent: str, lang: str, rng: random.Random) -> Optional[str]:
    ops = ["antonym", "modal", "negate"]
    rng.shuffle(ops)
    for op in ops:
        out = None
        if op == "antonym":
            out = _apply_pair_map(sent, lang, ANTONYMS.get(lang, []), rng)
        elif op == "modal":
            out = _apply_pair_map(sent, lang, MODAL_BOOST.get(lang, []), rng,
                                  one_way=True)
        elif op == "negate":
            out = _negate(sent, lang, rng)
        if out and out != sent:
            return out
    return None


def _apply_pair_map(sent: str, lang: str, pairs: Sequence[Tuple[str, str]],
                    rng: random.Random, one_way: bool = False) -> Optional[str]:
    """Replace one occurrence of a lexicon word by its counterpart."""
    hits: List[Tuple[int, int, str, str]] = []
    for a, b in pairs:
        directions = [(a, b)] if one_way else [(a, b), (b, a)]
        for src, dst in directions:
            pattern = re.escape(src) if lang in NO_SPACE_LANGS else rf"\b{re.escape(src)}\b"
            for m in re.finditer(pattern, sent, flags=re.IGNORECASE):
                hits.append((m.start(), m.end(), m.group(), dst))
    if not hits:
        return None
    s, e, surface, dst = rng.choice(hits)
    return sent[:s] + _match_case(dst, surface) + sent[e:]


def _clean_spaces(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", text).strip()
    return re.sub(r"\s+([.,;:!?»])", r"\1", text)


def _negate(sent: str, lang: str, rng: random.Random) -> Optional[str]:
    # 1a. French/Spanish discontinuous negation — remove BOTH particles so the
    #     meaning actually flips (dropping only "ne" or only "pas" doesn't).
    for pair in NEG_REMOVE_PAIR.get(lang, []):
        m = re.search(pair, sent, flags=re.IGNORECASE)
        if m:
            out = _clean_spaces(sent[:m.start()] + m.group(1) + sent[m.end():])
            if out and out != sent:
                return out
    # 1b. remove a single existing negation particle
    removals = []
    for pat in NEG_REMOVE.get(lang, []):
        removals += [m.span() for m in re.finditer(pat, sent, flags=re.IGNORECASE)]
    if removals:
        s, e = rng.choice(removals)
        out = _clean_spaces(sent[:s] + sent[e:])
        if out and out != sent:
            return out
    # 2. insert one after an auxiliary
    if lang in NEG_INSERT_AFTER:
        auxes, neg = NEG_INSERT_AFTER[lang]
        toks = sent.split(" ")
        idx = [i for i, t in enumerate(toks) if t.strip(_WORD_STRIP).lower() in auxes]
        if idx:
            i = rng.choice(idx)
            toks.insert(i + 1, neg)
            return " ".join(toks)
    # 3. insert one before terminal punctuation
    if lang in NEG_INSERT_FINAL:
        neg = NEG_INSERT_FINAL[lang]
        m = re.search(r"([.!?]+)\s*$", sent)
        if m:
            return sent[:m.start()] + f" {neg}" + sent[m.start():]
        return sent + f" {neg}"
    return None


# ════════════════════════════════════════════════════════════════════════════
# Variant generation
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Perturber:
    lang: str
    seed: int = 42
    bank: List[str] = field(default_factory=list)
    lower_freq: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def for_corpus(cls, sentences: Sequence[str], lang: str, seed: int = 42) -> "Perturber":
        lf = lowercase_freq(sentences)
        return cls(lang=lang, seed=seed, lower_freq=lf,
                   bank=build_entity_bank(sentences, lang, lf))

    def variants(self, sent: str, category: str, n: int) -> List[str]:
        """Up to n distinct perturbations of `sent` in `category` (deduped,
        never identical to the input)."""
        out: List[str] = []
        for v in range(n * 6):            # oversample: many draws are no-ops/dupes
            if len(out) >= n:
                break
            rng = _rng(self.seed, self.lang, category, sent, v)
            if category == "number":
                cand = _perturb_number(sent, self.lang, rng)
            elif category == "entity":
                cand = _perturb_entity(sent, self.lang, rng, self.bank, self.lower_freq)
            elif category == "causality":
                cand = _perturb_causality(sent, self.lang, rng)
            else:
                raise ValueError(f"unknown category {category!r}")
            if cand and cand != sent and cand not in out:
                out.append(cand)
        return out


# ════════════════════════════════════════════════════════════════════════════
# Pool construction
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Candidate:
    text: str
    block_id: int            # index into the block list
    kind: str                # "true" | "perturbed"
    category: str = ""       # for kind == "perturbed"
    position: int = -1       # perturbed sentence index within the block


def build_pool(data_by_split: Sequence[Tuple[str, FloresData, List[Block]]],
               pool_lang: str, categories: Sequence[str],
               variants_per_position: int, perturber: Perturber,
               ) -> Tuple[List[Candidate], List[int], Dict[str, int]]:
    """Returns (candidates, true_index_per_block, variant_stats).

    `candidates[i].block_id` indexes the *global* block list (splits concatenated
    in the order given). `true_idx[b]` is the pool position of block b's gold
    candidate.
    """
    cands: List[Candidate] = []
    true_idx: List[int] = []
    stats: Dict[str, int] = {c: 0 for c in categories}
    stats["blocks"] = 0
    bid = 0
    jn = joiner(pool_lang)
    for _split, data, blocks in data_by_split:
        sents = data.sentences[pool_lang]
        for blk in blocks:
            parts = [sents[r] for r in blk.rows]
            true_idx.append(len(cands))
            cands.append(Candidate(text=jn.join(parts), block_id=bid, kind="true"))
            for pos in range(blk.k):
                for cat in categories:
                    for v in perturber.variants(parts[pos], cat, variants_per_position):
                        newp = list(parts)
                        newp[pos] = v
                        cands.append(Candidate(text=jn.join(newp), block_id=bid,
                                               kind="perturbed", category=cat,
                                               position=pos))
                        stats[cat] += 1
            stats["blocks"] += 1
            bid += 1
    return cands, true_idx, stats


# ════════════════════════════════════════════════════════════════════════════
# Retrieval + metrics  (absolute margin = plain cosine, per xSIM++ footnote 6)
# ════════════════════════════════════════════════════════════════════════════
def _err_on_subset(sim: np.ndarray, cols: np.ndarray,
                   true_idx: np.ndarray) -> Tuple[float, np.ndarray]:
    """Error rate when retrieving over `cols` (pool positions). Returns
    (error_rate, predicted pool positions)."""
    pred = cols[sim[:, cols].argmax(1)]
    err = float((pred != true_idx).mean())
    return err, pred


def evaluate_blocks(q_emb: np.ndarray, p_emb: np.ndarray,
                    cands: List[Candidate], true_idx: List[int],
                    categories: Sequence[str]) -> Dict:
    """All xsim / xsim++ numbers for one (encoder, lang, k) cell."""
    sim = q_emb @ p_emb.T                                  # (N_blocks, M_cands)
    n = sim.shape[0]
    true_arr = np.asarray(true_idx)
    cand_block = np.asarray([c.block_id for c in cands])
    cand_kind = np.asarray([c.kind for c in cands])
    cand_cat = np.asarray([c.category for c in cands])
    cand_pos = np.asarray([c.position for c in cands])

    true_cols = np.asarray(true_idx)
    all_cols = np.arange(len(cands))

    xsim_err, _ = _err_on_subset(sim, true_cols, true_arr)
    xsimpp_err, pred_all = _err_on_subset(sim, all_cols, true_arr)

    # error typology on the full pool
    correct = pred_all == true_arr
    pred_block = cand_block[pred_all]
    own_perturbed = (~correct) & (pred_block == np.arange(n))
    misaligned = (~correct) & (pred_block != np.arange(n))
    breakdown = {"correct": float(correct.mean()),
                 "misaligned": float(misaligned.mean())}
    for c in categories:
        breakdown[c] = float((own_perturbed & (cand_cat[pred_all] == c)).mean())

    # per-category pools (paper Table 4 rows) and every category combination
    def _pool_err(cats: Sequence[str]) -> float:
        mask = (cand_kind == "true") | np.isin(cand_cat, list(cats))
        return _err_on_subset(sim, all_cols[mask], true_arr)[0]

    per_category = {c: _pool_err([c]) for c in categories}
    combos: Dict[str, float] = {}
    from itertools import combinations
    for r in range(2, len(categories) + 1):
        for combo in combinations(categories, r):
            combos["+".join(combo)] = _pool_err(combo)

    # push-apart detection: P[cos(q, gold) > cos(q, negative)] over own negatives
    gold_sim = sim[np.arange(n), true_arr]
    det_by_cat: Dict[str, Optional[float]] = {}
    det_by_pos: Dict[str, Optional[float]] = {}
    cov: Dict[str, int] = {}
    own = cand_block[None, :] == np.arange(n)[:, None]       # (N, M)
    is_pert = cand_kind == "perturbed"
    for c in categories:
        m = own & is_pert[None, :] & (cand_cat == c)[None, :]
        cov[c] = int(m.sum())
        det_by_cat[c] = float((gold_sim[:, None] > sim)[m].mean()) if m.any() else None
    for pos in sorted({int(p) for p in cand_pos if p >= 0}):
        m = own & is_pert[None, :] & (cand_pos == pos)[None, :]
        det_by_pos[str(pos)] = float((gold_sim[:, None] > sim)[m].mean()) if m.any() else None
    m_all = own & is_pert[None, :]
    detection = float((gold_sim[:, None] > sim)[m_all].mean()) if m_all.any() else None

    # margin between the gold block and the single best negative
    neg = sim.copy()
    neg[np.arange(n), true_arr] = -np.inf
    margin = float((gold_sim - neg.max(1)).mean())
    # margin against the *hardest own perturbation* only (isolates dilution)
    own_neg = np.where(own & is_pert[None, :], sim, -np.inf)
    has_own = np.isfinite(own_neg.max(1))
    margin_own = (float((gold_sim[has_own] - own_neg.max(1)[has_own]).mean())
                  if has_own.any() else None)

    return {"n_blocks": n, "pool_size": len(cands),
            "xsim_err": xsim_err, "xsimpp_err": xsimpp_err,
            "error_breakdown": breakdown,
            "per_category_err": per_category, "category_combos": combos,
            "detection_rate": detection, "detection_by_category": det_by_cat,
            "detection_by_position": det_by_pos,
            "n_negatives_by_category": cov,
            "margin_vs_best_negative": margin,
            "margin_vs_best_own_perturbation": margin_own}
