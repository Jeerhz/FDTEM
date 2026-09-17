"""xSIM++ hard negatives on blocks: one edit in one sentence, the other k-1 intact.

    python -m part1_block_alignment.perturb --source plus --langs de es fr ru --k_list 2 3 4 5 --backend spacy
    python -m part1_block_alignment.perturb --langs de --k_list 3 --dry_run     # coverage + examples, CPU

For every (lang, k) one CandidatePool is written to data/pools_<backend>_<lang>_k<k>.json:
every block's gold translation plus, per (sentence position, category), up to
`--variants_per_position` perturbed copies of the block.

Categories (xSIM++ §2.2, Chen et al. 2023): `causality` (antonym, negation,
modal boosting), `entity` (swap a named-entity-like span for one from a
corpus-wide bank), `number` (digits / ordinals). The heuristic backend here is
self-contained and multilingual; entity detection is a casing heuristic, so it
yields no entity negatives for zh/ja/th. `--backend spacy` (perturb_spacy.py)
uses real NER, morphology and the parse instead and reuses the lexicons below.
The two backends generate different negatives: never compare runs across them.

Every perturbation is deterministic: the RNG is seeded from
(seed, lang, category, sentence, variant index).
"""
from __future__ import annotations

import argparse
import logging
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from common.flores import NO_SPACE_LANGS, FloresCorpus, joiner
from part1_block_alignment import DATA_DIR
from part1_block_alignment.build_blocks import block_text, load_blocks
from part1_block_alignment.load_flores import load_corpus
from part1_block_alignment.models import CATEGORIES, Block, Candidate, CandidatePool

logger = logging.getLogger(__name__)


# ── lexicons (compact, per language) ─────────────────────────────────────────
ANTONYMS: dict[str, list[tuple[str, str]]] = {
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
MODAL_BOOST: dict[str, list[tuple[str, str]]] = {
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
NEG_REMOVE_PAIR: dict[str, list[str]] = {
    "fr": [r"\bne\s+(.+?)\s+(?:pas|plus|jamais|rien|personne|guère)\b",
           r"\bn'(.+?)\s+(?:pas|plus|jamais|rien|personne|guère)\b"],
}
# Single-particle negation removal patterns per language.
NEG_REMOVE: dict[str, list[str]] = {
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
NEG_INSERT_AFTER: dict[str, tuple[list[str], str]] = {
    "en": (["is", "are", "was", "were", "has", "have", "had", "can", "will",
            "would", "could", "should", "does", "did", "do"], "not"),
    "es": (["es", "son", "era", "fue", "ha", "han", "puede", "pueden"], "no"),
    "ru": (["будет", "было", "может", "могут", "есть"], "не"),
}
# Sentence-final negation insertion (before terminal punctuation).
NEG_INSERT_FINAL: dict[str, str] = {"de": "nicht"}

STOPWORDS_CAP: dict[str, set] = {
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

ORDINALS: dict[str, list[str]] = {
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


# ── entity bank (corpus-wide, per language) ───────────────────────────────────
def lowercase_freq(sentences: Sequence[str]) -> dict[str, int]:
    """How often each surface form occurs *lowercased* in the corpus."""
    freq: dict[str, int] = {}
    for s in sentences:
        for t in s.split(" "):
            core = t.strip(_WORD_STRIP)
            if core and core[:1].islower():
                freq[core] = freq.get(core, 0) + 1
    return freq


def _entity_token(tok: str, lang: str, lower_freq: dict[str, int]) -> bool:
    """Capitalised, ≥2 characters, not a capitalisable function word, and — the
    load-bearing test — never seen lowercased in the corpus. That separates
    'Stanford' from a sentence-initial 'Des'/'The' without a tagger. German
    capitalises every noun, so for `de` the bank is noun-like rather than
    strictly entity-like."""
    core = tok.strip(_WORD_STRIP)
    return (len(core) >= 2 and _is_upper_initial(core)
            and core.lower() not in STOPWORDS_CAP.get(lang, set())
            and lower_freq.get(core.lower(), 0) == 0)


def _spans_in(tokens: list[str], lang: str, lower_freq: dict[str, int]
              ) -> list[tuple[int, int]]:
    """Maximal runs of entity-like tokens."""
    spans: list[tuple[int, int]] = []
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
                      lower_freq: dict[str, int]) -> list[str]:
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


# ── the three xSIM++ perturbation categories ──────────────────────────────────
def _perturb_number(sent: str, lang: str, rng: random.Random) -> str | None:
    digit_spans = [(m.start(), m.end(), m.group()) for m in _DIGITS_RE.finditer(sent)]
    ords = ORDINALS.get(lang, [])
    ord_spans: list[tuple[int, int, str]] = []
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
                    bank: Sequence[str], lower_freq: dict[str, int]) -> str | None:
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


def _perturb_causality(sent: str, lang: str, rng: random.Random) -> str | None:
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


def _apply_pair_map(sent: str, lang: str, pairs: Sequence[tuple[str, str]],
                    rng: random.Random, one_way: bool = False) -> str | None:
    """Replace one occurrence of a lexicon word by its counterpart."""
    hits: list[tuple[int, int, str, str]] = []
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


def _negate(sent: str, lang: str, rng: random.Random) -> str | None:
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


# ── variant generation ────────────────────────────────────────────────────────
@dataclass
class Perturber:
    """The heuristic backend. `variants()` is the interface both backends share."""

    lang: str
    seed: int = 42
    bank: list[str] = field(default_factory=list)
    lower_freq: dict[str, int] = field(default_factory=dict)

    @classmethod
    def for_corpus(cls, sentences: Sequence[str], lang: str, seed: int = 42) -> "Perturber":
        lf = lowercase_freq(sentences)
        return cls(lang=lang, seed=seed, lower_freq=lf,
                   bank=build_entity_bank(sentences, lang, lf))

    def variants(self, sent: str, category: str, n: int) -> list[str]:
        """Up to n distinct perturbations of `sent` in `category` (deduped,
        never identical to the input)."""
        out: list[str] = []
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


def build_perturber(sentences: Sequence[str], lang: str, seed: int = 42,
                    backend: str = "heuristic", wordnet_langs: Sequence[str] = ("en",)):
    """`backend` in {heuristic, spacy, auto}; `auto` prefers spaCy and falls back."""
    if backend == "heuristic":
        return Perturber.for_corpus(sentences, lang, seed)
    from part1_block_alignment.perturb_spacy import SpacyPerturber
    try:
        p = SpacyPerturber.for_corpus(sentences, lang, seed, wordnet_langs)
        logger.info(f"  [{lang}] spaCy backend: {p.coverage()}")
        return p
    except Exception as exc:  # noqa: BLE001
        if backend == "spacy":
            raise
        logger.warning(f"  [{lang}] spaCy backend unavailable ({exc}) — "
                       "falling back to the heuristic perturber")
        return Perturber.for_corpus(sentences, lang, seed)


def backend_name(perturber) -> str:
    return "heuristic" if isinstance(perturber, Perturber) else "spacy"


# ── pool construction ─────────────────────────────────────────────────────────
def build_pool(per_split: Sequence[tuple[FloresCorpus, list[Block]]], lang: str,
               query_lang: str, pool_lang: str, k: int, categories: Sequence[str],
               variants_per_position: int, perturber, seed: int) -> CandidatePool:
    """Gold + single-edit negatives for every block, splits concatenated in order.

    `candidates[i].block_id` indexes the global block list; `true_index[b]` is
    the pool position of block b's gold candidate.
    """
    queries: list[str] = []
    cands: list[Candidate] = []
    true_idx: list[int] = []
    stats: dict[str, int] = {c: 0 for c in categories}
    stats["blocks"] = 0
    bid = 0
    jn = joiner(pool_lang)
    for corpus, blocks in per_split:
        sents = corpus.sentences[pool_lang]
        for blk in blocks:
            queries.append(block_text(corpus, blk, query_lang))
            parts = [sents[r] for r in blk.rows]
            true_idx.append(len(cands))
            cands.append(Candidate(text=jn.join(parts), block_id=bid, kind="true"))
            for pos in range(blk.k):
                for cat in categories:
                    for v, text in enumerate(perturber.variants(parts[pos], cat,
                                                                variants_per_position)):
                        newp = list(parts)
                        newp[pos] = text
                        cands.append(Candidate(text=jn.join(newp), block_id=bid,
                                               kind="perturbed", category=cat,
                                               position=pos, variant=v))
                        stats[cat] += 1
            stats["blocks"] += 1
            bid += 1
    return CandidatePool(lang=lang, query_lang=query_lang, pool_lang=pool_lang, k=k,
                         backend=backend_name(perturber), seed=seed,
                         splits=[c.split for c, _ in per_split], queries=queries,
                         candidates=cands, true_index=true_idx, variant_counts=stats)


def pool_path(backend: str, lang: str, k: int) -> Path:
    return DATA_DIR / f"pools_{backend}_{lang}_k{k}.json"


def load_pool(backend: str, lang: str, k: int) -> CandidatePool:
    path = pool_path(backend, lang, k)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found — run `python -m part1_block_alignment.perturb "
            f"--backend {backend} --langs {lang} --k_list {k}` first")
    return CandidatePool.load(path)


def pool_categories(pool: CandidatePool) -> list[str]:
    return [c for c in CATEGORIES if c in pool.variant_counts]


# ── CLI ───────────────────────────────────────────────────────────────────────
def _print_pool(pool: CandidatePool, duel_sizes=(6, 5, 4)) -> None:
    """Coverage of one pool: variants per block per category, duel eligibility."""
    n = pool.n_blocks
    per_blk = {c: round(pool.variant_counts[c] / max(1, n), 2) for c in pool_categories(pool)}
    ms = [len(d.negatives) for d in pool.duel_items()]
    cov = "  ".join(f">={D}: {sum(1 for m in ms if m >= D)} "
                    f"({sum(1 for m in ms if m >= D) / max(1, n):.0%})" for D in duel_sizes)
    hist = {m: ms.count(m) for m in sorted(set(ms))}
    logger.info(f"  k={pool.k}: {n} blocks, pool={len(pool.candidates)}, "
                f"variants/block={per_blk}  duel {cov}  m-histogram={hist}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["plus", "raw"], default="plus")
    ap.add_argument("--splits", nargs="+", default=["dev", "devtest"],
                    help="Splits pooled into ONE pool per (lang, k); k=5 blocks are scarce in one split.")
    ap.add_argument("--langs", nargs="+", default=["de", "es", "fr", "ru"],
                    help="Non-pivot languages. The heuristic backend needs letter case "
                         "for entities (no zh/ja/th); the spacy backend lifts that.")
    ap.add_argument("--pivot", default="en")
    ap.add_argument("--direction", choices=["en2xx", "xx2en"], default="en2xx",
                    help="en2xx: query=source block, pool=translations (perturbed). "
                         "xx2en: the original xSIM++ direction (pool=English).")
    ap.add_argument("--k_list", nargs="+", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=list(CATEGORIES))
    ap.add_argument("--backend", choices=["spacy", "heuristic", "auto"], default="spacy")
    ap.add_argument("--variants_per_position", type=int, default=2,
                    help="Hard negatives per (block, sentence position, category).")
    ap.add_argument("--wordnet_langs", nargs="*", default=["en"],
                    help="Languages whose antonyms may also come from WordNet (spacy only).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_blocks", type=int, default=None,
                    help="Truncate the block list per split (smoke tests).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print coverage and example negatives, write nothing.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    corpora = [load_corpus(args.source, sp) for sp in args.splits]
    for lang in args.langs:
        if lang == args.pivot:
            continue
        query_lang = args.pivot if args.direction == "en2xx" else lang
        pool_lang = lang if args.direction == "en2xx" else args.pivot
        sentences = [s for c in corpora for s in c.sentences[pool_lang]]
        pert = build_perturber(sentences, pool_lang, args.seed, args.backend, args.wordnet_langs)
        if (backend_name(pert) == "heuristic" and pool_lang in NO_SPACE_LANGS
                and "entity" in args.categories):
            logger.warning(f"  [{pool_lang}] entity perturbation needs letter case — "
                           "no entity negatives will be generated for this language.")
        logger.info(f"\n== {pool_lang} ({backend_name(pert)}, entity bank: {len(pert.bank)}) ==")
        for k in args.k_list:
            per_split = [(c, load_blocks(args.source, c.split, k)[:args.max_blocks]) for c in corpora]
            pool = build_pool(per_split, lang, query_lang, pool_lang, k, args.categories,
                              args.variants_per_position, pert, args.seed)
            _print_pool(pool)
            if pool.n_blocks < 2:
                logger.warning(f"  [{lang}] k={k}: {pool.n_blocks} blocks — not written")
            elif not args.dry_run:
                pool.save(pool_path(pool.backend, lang, k))
                logger.info(f"  -> {pool_path(pool.backend, lang, k)}")
        if args.dry_run:
            sample = next(s for s in sentences if len(s) > 40)
            logger.info(f"  example: {sample}")
            for c in args.categories:
                for v in pert.variants(sample, c, 2):
                    logger.info(f"    [{c}] {v}")


if __name__ == "__main__":
    main()
