"""Data and result models of part 1 — the contract between its scripts.

Data models (blocks, candidate pools) are strict; result models mirror the JSON
files under results/ and ignore unknown keys. Keys `k` and `D` are strings.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

CATEGORIES: tuple[str, ...] = ("causality", "entity", "number")


class DataModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResultModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ── data ──────────────────────────────────────────────────────────────────────
class Block(DataModel):
    """k consecutive rows of one article; row indices into a FloresCorpus."""

    rows: list[int]
    url: str
    split: str = ""

    @property
    def k(self) -> int:
        return len(self.rows)


class Candidate(DataModel):
    text: str
    block_id: int                                   # index into the block list
    kind: str = Field(pattern="^(true|perturbed)$")
    category: str | None = None                     # perturbed only
    position: int | None = None                     # perturbed sentence index in the block
    variant: int | None = None                      # rank among the (position, category) variants


class Negative(DataModel):
    position: int
    category: str
    text: str
    pool_index: int


class DuelItem(DataModel):
    block_id: int
    query: str
    gold: str
    negatives: list[Negative]


class CandidatePool(DataModel):
    """Queries and candidates of one (lang, k): every block's gold plus its own negatives."""

    lang: str
    query_lang: str
    pool_lang: str
    k: int
    backend: str
    seed: int
    splits: list[str]
    queries: list[str]
    candidates: list[Candidate]
    true_index: list[int]                           # pool position of each block's gold
    variant_counts: dict[str, int]                  # per category, plus "blocks"

    @property
    def n_blocks(self) -> int:
        return len(self.queries)

    def own_negatives(self) -> list[list[int]]:
        """Pool positions of each block's own perturbed candidates."""
        out: list[list[int]] = [[] for _ in range(self.n_blocks)]
        for i, c in enumerate(self.candidates):
            if c.kind == "perturbed":
                out[c.block_id].append(i)
        return out

    def duel_items(self) -> list[DuelItem]:
        """One duel per block: gold + the first variant of every (position, category)."""
        items = [DuelItem(block_id=b, query=q, gold=self.candidates[self.true_index[b]].text,
                          negatives=[]) for b, q in enumerate(self.queries)]
        for i, c in enumerate(self.candidates):
            if c.kind == "perturbed" and c.variant == 0:
                items[c.block_id].negatives.append(Negative(
                    position=c.position, category=c.category, text=c.text, pool_index=i))
        return items

    @classmethod
    def load(cls, path: Path) -> CandidatePool:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")


class PerturbedBlock(DataModel):
    """One row of the every-sentence dataset (perturb_every_sentence.py): a k-sentence
    block, its reference and distractors perturbed in every sentence.

    Rows of one window share `window` across k and languages; distractor d at k+1 is
    distractor d at k plus one perturbed sentence."""

    id: str                                         # <lang>-w<window>-k<k>
    lang_pair: str                                  # en-de
    window: int
    k: int
    split: str
    url: str
    rows: list[int]                                 # FLORES rows of the split, in order
    source: str                                     # the English block: the query
    source_n_tokens: int                            # tokens of `source`, special tokens excluded
    reference: str
    distractors: list[str]
    categories: list[list[str]]                     # [d][j]: category of sentence j of distractor d


# ── per-cell metrics ──────────────────────────────────────────────────────────
class PoolMetrics(ResultModel):
    """One (encoder, lang, k) cell of evaluate_encoders; older files lack some fields."""

    n_blocks: int | None = None
    pool_size: int | None = None
    xsim_err: float | None = None
    xsimpp_err: float | None = None
    xsimpp_err_hard_pool: float | None = None
    xsimpp_err_own_pool: float | None = None
    pool_ablation: dict[str, float | None] | None = None
    own_pool_breakdown: dict[str, float | None] | None = None
    own_pool_err_by_category: dict[str, float | None] | None = None
    own_pool_n_blocks: int | None = None
    own_pool_n_blocks_by_category: dict[str, int] | None = None
    own_pool_chance_err: float | None = None
    own_pool_negatives_per_block: float | None = None
    error_breakdown: dict[str, float] | None = None
    per_category_err: dict[str, float] | None = None
    category_combos: dict[str, float] | None = None
    detection_rate: float | None = None
    detection_by_category: dict[str, float | None] | None = None
    detection_by_position: dict[str, float | None] | None = None
    n_negatives_by_category: dict[str, int] | None = None
    margin_vs_best_negative: float | None = None
    margin_vs_best_own_perturbation: float | None = None
    variant_stats: dict[str, int] = {}


class DuelSizeMetrics(ResultModel):
    n_used: int | None = None
    n_skipped: int | None = None
    coverage: float | None = None
    mean_negatives: float | None = None
    duel_err: float | None = None
    all_negatives_err: float | None = None


class DuelMetrics(ResultModel):
    n_blocks: int | None = None
    by_duel_size: dict[str, DuelSizeMetrics] = {}
    detection_rate: float | None = None
    beat_by_category: dict[str, float | None] = {}
    beat_by_position: dict[str, float | None] = {}


class ScoreMetrics(ResultModel):
    n_blocks: int | None = None
    n_duels: int | None = None
    duel_accuracy: float | None = None
    duel_ties: int | None = None
    duel_accuracy_by_category: dict[str, float | None] = {}
    duel_n_by_category: dict[str, int] = {}
    duel_accuracy_by_position: dict[str, float | None] = {}
    n_shortlists: int | None = None
    shortlist_skipped: int | None = None
    shortlist_coverage: float | None = None
    shortlist_accuracy: float | None = None
    shortlist_mean_rank: float | None = None
    frac_negatives_beaten: float | None = None
    variant_stats: dict[str, int] = {}


class DecisionMetrics(ResultModel):
    """Reference vs D distractors, averaged over every D-subset of a window's distractors."""

    n_used: int | None = None                       # windows with at least D distractors
    coverage: float | None = None
    success: float | None = None                    # A(D) = mean C(w, D) / C(m, D)
    success_ci: list[float] | None = None           # 95 %, articles resampled
    mean_rank: float | None = None                  # R(D) = mean 1 + D (m - w) / m


class EverySentenceMetrics(ResultModel):
    n_windows: int | None = None
    pairwise_win: float | None = None               # mean w / m
    ties: int | None = None                         # distractors scored exactly like the reference
    mean_source_tokens: float | None = None
    by_D: dict[str, DecisionMetrics] = {}


# ── mean over languages ───────────────────────────────────────────────────────
def _mean_values(values: list):
    """Counts (int) are summed, rates (float) averaged, None skipped, dicts recursed."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if all(isinstance(v, dict) for v in vals):
        return _mean_dicts(vals)
    if all(isinstance(v, int) for v in vals):
        return int(sum(vals))
    if all(isinstance(v, (int, float)) for v in vals):
        return float(sum(vals) / len(vals))
    return None


def _mean_dicts(dicts: list[dict]) -> dict:
    keys = list(dict.fromkeys(k for d in dicts for k in d))
    return {k: _mean_values([d.get(k) for d in dicts]) for k in keys}


def mean_over_langs(cells: Sequence[ResultModel]) -> ResultModel:
    """The mean over languages of every numeric field of the cells, nested dicts included."""
    return type(cells[0]).model_validate(_mean_dicts([c.model_dump() for c in cells]))


def mean_by_k(by_lang: dict[str, dict[str, ResultModel]]) -> dict[str, ResultModel]:
    ks = sorted({k for cells in by_lang.values() for k in cells}, key=int)
    return {k: mean_over_langs([cells[k] for cells in by_lang.values() if k in cells])
            for k in ks}


# ── run results: one model's cells, then the whole file, per evaluation ──────
class ModelCells(ResultModel):
    kind: str | None = None                         # scorer rule (comet_score.json only)
    spec: str | None = None                         # CLI spec the model was built from


class EncoderCells(ModelCells):
    by_lang: dict[str, dict[str, PoolMetrics]] = {}  # lang -> k -> cell
    mean_by_k: dict[str, PoolMetrics] = {}


class DuelCells(ModelCells):
    by_lang: dict[str, dict[str, DuelMetrics]] = {}
    mean_by_k: dict[str, DuelMetrics] = {}


class ScoreCells(ModelCells):
    by_lang: dict[str, dict[str, ScoreMetrics]] = {}
    mean_by_k: dict[str, ScoreMetrics] = {}


class ConditionCells(ResultModel):
    by_lang: dict[str, dict[str, EverySentenceMetrics]] = {}
    mean_by_k: dict[str, EverySentenceMetrics] = {}


class EverySentenceCells(ModelCells):
    conditions: dict[str, ConditionCells] = {}      # every_sentence | first_sentence


class RunResult(ResultModel):
    experiment: str
    timestamp: str
    config: dict = {}
    question: str | None = None
    protocols: dict[str, str] | None = None
    candidate_set: str | None = None

    @classmethod
    def load(cls, path: Path):
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")


class EncoderRunResult(RunResult):
    models: dict[str, EncoderCells] = {}


class DuelRunResult(RunResult):
    models: dict[str, DuelCells] = {}


class ScoreRunResult(RunResult):
    models: dict[str, ScoreCells] = {}


class EverySentenceRunResult(RunResult):
    models: dict[str, EverySentenceCells] = {}
