"""Data and result models of part 1 — the contract between its scripts.

Data models (blocks, candidate pools) are strict; result models mirror the JSON
files under results/ and ignore unknown keys. Keys `k` and `D` are strings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Generic, Literal, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict

Category = Literal["causality", "entity", "number"]
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
    kind: Literal["true", "perturbed"]
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
    def load(cls, path: Path) -> "CandidatePool":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")


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


M = TypeVar("M", PoolMetrics, DuelMetrics, ScoreMetrics)


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


def mean_over_langs(cells: Sequence[M]) -> M:
    """The mean over languages of every numeric field of the cells, nested dicts included."""
    return type(cells[0]).model_validate(_mean_dicts([c.model_dump() for c in cells]))


def mean_by_k(by_lang: dict[str, dict[str, M]]) -> dict[str, M]:
    ks = sorted({k for cells in by_lang.values() for k in cells}, key=int)
    return {k: mean_over_langs([cells[k] for cells in by_lang.values() if k in cells])
            for k in ks}


# ── run results ───────────────────────────────────────────────────────────────
class ModelCells(ResultModel, Generic[M]):
    kind: str | None = None                         # scorer rule (comet_score.json only)
    spec: str | None = None                         # CLI spec the model was built from
    by_lang: dict[str, dict[str, M]] = {}           # lang -> k -> cell
    mean_by_k: dict[str, M] = {}


class RunResult(ResultModel, Generic[M]):
    experiment: str
    timestamp: str
    config: dict = {}
    question: str | None = None
    protocols: dict[str, str] | None = None
    candidate_set: str | None = None
    models: dict[str, ModelCells[M]] = {}

    @classmethod
    def load(cls, path: Path):
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")


EncoderRunResult = RunResult[PoolMetrics]
DuelRunResult = RunResult[DuelMetrics]
ScoreRunResult = RunResult[ScoreMetrics]
