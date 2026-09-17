"""Data models of part 2: CSV schemas, mixes, arms, presets and the three result files.

Result models mirror the committed JSON (`results/*.json`): written with
`model_dump_json(indent=2)`, read with `model_validate_json`. k keys stay strings.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from common.stats import CorrelationCell

# ── CSV schemas ──────────────────────────────────────────────────────────────


class PoolRow(BaseModel):
    """One row of the WMT pools and of every training mix (k=1 sentence, 2..6 window, 0 native doc)."""
    model_config = ConfigDict(extra="forbid")
    src: str
    mt: str
    ref: str
    score: float
    lp: str
    k: int
    system: str
    doc_id: str
    seg_start: int
    domain: str
    source_set: str
    origin: str


class HeldoutRow(BaseModel):
    """One row of a held-out evaluation set (WMT23/24 paragraphs, raw MQM score)."""
    model_config = ConfigDict(extra="forbid")
    src: str
    mt: str
    ref: str
    score: float
    lp: str
    k: int
    system: str
    doc_id: str
    seg_start: str


POOL_COLUMNS = list(PoolRow.model_fields)
HELDOUT_COLUMNS = list(HeldoutRow.model_fields)

# ── mixes ────────────────────────────────────────────────────────────────────


class MixSpec(BaseModel):
    name: str
    kind: Literal["mix", "nat", "agg", "uncontrolled"]
    sentence_fraction: int
    description: str  # figure label, named by what the arm was trained on


class MixCounts(BaseModel):
    sent: int
    agg: int
    native: int
    per_lp: dict[str, int]
    per_k: dict[str, int]
    per_origin_lp: dict[str, int]

    @property
    def total(self) -> int:
        return self.sent + self.agg + self.native


class MixEntry(BaseModel):
    counts: MixCounts
    checksums_md5: dict[str, str]


class MixManifest(BaseModel):
    """`<mix_dir>/manifest.json` written by make_mixtures.py for the controlled mixes."""
    seed: int
    total_per_mix: int
    total_policy: str
    n_feasible: dict[str, int]
    pool_sizes: dict[str, int]
    lp_coverage: dict[str, list[str]]
    lp_coverage_shared: list[str]
    match_lp_coverage: bool
    mixes: dict[str, MixEntry]
    skipped: dict[str, dict[str, dict[str, int]]] = {}

    def assert_equal_totals(self, arms: list[str]) -> int:
        """The design guarantee: one epoch is the same number of rows in every arm. Returns N."""
        missing = [a for a in arms if a not in self.mixes]
        if missing:
            raise SystemExit(f"mixes {missing} are absent from the manifest (present: "
                             f"{sorted(self.mixes)}). A pure-pool arm needs a whole N from one "
                             f"pool: rebuild with make_mixtures --total_policy pure --arms {' '.join(arms)}")
        totals = {a: self.mixes[a].counts.total for a in arms}
        if len(set(totals.values())) != 1:
            raise SystemExit(f"arms have unequal row counts, so an epoch means different things "
                             f"in each: {totals}")
        return next(iter(totals.values()))


class UncontrolledManifest(BaseModel):
    """`<mix_dir>/uncontrolled/manifest.json`: the contaminated everything-mix."""
    mix: str
    contaminated: bool
    contaminated_lenses: list[str]
    clean_lenses: list[str]
    seed: int
    rows: dict[str, int]
    provenance: dict[str, int]
    val_note: str
    checksums_md5: dict[str, str]

# ── arms and presets ─────────────────────────────────────────────────────────

DIR_PREFIX = {"da": "mix-", "qe": "kiwi-mix-"}  # checkpoint directory prefix per family


class ArmLabel(BaseModel):
    """One trained arm. `label` is the results key, `dir_name` the checkpoint directory."""
    family: Literal["da", "qe"]
    mix: str
    frozen: bool = False

    @property
    def label(self) -> str:
        return f"{self.family}-{self.mix}" + ("-frozen" if self.frozen else "")

    @property
    def dir_name(self) -> str:
        return f"{DIR_PREFIX[self.family]}{self.mix}" + ("-frozen" if self.frozen else "")

    @classmethod
    def parse(cls, label: str) -> "ArmLabel":
        """`qe-frac040-frozen` -> ArmLabel(qe, frac040, frozen)."""
        family, _, rest = label.partition("-")
        frozen = rest.endswith("-frozen")
        return cls(family=family, mix=rest.removesuffix("-frozen"), frozen=frozen)

    @classmethod
    def from_dir(cls, dir_name: str) -> "ArmLabel":
        """`kiwi-mix-frac040-frozen` -> ArmLabel(qe, frac040, frozen). Longest prefix wins."""
        for family, prefix in sorted(DIR_PREFIX.items(), key=lambda kv: -len(kv[1])):
            if dir_name.startswith(prefix):
                rest = dir_name[len(prefix):]
                return cls(family=family, mix=rest.removesuffix("-frozen"),
                           frozen=rest.endswith("-frozen"))
        raise ValueError(f"{dir_name!r} is not an arm directory (prefixes: {list(DIR_PREFIX.values())})")


class TrainPreset(BaseModel):
    max_epochs: int
    patience: int
    encoder_lr: float | None = None
    head_lr: float | None = None
    nr_frozen_epochs: float | None = None
    val_files_from_mix: bool = False  # monitor the mix's own slice (contaminated arm)

# ── results: correlation lens ────────────────────────────────────────────────


class MeanCorrelation(BaseModel):
    pearson: float | None
    spearman: float | None
    kendall: float | None


class ModelCorrelation(BaseModel):
    by_file: dict[str, dict[str, CorrelationCell]]  # file stem -> k -> cell
    mean_by_k: dict[str, MeanCorrelation]


class CorrelationResults(BaseModel):
    model_config = ConfigDict(extra="ignore")
    experiment: str
    timestamp: str
    data_dir: str
    models: dict[str, ModelCorrelation]

# ── results: length profile ──────────────────────────────────────────────────


class ScoreDistribution(BaseModel):
    n: int
    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    quantiles: dict[str, float]
    iqr: float | None
    hist: list[int]
    below_grid: int
    above_grid: int


class ProfileCell(BaseModel):
    pred: ScoreDistribution
    gold: ScoreDistribution
    corr: CorrelationCell
    tokens: dict[str, float | None]  # mean, median


class KProfileCell(ProfileCell):
    spread_ratio_vs_k1: float | None


class FileProfile(BaseModel):
    all: ProfileCell
    by_k: dict[str, KProfileCell]
    by_tokens: dict[str, dict[str, ProfileCell]]  # n_mt | n_concat -> bin -> cell


class LengthProfileResults(BaseModel):
    model_config = ConfigDict(extra="ignore")
    experiment: str
    question: str
    timestamp: str
    data_dir: str
    token_edges: list[int]
    token_bins: list[str]
    score_grid: list[float]
    quantiles: list[int]
    models: dict[str, dict[str, FileProfile]]  # label -> file stem -> profile

# ── results: MetaDocEval ─────────────────────────────────────────────────────


class AccuracyCell(BaseModel):
    n_docs: int
    accuracy: float | None
    ties: int
    mean_diff: float | None
    p_value: float | None


class ModelAccuracy(BaseModel):
    referenceless: bool
    per_cell: dict[str, AccuracyCell]  # "lp|system|perturbation|w<w>"
    micro: dict[str, AccuracyCell]     # "perturbation|w<w>"


class MetaDocEvalResults(BaseModel):
    model_config = ConfigDict(extra="ignore")
    experiment: str
    reference: str
    test_set: str
    protocol: str
    timestamp: str
    testset_commit: str | None
    windows: list[int]
    counts_per_lp_system: dict[str, dict[str, int]]
    counts_per_perturbation: dict[str, dict[str, int]]
    readme_mismatches: dict[str, dict[str, dict[str, int]]]
    quality_preserving: list[str]
    models: dict[str, ModelAccuracy]
