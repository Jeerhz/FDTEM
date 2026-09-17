"""Data rows and result files of part 3: the contract between load_bio_mqm.py and evaluate.py."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from common.stats import CorrelationCell

# MQM severity -> penalty (Freitag et al. 2021); a segment's score is the annotator mean.
MQM_WEIGHTS: dict[str, float] = {
    "minor": -1.0,
    "major": -5.0,
    "critical": -25.0,
    "no-error": 0.0,
    "neutral": 0.0,
}

LANG_PAIRS = ["en-de", "de-en", "en-es", "es-en", "en-fr", "fr-en",
              "en-ru", "ru-en", "en-zh", "zh-en"]


class BioMqmRow(BaseModel):
    """One line of <lp>_{train,val}.csv. COMET reads src/mt/ref/score; evaluate.py groups by doc_id."""
    model_config = ConfigDict(extra="forbid")

    src: str
    mt: str
    ref: str
    score: float
    lp: str
    system: str
    doc_id: str
    seg_id: str


class MeanCorrelation(BaseModel):
    """Mean over language pairs of the per-pair correlations."""
    n_lps: int
    pearson: float | None
    spearman: float | None
    kendall: float | None


class BioEvalResults(BaseModel):
    """results/correlation.json: base vs fine-tuned model on the Bio-MQM validation split."""
    model_config = ConfigDict(extra="ignore")

    timestamp: str
    data_dir: str
    models: dict[str, dict[str, CorrelationCell]]  # label -> lp -> cell
    mean: dict[str, MeanCorrelation]  # label -> mean over lps
    delta_vs_base: dict[str, dict[str, tuple[float, float, float]]]  # label -> lp -> (delta tau, lo, hi)
