"""Every committed result JSON loads through its pydantic model without losing a value.

    PYTHONPATH=. pytest tests
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.paths import ROOT
from part1_block_alignment.models import (DuelRunResult, EncoderRunResult, EverySentenceRunResult,
                                          ScoreRunResult)
from part2_length_training.models import CorrelationResults, LengthProfileResults, MetaDocEvalResults

P1 = ROOT / "part1_block_alignment" / "results"
P2 = ROOT / "part2_length_training" / "results"

CASES = [
    (P1 / "duel.json", DuelRunResult),
    (P1 / "comet_score.json", ScoreRunResult),
    (P1 / "every_sentence.json", EverySentenceRunResult),
    (P1 / "encoder_cosine_2026-07-16_heuristic.json", EncoderRunResult),
    (P1 / "encoder_cosine_arm_frac000.json", EncoderRunResult),
    (P1 / "encoder_cosine_arm_frac100.json", EncoderRunResult),
    (P1 / "encoder_cosine_spacy_smoke.json", EncoderRunResult),
    (P2 / "correlation_val.json", CorrelationResults),
    (P2 / "correlation_heldout.json", CorrelationResults),
    (P2 / "wave1_correlation_heldout.json", CorrelationResults),
    (P2 / "metadoceval.json", MetaDocEvalResults),
    (P2 / "wave1_metadoceval.json", MetaDocEvalResults),
    (P2 / "length_profile_val.json", LengthProfileResults),
    (P2 / "length_profile_heldout.json", LengthProfileResults),
]


def leaves(node, path=()):
    """(path, value) for every scalar of a nested JSON structure."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaves(v, path + (i,))
    else:
        yield path, node


def lookup(node, path):
    for key in path:
        node = node[key]
    return node


@pytest.mark.parametrize("path,model", CASES, ids=[c[0].name for c in CASES])
def test_round_trip_keeps_every_value(path: Path, model):
    raw = json.loads(path.read_text(encoding="utf-8"))
    dumped = model.model_validate_json(path.read_text(encoding="utf-8")).model_dump(mode="json")
    for leaf_path, value in leaves(raw):
        assert lookup(dumped, leaf_path) == value, f"{path.name}: {leaf_path} changed"
