"""The small contracts the shell and the figures rely on."""
from __future__ import annotations

import numpy as np

from common.stats import correlations
from part1_block_alignment.models import Candidate, CandidatePool
from part2_length_training.arms import MIX_SPECS, TRAIN_PRESETS
from part2_length_training.models import ArmLabel


def test_arm_label_round_trips_both_encodings():
    arm = ArmLabel(family="qe", mix="frac040", frozen=True)
    assert arm.label == "qe-frac040-frozen"
    assert arm.dir_name == "kiwi-mix-frac040-frozen"
    assert ArmLabel.parse(arm.label) == arm
    assert ArmLabel.from_dir(arm.dir_name) == arm
    assert ArmLabel.from_dir("mix-frac000").label == "da-frac000"


def test_mix_specs_cover_the_four_arms_and_the_ladder():
    assert {"frac100", "frac000agg", "frac000nat", "frac000", "uncontrolled"} <= set(MIX_SPECS)
    assert MIX_SPECS["frac040"].sentence_fraction == 40
    assert MIX_SPECS["uncontrolled"].kind == "uncontrolled"
    assert TRAIN_PRESETS["default"].max_epochs == 60 and TRAIN_PRESETS["wave1"].max_epochs == 6


def test_correlations_degenerate_cell_is_none():
    cell = correlations(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))
    assert cell.n == 3 and cell.kendall is None


def test_duel_items_take_the_first_variant_per_position_and_category():
    cands = [Candidate(text="gold", block_id=0, kind="true"),
             Candidate(text="n0", block_id=0, kind="perturbed", category="number", position=0, variant=0),
             Candidate(text="n1", block_id=0, kind="perturbed", category="number", position=0, variant=1),
             Candidate(text="e0", block_id=0, kind="perturbed", category="entity", position=1, variant=0)]
    pool = CandidatePool(lang="de", query_lang="en", pool_lang="de", k=2, backend="heuristic", seed=42,
                         splits=["dev"], queries=["src"], candidates=cands, true_index=[0],
                         variant_counts={"number": 2, "entity": 1, "blocks": 1})
    (item,) = pool.duel_items()
    assert item.gold == "gold" and [n.text for n in item.negatives] == ["n0", "e0"]
    assert pool.own_negatives() == [[1, 2, 3]]
