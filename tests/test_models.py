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


class FakePerturber:
    """`variants()` of the real backends: "<sentence>|<category><i>", as many as `self.n[category]`."""

    def __init__(self, n: dict[str, int]):
        self.n = n

    def variants(self, sent, category, n):
        return [f"{sent}|{category}{i}" for i in range(min(n, self.n.get(category, 0)))]


def test_every_sentence_rotates_categories_and_nests_over_k():
    from common.flores import FloresCorpus
    from part1_block_alignment.models import Block
    from part1_block_alignment.perturb_every_sentence import build_rows

    corpus = FloresCorpus(source="plus", split="dev", langs=["en", "de"], urls=["u"] * 3,
                          sentences={"en": ["e0", "e1", "e2"], "de": ["s0", "s1", "s2"]})
    rows = build_rows(corpus, [Block(rows=[0, 1, 2], url="u", split="dev")], 0, "de", "en",
                      [1, 2, 3], FakePerturber({"causality": 2, "entity": 2, "number": 2}), 6,
                      lambda text: len(text.split()))
    k1, k2, k3 = rows
    assert [r.k for r in rows] == [1, 2, 3] and k3.source == "e0 e1 e2" and k3.source_n_tokens == 3
    assert len(set(k1.distractors)) == 6                          # distinct from k = 1 on
    assert k3.categories[0] == ["causality", "entity", "number"]  # (d + j) % 3
    assert k3.categories[1] == ["entity", "number", "causality"]
    for short, long in ((k1, k2), (k2, k3)):                      # k+1 = k plus one perturbed sentence
        assert all(b.startswith(a + " ") for a, b in zip(short.distractors, long.distractors))
    assert all("|" in s for d in k3.distractors for s in d.split(" "))   # every sentence perturbed


def test_every_sentence_falls_back_and_keeps_distinct_first_sentences():
    from part1_block_alignment.perturb_every_sentence import perturb_window

    dist = perturb_window(["s0", "s1"], FakePerturber({"causality": 2}), 6)
    assert [[c for c, _ in d] for d in dist] == [["causality"] * 2] * 2
    assert len({d[0][1] for d in dist}) == 2                       # the other four repeated s0's edits
    assert perturb_window(["s0", "s1"], FakePerturber({}), 6) == []


def test_first_sentence_control_and_closed_form():
    from itertools import combinations
    from math import comb

    from part1_block_alignment.evaluate_every_sentence import first_sentence_only, success, wins
    from part1_block_alignment.models import PerturbedBlock

    def row(k, ref, dist):
        return PerturbedBlock(id=f"de-w000-k{k}", lang_pair="en-de", window=0, k=k, split="dev",
                              url="u", rows=list(range(k)), source="e", source_n_tokens=1,
                              reference=ref, distractors=dist, categories=[["causality"] * k])
    k1, k3 = row(1, "a", ["A"]), row(3, "a b c", ["A B C"])
    assert first_sentence_only(k3, k1) == ["A b c"] and first_sentence_only(k1, k1) == ["A"]

    scores = np.array([0.5, 0.1, 0.9, 0.2, 0.5])                  # reference, then 4 distractors
    w, m, ties = wins(scores, [4])
    assert (w[0], m[0], ties) == (2, 4, 1)                        # a tie counts against the reference
    for D in (1, 2, 3):
        _, p, rank = success(w, m, D)
        subsets = list(combinations(scores[1:], D))
        assert np.isclose(p[0], sum(all(s < 0.5 for s in sub) for sub in subsets) / comb(4, D))
        assert np.isclose(rank[0], 1 + np.mean([sum(s >= 0.5 for s in sub) for sub in subsets]))
