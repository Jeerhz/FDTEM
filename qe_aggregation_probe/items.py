#!/usr/bin/env python3
"""
items.py — assemble the experimental items and scoring jobs (nested design).

Protocol guarantees (in order of the design constraints they answer):

1. LENGTH IS THE ONLY VARIABLE ACROSS SCALES. Scales are prefixes of the same
   family, and the perturbed variant of unit i under condition c is generated
   ONCE per family and frozen — the same perturbation instance appears at every
   scale where position i is perturbed. Growing k never re-perturbs previous
   segments with edits of different difficulty; new perturbations only ever
   arrive with newly added segments (monotone chains, see 4).

2. IDENTICAL CANDIDATE POOL AT EVERY SCALE. Families require k_max units and
   pass the token filter at k_max (data.build_families), and a (family,
   condition) cell that fails QC is dropped at ALL scales, so every scale sees
   exactly the same families and the same surviving cells.

3. A FAILED PERTURBATION NEVER FALLS BACK TO THE CLEAN SENTENCE. Perturber
   returns None on failure → the whole (family, condition) cell is dropped; at
   assembly time we additionally assert that every perturbed position differs
   textually from its clean counterpart.

4. ONE PERTURBATION PER SEGMENT, CONTROLLED POSITIONS AND TYPES. Each perturbed
   segment carries exactly one edit; all perturbed segments of an item share
   one condition (severity × error-family is a between-item factor). Positions
   follow a deterministic per-family permutation π (seeded by family identity,
   shared across conditions): the perturbation sets are the nested chain
   P_1 = {π₁} ⊂ P_2 ⊂ ... ⊂ P_k_max. At scale k the realized set is
   P_j ∩ [0,k), which yields exactly one item per m ∈ {1..k} — full, balanced
   m-coverage at every scale, and item(k, m) ⊂ item(k', m') whenever the chain
   extends (the perturbed prefix text is literally identical).

Part scores s_i are defined as the score of sentence i ALONE (single-sentence
input); S is the score of the concatenated paragraph. At k=1, S ≡ s_1 by
construction (same input text) — the trivial anchor; k=1 items feed the
dilution/severity tables, not the aggregation fits.
"""
from __future__ import annotations

import logging
import random
from typing import Dict, List, Sequence

import pandas as pd

from common import COND_NONE, all_conditions
from data import Family
from perturb import Perturber
from scoring import job_key

logger = logging.getLogger(__name__)


def chain_permutation(family: Family, seed: int) -> List[int]:
    """Deterministic per-family order in which positions get perturbed along the
    chain. Seeded by family IDENTITY (not index), so adding/removing families
    never reshuffles the others; shared across conditions, so conditions differ
    only by error type. Uniformly random across families → position balance."""
    rng = random.Random(f"{seed}|chain|{family.family_id}")
    perm = list(range(family.k_max))
    rng.shuffle(perm)
    return perm


def build_items(cfg: dict, families: List[Family], perturber: Perturber) -> List[dict]:
    """Returns a list of item dicts (texts + design metadata, no scores yet)."""
    conditions = all_conditions(cfg)
    k_list = sorted(cfg["design"]["k_list"])
    seed = int(cfg["seed"])
    items: List[dict] = []
    n_dropped_cells = 0

    for fam in families:
        perm = chain_permutation(fam, seed)

        # one perturbed variant of every unit, per condition — generated ONCE
        # and reused at every scale/chain step (guarantee 1)
        pert_units: Dict[str, List[str]] = {}
        for cond in conditions:
            outs = [perturber.perturb(fam.src_units[i], fam.tgt_units[i], fam.lang,
                                      cond, seed_key=f"{fam.family_id}|{i}")
                    for i in range(fam.k_max)]
            if any(o is None for o in outs):
                n_dropped_cells += 1        # drop at ALL scales (guarantees 2+3)
                continue
            for i, o in enumerate(outs):    # guarantee 3, belt and braces
                assert o != fam.tgt_units[i], (
                    f"clean sentence leaked into perturbed set: "
                    f"{fam.family_id} unit {i} cond {cond}")
            pert_units[cond] = outs  # type: ignore[assignment]

        def make_item(k: int, cond: str, positions: Sequence[int],
                      chain_step: int) -> dict:
            if positions:
                var_units = [pert_units[cond][i] if i in positions
                             else fam.tgt_units[i] for i in range(k)]
                severity, family_ = cond.split("_", 1)
            else:
                var_units = list(fam.tgt_units[:k])
                severity = family_ = None
            nts, ntt = fam.prefix_tokens[k]
            return {
                "lang": fam.lang, "lp": fam.lp, "k": k, "m": len(positions),
                "condition": cond if positions else COND_NONE,
                "severity": severity, "family": family_,
                "positions": list(positions), "chain_step": chain_step,
                "chain_plan": ",".join(map(str, perm)),
                "doc_id": fam.doc_id, "family_id": fam.family_id,
                "block_id": f"{fam.family_id}_k{k}",
                "family_idx": fam.family_idx,
                "n_tokens_src": nts, "n_tokens_tgt": ntt,
                "truncated": fam.truncated,
                "src_units": fam.src_units[:k], "ref_units": fam.tgt_units[:k],
                "var_units": var_units,
                "src_par": fam.join_src(k), "ref_par": fam.join_tgt(k=k),
                "tgt_par": fam.join_tgt(var_units, k=k),
            }

        for k in k_list:
            items.append(make_item(k, COND_NONE, (), 0))       # m=0, shared
            for cond in pert_units:
                realized: List[int] = []
                for j, pos in enumerate(perm, start=1):
                    if pos >= k:
                        continue            # this chain step only extends beyond k
                    realized = sorted(realized + [pos])
                    items.append(make_item(k, cond, tuple(realized), j))

    logger.info(f"  items: {len(items)} paragraph variants "
                f"({n_dropped_cells} (family,condition) cells dropped whole by QC)")
    return items


# ════════════════════════════════════════════════════════════════════════════
# Scoring jobs
# ════════════════════════════════════════════════════════════════════════════
def _sent_job(item: dict, i: int) -> dict:
    is_pert = i in item["positions"] and item["m"] > 0
    return {"src": item["src_units"][i], "mt": item["var_units"][i],
            "ref": item["ref_units"][i],
            "meta": {"level": "sent", "is_pert": is_pert,
                     "condition": item["condition"] if is_pert else None}}


def _par_job(item: dict) -> dict:
    parts = [{"text": item["var_units"][i],
              "is_pert": (i in item["positions"] and item["m"] > 0),
              "condition": item["condition"] if (i in item["positions"] and item["m"] > 0)
              else None}
             for i in range(item["k"])]
    return {"src": item["src_par"], "mt": item["tgt_par"], "ref": item["ref_par"],
            "meta": {"level": "par", "parts": parts}}


def build_jobs(items: List[dict]) -> List[dict]:
    """Sentence jobs first, then paragraph jobs (dedup happens in score_jobs)."""
    jobs: List[dict] = []
    for it in items:
        for i in range(it["k"]):
            jobs.append(_sent_job(it, i))
    for it in items:
        jobs.append(_par_job(it))
    return jobs


# ════════════════════════════════════════════════════════════════════════════
# Tidy dataframe (one row = one paragraph variant × one metric)
# ════════════════════════════════════════════════════════════════════════════
def assemble_df(items: List[dict], metric_name: str,
                scores: Dict[str, float], k_max: int) -> pd.DataFrame:
    rows = []
    for it in items:
        s_parts = [scores[job_key(metric_name, _sent_job(it, i))] for i in range(it["k"])]
        S = scores[job_key(metric_name, _par_job(it))]
        row = {
            "metric": metric_name, "lang": it["lang"], "lp": it["lp"],
            "k": it["k"], "m": it["m"], "condition": it["condition"],
            "severity": it["severity"], "family": it["family"],
            "positions": ",".join(map(str, it["positions"])),
            "chain_step": it["chain_step"], "chain_plan": it["chain_plan"],
            "doc_id": it["doc_id"], "family_id": it["family_id"],
            "block_id": it["block_id"],
            "n_tokens_src": it["n_tokens_src"], "n_tokens_tgt": it["n_tokens_tgt"],
            "truncated": it["truncated"], "S": S,
        }
        for i in range(k_max):
            row[f"s_{i + 1}"] = s_parts[i] if i < it["k"] else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
