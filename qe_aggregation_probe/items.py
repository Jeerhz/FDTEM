#!/usr/bin/env python3
"""
items.py — assemble the experimental items and scoring jobs.

For each k-sentence block we create paragraph variants with m ∈ {0..k} perturbed
sentences per condition (severity × family). WHICH positions carry the
perturbation is counterbalanced deterministically:

    positions(block, m) = combinations(range(k), m)[ block_idx mod C(k,m) ]

so across blocks every position set appears equally often, yet the design is
fully reproducible. The m=0 variant is shared across conditions (condition
"none"). A (block, condition) cell is dropped whole if any of its k unit
perturbations fails QC — this keeps the m-design balanced within every cell.

Part scores s_i are defined as the score of sentence i ALONE (single-sentence
input); S is the score of the concatenated paragraph.
"""
from __future__ import annotations

import itertools
import logging
from typing import Dict, List, Tuple

import pandas as pd

from common import COND_NONE, all_conditions
from data import Block
from perturb import Perturber
from scoring import job_key

logger = logging.getLogger(__name__)


def positions_for(k: int, m: int, block_idx: int) -> Tuple[int, ...]:
    combs = list(itertools.combinations(range(k), m))
    return combs[block_idx % len(combs)]


def build_items(cfg: dict, blocks: List[Block], perturber: Perturber) -> List[dict]:
    """Returns a list of item dicts (texts + design metadata, no scores yet)."""
    conditions = all_conditions(cfg)
    items: List[dict] = []
    n_dropped_cells = 0

    for b in blocks:
        # one perturbed variant of every unit, per condition
        pert_units: Dict[str, List[str]] = {}
        for cond in conditions:
            outs = [perturber.perturb(b.src_units[i], b.tgt_units[i], b.lang, cond,
                                      seed_key=f"{b.block_id}|{i}")
                    for i in range(b.k)]
            if any(o is None for o in outs):
                n_dropped_cells += 1
                continue
            pert_units[cond] = outs  # type: ignore[assignment]

        def make_item(cond: str, m: int, positions: Tuple[int, ...]) -> dict:
            if m == 0:
                var_units = list(b.tgt_units)
                severity = family = None
            else:
                var_units = [pert_units[cond][i] if i in positions else b.tgt_units[i]
                             for i in range(b.k)]
                severity, family = cond.split("_", 1)
            return {
                "lang": b.lang, "lp": b.lp, "k": b.k, "m": m,
                "condition": cond if m else COND_NONE,
                "severity": severity, "family": family,
                "positions": list(positions),
                "doc_id": b.doc_id, "block_id": b.block_id, "block_idx": b.block_idx,
                "n_tokens_src": b.n_tokens_src, "n_tokens_tgt": b.n_tokens_tgt,
                "truncated": b.truncated,
                "src_units": b.src_units, "ref_units": b.tgt_units,
                "var_units": var_units,
                "src_par": b.join_src(), "ref_par": b.join_tgt(),
                "tgt_par": b.join_tgt(var_units),
            }

        items.append(make_item(COND_NONE, 0, ()))
        for cond in pert_units:
            for m in range(1, b.k + 1):
                items.append(make_item(cond, m, positions_for(b.k, m, b.block_idx)))

    logger.info(f"  items: {len(items)} paragraph variants "
                f"({n_dropped_cells} (block,condition) cells dropped by QC)")
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
            "doc_id": it["doc_id"], "block_id": it["block_id"],
            "n_tokens_src": it["n_tokens_src"], "n_tokens_tgt": it["n_tokens_tgt"],
            "truncated": it["truncated"], "S": S,
        }
        for i in range(k_max):
            row[f"s_{i + 1}"] = s_parts[i] if i < it["k"] else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
