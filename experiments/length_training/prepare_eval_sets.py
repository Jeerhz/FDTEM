#!/usr/bin/env python3
"""
prepare_wmt_eval_data.py — held-out paragraph-level TEST sets from WMT MQM.

Builds evaluation CSVs (never trained on) from the raw per-error MQM releases:

  wmt23  generalMT2023 en-de           (native paragraphs, refs by src match)
  wmt24  generalMT2024 en-de/en-es/ja-zh (native paragraphs, refs via .docs)

Per-segment score = -(mean over raters of the MQM penalty), penalty per rater
= Σ error weights: severity 'major' → 5 ('Non-translation' category → 25),
'minor' → 1 (Fluency/Punctuation → 0.1), others 0.  Higher = better.
Rank-based evaluation downstream, so no normalisation is applied.

Output: {set}-{lp}_val.csv in --output_dir (named *_val.csv so
experiments/length_training/eval_correlation.py picks them up), columns
src, mt, ref, score, lp, k(=0 native marker), system, doc_id, seg_start.
Rows without a joinable reference are dropped (count reported).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
csv.field_size_limit(10 ** 9)

SETS = {
    ("wmt23", "en-de"): "generalMT2023/ende/mqm_generalMT2023_ende.tsv",
    ("wmt24", "en-de"): "generalMT2024/mqm_generalMT2024_ende.tsv",
    ("wmt24", "en-es"): "generalMT2024/mqm_generalMT2024_enes.tsv",
    ("wmt24", "ja-zh"): "generalMT2024/mqm_generalMT2024_jazh.tsv",
}


def norm_text(s: str) -> str:
    # drop MQM error-span markup (<v>…</v>) before whitespace normalisation
    s = s.replace("<v>", "").replace("</v>", "")
    return re.sub(r"\s+", " ", s.replace("\\n", " ")).strip()


def error_weight(category: str, severity: str) -> float:
    sev = (severity or "").strip().lower()
    cat = (category or "").strip().lower()
    if sev == "major":
        return 25.0 if "non-translation" in cat else 5.0
    if sev == "minor":
        return 0.1 if ("fluency" in cat and "punctuation" in cat) else 1.0
    return 0.0


def load_raw(path: Path):
    """→ {(system, doc, seg): {'src','mt','raters':{rater: penalty}}}"""
    out = {}
    with open(path) as fh:
        r = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        hdr = [h.strip() for h in next(r)]
        idx = {c: i for i, c in enumerate(hdr)}
        seg_c = "docSegId" if "docSegId" in idx else "seg_id"
        for row in r:
            try:
                src = row[idx["source"]]
                if "CANARY" in src:
                    continue
                key = (row[idx["system"]], row[idx["doc"]], row[idx[seg_c]])
                e = out.setdefault(key, {"src": src, "mt": row[idx["target"]],
                                         "raters": defaultdict(float)})
                e["raters"][row[idx["rater"]]] += error_weight(
                    row[idx["category"]], row[idx["severity"]])
            except IndexError:
                continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mqm_dir", default="~/scratch/wmt_data/wmt-mqm-human-evaluation")
    ap.add_argument("--refs_dir", default="~/scratch/wmt_data/refs")
    ap.add_argument("--output_dir", default="~/scratch/wmt_eval_paragraph")
    args = ap.parse_args()

    mqm = Path(args.mqm_dir).expanduser()
    refs = Path(args.refs_dir).expanduser()
    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    for (set_name, lp), rel in SETS.items():
        raw = load_raw(mqm / rel)

        # reference lookup
        ref_of = {}
        if set_name == "wmt24":
            docs = [l.split("\t")[-1] for l in
                    (refs / f"wmt24.{lp}.docs").read_text().splitlines()]
            ref_lines = (refs / f"wmt24.{lp}.refA.txt").read_text().splitlines()
            pos = defaultdict(int)
            for doc, ref in zip(docs, ref_lines):
                pos[doc] += 1
                ref_of[(doc, str(pos[doc]))] = ref
            def get_ref(doc, seg, src):
                return ref_of.get((doc, seg))
        else:  # wmt23: join by normalised source text
            src_lines = (refs / "wmt23.en-de.src.en").read_text().splitlines()
            ref_lines = (refs / "wmt23.en-de.refA.de").read_text().splitlines()
            by_src = {norm_text(s): r for s, r in zip(src_lines, ref_lines)}
            def get_ref(doc, seg, src):
                return by_src.get(norm_text(src))

        rows, dropped = [], 0
        for (sys_, doc, seg), e in raw.items():
            ref = get_ref(doc, seg, e["src"])
            if not ref:
                dropped += 1
                continue
            pen = float(np.mean(list(e["raters"].values())))
            rows.append(dict(src=norm_text(e["src"]), mt=norm_text(e["mt"]),
                             ref=ref, score=-pen, lp=lp, k=0, system=sys_,
                             doc_id=doc, seg_start=seg))
        df = pd.DataFrame(rows)
        p = out / f"{set_name}-{lp}_val.csv"
        df.to_csv(p, index=False)
        report[f"{set_name}-{lp}"] = dict(
            rows=len(df), dropped_no_ref=dropped,
            systems=int(df.system.nunique()) if len(df) else 0,
            docs=int(df.doc_id.nunique()) if len(df) else 0,
            score_mean=float(df.score.mean()) if len(df) else None)
        logger.info(f"  {set_name}-{lp}: {len(df):,} rows "
                    f"({dropped} dropped w/o ref) → {p.name}")

    with open(out / "build_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info(f"Done → {out}")


if __name__ == "__main__":
    main()
