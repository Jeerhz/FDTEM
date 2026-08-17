#!/usr/bin/env python3
"""
prepare_wmt_length_data.py — WMT training data for the length-composition sweep.

Three pools, one COMET-ready schema (src, mt, ref, score, …):

  sent    generalMT2022 avg_seg_scores (en-de, en-ru, zh-en): single segments
          (k=1) with per-segment MQM scores, src+ref+doc+domain self-contained.
  agg     the same segments aggregated into windows of k ∈ {2,3,4,6}
          CONSECUTIVE same-(system, doc) segments; score = mean of the
          per-segment MQM scores (train stride 1, val stride k).
  native  WMT25 general-MT human evaluation (ESA 0-100): the scored unit is a
          whole document; mt = tgt_text[system], ref = tgt_text['refA'].
          Only documents WITH refA are kept, so DA and QE arms can train on
          byte-identical rows. k is recorded as 0 (marker for "native doc").

Scores are z-normalised per (source_set, lp) on train and sigmoid-squashed —
monotone, so rank-based evaluation is unaffected. Splits are document-disjoint
(hash of doc id; 90/10 train/val). Rows longer than --max_tokens XLM-R tokens
on any side are dropped.

Outputs (--output_dir):
  {pool}_{lp}_train.csv / _val.csv     per pool × lp
  {lp}_trainpool.csv                   full per-lp stock (mix sampling pools)
  all_train.csv / all_val.csv          concatenations
  stats.json                           counts + normalisation parameters

Usage:
  python scripts/prepare_wmt_length_data.py \
      --mqm_dir ~/scratch/wmt_data/wmt-mqm-human-evaluation \
      --wmt25 ~/scratch/wmt_data/wmt25/wmt25-genmt-humeval.jsonl \
      --output_dir ~/scratch/wmt_length_data
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
csv.field_size_limit(10 ** 9)

MQM_SETS = {  # lp -> avg_seg_scores tsv (self-contained: sys/hyp/domain/doc/source/ref/score/seg_id)
    "en-de": "generalMT2022/ende/mqm_generalMT2022_ende.avg_seg_scores.tsv",
    "en-ru": "generalMT2022/enru/mqm_generalMT2022_enru.avg_seg_scores.tsv",
    "zh-en": "generalMT2022/zhen/mqm_generalMT2022_zhen.avg_seg_scores.tsv",
}
K_LIST = (2, 3, 4, 6)


def val_doc(doc_id: str, val_frac: float = 0.10) -> bool:
    h = int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 10_000
    return h < val_frac * 10_000


class Tok:
    def __init__(self, model="xlm-roberta-large"):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(model)

    def n(self, text: str) -> int:
        return len(self.t(text, add_special_tokens=False).input_ids)


def load_mqm(mqm_dir: Path, lp: str, rel: str) -> pd.DataFrame:
    rows = []
    with open(mqm_dir / rel) as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE):
            try:
                score = float(r["score"])
            except (ValueError, KeyError):
                continue
            rows.append(dict(system=r["sys"], src=r["source"], mt=r["hyp"],
                             ref=r["ref"], score=score, doc_id=r["doc"],
                             domain=r["domain"], seg_id=int(r["seg_id"])))
    df = pd.DataFrame(rows)
    # ratings are per (system, doc, seg): average duplicates (multiple raters)
    df = (df.groupby(["system", "doc_id", "seg_id"], as_index=False)
            .agg(src=("src", "first"), mt=("mt", "first"), ref=("ref", "first"),
                 score=("score", "mean"), domain=("domain", "first")))
    df["lp"] = lp
    logger.info(f"  [{lp}] {len(df):,} scored segments, "
                f"{df.doc_id.nunique()} docs, {df.system.nunique()} systems")
    return df


def windows(df: pd.DataFrame, k: int, stride: int) -> pd.DataFrame:
    out = []
    for (sys_, doc), g in df.groupby(["system", "doc_id"], sort=False):
        g = g.sort_values("seg_id")
        segs = g.to_dict("records")
        i = 0
        while i + k <= len(segs):
            w = segs[i:i + k]
            out.append(dict(
                system=sys_, doc_id=doc, lp=w[0]["lp"], domain=w[0]["domain"],
                seg_start=w[0]["seg_id"], k=k,
                src=" ".join(s["src"] for s in w),
                mt=" ".join(s["mt"] for s in w),
                ref=" ".join(s["ref"] for s in w),
                score=float(np.mean([s["score"] for s in w]))))
            i += stride
    return pd.DataFrame(out)


def load_wmt25(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            tgt = d.get("tgt_text") or {}
            ref = tgt.get("refA")
            if not ref or not isinstance(ref, str):
                continue
            doc_id = d["doc_id"]
            # doc_id prefix is e.g. "cs-de_DE" or "en-sr_Cyrl_RS" → lp "cs-de"
            lp = doc_id.split("_#_")[0].split("_")[0]
            for sys_, anns in (d.get("scores") or {}).items():
                if sys_.startswith("ref"):
                    continue
                mt = tgt.get(sys_)
                scores = [a["score"] for a in anns
                          if isinstance(a.get("score"), (int, float))]
                if not mt or not isinstance(mt, str) or not scores:
                    continue
                rows.append(dict(system=sys_, src=d["src_text"], mt=mt, ref=ref,
                                 score=float(np.mean(scores)), doc_id=doc_id,
                                 domain=doc_id.split("_#_")[1] if "_#_" in doc_id else "",
                                 seg_start=0, k=0, lp=lp))
    df = pd.DataFrame(rows)
    logger.info(f"  [wmt25] {len(df):,} scored (doc, system) rows with refA, "
                f"{df.doc_id.nunique()} docs, {df.lp.nunique()} lps")
    # keep lps with enough mass to matter
    keep = df.lp.value_counts()
    keep = set(keep[keep >= 300].index)
    df = df[df.lp.isin(keep)].copy()
    logger.info(f"  [wmt25] {len(df):,} rows after lp>=300 filter "
                f"({sorted(keep)})")
    return df


def token_filter(df: pd.DataFrame, tok: Tok, max_tokens: int, tag: str) -> pd.DataFrame:
    def ok(row):
        return (tok.n(row["src"]) <= max_tokens and
                tok.n(row["mt"]) <= max_tokens and
                tok.n(row["ref"]) <= max_tokens)
    keep = df.apply(ok, axis=1)
    logger.info(f"  [{tag}] token filter: {keep.sum():,}/{len(df):,} kept "
                f"(<= {max_tokens} XLM-R tokens per side)")
    return df[keep].copy()


def normalise(train: pd.DataFrame, val: pd.DataFrame, stats: dict):
    for (ss, lp), g in train.groupby(["source_set", "lp"]):
        mu, sd = g.score.mean(), g.score.std() or 1.0
        stats.setdefault("norm", {})[f"{ss}|{lp}"] = dict(mu=float(mu), sigma=float(sd))
        for df in (train, val):
            m = (df.source_set == ss) & (df.lp == lp)
            z = (df.loc[m, "score"] - mu) / sd
            df.loc[m, "score"] = 1 / (1 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mqm_dir", default="~/scratch/wmt_data/wmt-mqm-human-evaluation")
    ap.add_argument("--wmt25", default="~/scratch/wmt_data/wmt25/wmt25-genmt-humeval.jsonl")
    ap.add_argument("--output_dir", default="~/scratch/wmt_length_data")
    ap.add_argument("--max_tokens", type=int, default=480)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    tok = Tok()
    stats = {"args": vars(args)}

    frames_train, frames_val = [], []

    # ── sent + agg pools from generalMT2022 ──────────────────────────────────
    for lp, rel in MQM_SETS.items():
        seg = load_mqm(Path(args.mqm_dir).expanduser(), lp, rel)
        seg["is_val"] = seg.doc_id.map(val_doc)
        sent = seg.rename(columns={"seg_id": "seg_start"}).copy()
        sent["k"] = 1
        sent["source_set"] = "mqm2022"
        sent["origin"] = "sent"
        for k in K_LIST:
            for split, stride in (("train", 1), ("val", k)):
                sub = seg[~seg.is_val] if split == "train" else seg[seg.is_val]
                w = windows(sub, k, stride)
                if w.empty:
                    continue
                w["source_set"] = "mqm2022"
                w["origin"] = "agg"
                w["is_val"] = split == "val"
                (frames_train if split == "train" else frames_val).append(w)
        frames_train.append(sent[~sent.is_val].drop(columns="is_val"))
        frames_val.append(sent[sent.is_val].drop(columns="is_val"))

    # ── native pool from WMT25 ───────────────────────────────────────────────
    nat = load_wmt25(Path(args.wmt25).expanduser())
    nat["source_set"] = "wmt25"
    nat["origin"] = "native"
    nat["is_val"] = nat.doc_id.map(val_doc)
    frames_train.append(nat[~nat.is_val].drop(columns="is_val"))
    frames_val.append(nat[nat.is_val].drop(columns="is_val"))

    cols = ["src", "mt", "ref", "score", "lp", "k", "system", "doc_id",
            "seg_start", "domain", "source_set", "origin"]
    train = pd.concat(frames_train, ignore_index=True)
    val = pd.concat(frames_val, ignore_index=True)
    train = train.drop(columns=[c for c in train.columns if c not in cols])
    val = val.drop(columns=[c for c in val.columns if c not in cols])

    train = token_filter(train, tok, args.max_tokens, "train")
    val = token_filter(val, tok, args.max_tokens, "val")
    normalise(train, val, stats)

    # ── write ────────────────────────────────────────────────────────────────
    for df, split in ((train, "train"), (val, "val")):
        for (origin, lp), g in df.groupby(["origin", "lp"]):
            g[cols].to_csv(out / f"{origin}_{lp}_{split}.csv", index=False)
        df[cols].to_csv(out / f"all_{split}.csv", index=False)
    for lp, g in train.groupby("lp"):
        g[cols].to_csv(out / f"{lp}_trainpool.csv", index=False)

    stats["counts"] = {
        "train": {f"{o}|{lp}": int(n) for (o, lp), n in
                  train.groupby(["origin", "lp"]).size().items()},
        "val": {f"{o}|{lp}": int(n) for (o, lp), n in
                val.groupby(["origin", "lp"]).size().items()},
        "train_total": len(train), "val_total": len(val),
        "train_by_origin": {k: int(v) for k, v in
                            train.origin.value_counts().items()},
    }
    with open(out / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    logger.info(f"train={len(train):,} val={len(val):,} → {out}")
    logger.info(f"by origin: {stats['counts']['train_by_origin']}")


if __name__ == "__main__":
    main()
