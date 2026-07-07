#!/usr/bin/env python3
"""
data.py — load / clean / split CAP major-topic data for the length-probe.

Primary source (config data.source = btl_local | btl_csv)
---------------------------------------------------------
The "Beyond Token Limits" (Sebok et al. 2025) long-document CAP release, staged on
the cluster. We expect either

    <btl_dir>/train/train_long.csv  +  <btl_dir>/test/test_long.csv
    <btl_dir>/clean/pooled_clean_with_token_counts.csv   (or pooled_clean.csv)
    <pooled_csv>                                          (any single CSV)

with columns: text, label (0-indexed BTL code), and optionally label_cap (original
CAP code), language (english/hungarian/dutch/french/italian),
token_count_xlm_roberta_large. We pool everything and build our *own* stratified
split, so the paired length design is honoured (the same documents feed every
length bucket).

Fallbacks (logged): poltextlab_hf (a small HF CAP set — short text, logic only) and
synthetic (a seeded multilingual generator used by the local logic smoke test, no
torch / no network).

This module is torch-free: token counts come from a precomputed column, else the
XLM-R tokenizer if `transformers` is importable, else a word-count proxy (logged).
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common import (
    BTL_IDX_TO_CAP_CODE, CAP_CODES, CAP_CODE_TO_LABEL, CAP_LABEL_NAMES,
    LANG_NAME_TO_ISO, doc_id, ensure_dirs, expanduser_path, load_config, banner,
)

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")


# ════════════════════════════════════════════════════════════════════════════
# Source loading → a raw DataFrame with at least [text, label or label_cap, language]
# ════════════════════════════════════════════════════════════════════════════
def _read_csv_any(path: str) -> pd.DataFrame:
    for kw in ({}, {"sep": ";"}, {"encoding": "unicode_escape"}):
        try:
            return pd.read_csv(path, low_memory=False, **kw)
        except Exception:  # noqa: BLE001
            continue
    return pd.read_csv(path, low_memory=False, engine="python")


def _load_btl(cfg: dict) -> Tuple[pd.DataFrame, str]:
    d = cfg["data"]
    pooled = expanduser_path(d.get("pooled_csv"))
    if pooled and os.path.isfile(pooled):
        return _read_csv_any(pooled), f"btl:pooled_csv={pooled}"

    root = expanduser_path(d.get("btl_dir"))
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(
            f"BTL data dir not found: {root!r}. Set data.btl_dir / data.pooled_csv "
            f"(or $BTL_DIR) to the staged Beyond Token Limits CSVs.")

    # Prefer a pooled file with token counts; else concatenate the *_long splits.
    for name in ("clean/pooled_clean_with_token_counts.csv",
                 "clean/pooled_clean.csv", "pooled_clean_with_token_counts.csv",
                 "pooled_clean.csv"):
        p = os.path.join(root, name)
        if os.path.isfile(p):
            return _read_csv_any(p), f"btl:{p}"

    csvs = sorted(glob.glob(os.path.join(root, "**", "*long*.csv"), recursive=True))
    if not csvs:
        csvs = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found under {root}")
    frames = [_read_csv_any(p) for p in csvs]
    logger.info("  pooled %d BTL csvs: %s", len(csvs), [os.path.basename(c) for c in csvs])
    return pd.concat(frames, ignore_index=True), f"btl:dir={root} ({len(csvs)} csvs)"


def _load_poltextlab_hf(cfg: dict) -> Tuple[pd.DataFrame, str]:
    from datasets import load_dataset  # noqa: PLC0415
    ds = load_dataset("poltextlab/HU_Res_of_Parl_2025_CAP", split="test")
    df = ds.to_pandas().rename(columns={"complete_title": "text", "majortopic": "label_cap"})
    df["language"] = "hungarian"
    return df, "fallback:poltextlab/HU_Res_of_Parl_2025_CAP (titles, short — logic only)"


# ---- synthetic generator (logic smoke test; deterministic, torch-free) --------
_TOPIC_WORDS = {  # a few stem words per CAP code → label-correlated content
    1: "inflation deficit gdp fiscal monetary tax budget interest treasury",
    3: "hospital patient disease vaccine clinic medicine doctor health insurance",
    5: "union worker wage strike employment labor overtime workplace pension",
    7: "emission pollution climate forest wildlife conservation habitat water",
    9: "migrant visa asylum border refugee citizenship deportation immigrant",
    12: "court police crime sentence prosecutor felony arrest prison statute",
    16: "army defense missile soldier weapon military veteran navy troop",
    19: "treaty embassy diplomatic foreign sanction summit alliance nato war",
    20: "ministry parliament regulation agency oversight election bureau audit",
}
_FILLER = ("the of and to in for on with that this from these those which by as an be "
           "it its their our about into over under between through during after before")


def _synthetic(cfg: dict, n_per_cell: int = 14) -> Tuple[pd.DataFrame, str]:
    """Deterministic multilingual generator for the logic smoke test only.

    Topic words are *sparse* (low density) and each doc gets its own density, so
    short prefixes are ambiguous and accuracy genuinely rises with length — this
    exercises the slope / CI / gap math on non-constant data (not a real corpus).
    """
    rng = random.Random(20250629)
    langs = [l for l in cfg["data"]["languages"] if l in LANG_NAME_TO_ISO.values()] or ["en"]
    iso_to_name = {v: k for k, v in LANG_NAME_TO_ISO.items()}
    codes = list(_TOPIC_WORDS.keys())
    filler = _FILLER.split()
    rows = []
    for lang in langs:
        for code in codes:
            topic_vocab = _TOPIC_WORDS[code].split()
            for _ in range(n_per_cell):
                n_words = rng.randint(620, 900)            # > 512 token proxy
                density = rng.uniform(0.06, 0.14)          # sparse, per-doc signal strength
                toks = [(rng.choice(topic_vocab) if rng.random() < density
                         else rng.choice(filler)) for _ in range(n_words)]
                text = f"[{lang}] " + " ".join(toks)
                rows.append({"text": text, "label_cap": code, "language": iso_to_name[lang]})
    df = pd.DataFrame(rows)
    return df, f"synthetic:{len(df)} docs ({len(langs)} langs x {len(codes)} topics)"


def _load_source(cfg: dict) -> Tuple[pd.DataFrame, str]:
    src = cfg["data"]["source"]
    if src in ("btl_local", "btl_csv"):
        try:
            return _load_btl(cfg)
        except FileNotFoundError as exc:
            logger.warning("  BTL source unavailable (%s) → falling back to synthetic", exc)
            return _synthetic(cfg)
    if src == "poltextlab_hf":
        return _load_poltextlab_hf(cfg)
    if src == "synthetic":
        return _synthetic(cfg)
    raise ValueError(f"Unknown data.source {src!r}")


# ════════════════════════════════════════════════════════════════════════════
# Cleaning / normalisation
# ════════════════════════════════════════════════════════════════════════════
def _clean_text(s: str) -> str:
    return _WS_RE.sub(" ", str(s)).strip()


def _to_cap_code(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Resolve each row's CAP major-topic code (1..23 / 999)."""
    d = cfg["data"]
    cap_col = d.get("label_cap_col")
    if cap_col and cap_col in df.columns:
        codes = pd.to_numeric(df[cap_col], errors="coerce")
    else:
        lab = pd.to_numeric(df[d["label_col"]], errors="coerce")
        if d.get("label_is_zero_indexed", True):
            codes = lab.map(BTL_IDX_TO_CAP_CODE)
        else:
            codes = lab
    return pd.to_numeric(codes, errors="coerce")


def _n_tokens(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, str]:
    """Token length under the XLM-R tokenizer (precomputed col → tokenizer → proxy)."""
    col = cfg["data"].get("token_count_col")
    if col and col in df.columns and df[col].notna().mean() > 0.5:
        return pd.to_numeric(df[col], errors="coerce"), f"precomputed column '{col}'"
    # try the real tokenizer (no torch needed, just transformers)
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
        ref = "FacebookAI/xlm-roberta-large"
        tok = AutoTokenizer.from_pretrained(ref)
        counts = df["text"].map(lambda t: len(tok.encode(t, add_special_tokens=True,
                                                          truncation=False)))
        return counts.astype(int), f"XLM-R tokenizer ({ref})"
    except Exception as exc:  # noqa: BLE001
        logger.warning("  token counting via tokenizer unavailable (%s) → word-count proxy", exc)
        # XLM-R subwords ~1.4x words for these languages; scale the proxy threshold accordingly
        return (df["text"].str.split().map(len) * 1.4).round().astype(int), "word-count proxy (x1.4)"


def load_clean(cfg: dict) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load + clean → DataFrame[doc_id, text, cap_code, cap_name, label, lang, language, n_tokens]."""
    d = cfg["data"]
    raw, provenance = _load_source(cfg)
    prov = {"source": provenance, "raw_rows": str(len(raw))}

    # text
    tcol = d["text_col"]
    if tcol not in raw.columns:
        raise KeyError(f"text column {tcol!r} not in {list(raw.columns)[:20]}")
    df = pd.DataFrame({"text": raw[tcol].map(_clean_text)})

    # label → CAP code
    df["cap_code"] = _to_cap_code(raw, cfg).values
    n0 = len(df)
    df = df[df["cap_code"].notna()]
    df["cap_code"] = df["cap_code"].astype(int)

    # drop No Policy Content + keep only the 21 target topics
    if d.get("drop_no_policy_content", True):
        df = df[df["cap_code"] != 999]
    df = df[df["cap_code"].isin(CAP_CODES)]

    # language → ISO
    lcol = d.get("lang_col")
    if lcol and lcol in raw.columns:
        lang_name = raw.loc[df.index, lcol].astype(str).str.lower().str.strip()
        df["language"] = lang_name
        df["lang"] = lang_name.map(LANG_NAME_TO_ISO).fillna(lang_name)
    else:
        logger.warning("  no language column %r → tagging all rows 'unk'", lcol)
        df["language"], df["lang"] = "unknown", "unk"
    keep = set(d["languages"])
    df = df[df["lang"].isin(keep)]

    # drop empty / trivially short, dedup
    df = df[df["text"].str.len() >= int(d.get("min_chars", 20))]
    if d.get("dedup", True):
        df = df.drop_duplicates(subset=["text"])

    # token length + the >=512 filter (paired-design gate)
    df = df.reset_index(drop=True)
    df["n_tokens"], tok_method = _n_tokens(df, cfg)
    prov["token_count_method"] = tok_method
    min_tok = int(cfg["min_tokens"])
    before = len(df)
    df = df[df["n_tokens"] >= min_tok]
    prov["kept_ge_min_tokens"] = f"{len(df)}/{before} (>= {min_tok} tokens via {tok_method})"

    # optional down-cap of dominant (lang x label) cells
    cap = d.get("max_per_class_lang")
    if cap:
        df = (df.groupby(["lang", "cap_code"], group_keys=False)
                .apply(lambda g: g.sample(min(len(g), int(cap)), random_state=cfg["split"]["seed"])))

    # finalise labels / ids
    df["label"] = df["cap_code"].map(CAP_CODE_TO_LABEL).astype(int)
    df["cap_name"] = df["cap_code"].map(CAP_LABEL_NAMES)
    df["doc_id"] = df["text"].map(doc_id)
    df = df.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)

    prov["clean_rows"] = str(len(df))
    if len(df) == 0:
        raise RuntimeError("No documents survived cleaning/filtering — check data.source / min_tokens.")
    cols = ["doc_id", "text", "cap_code", "cap_name", "label", "lang", "language", "n_tokens"]
    return df[cols], prov


# ════════════════════════════════════════════════════════════════════════════
# Stratified split by (language x label) — robust to tiny strata
# ════════════════════════════════════════════════════════════════════════════
def make_split(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    seed = int(cfg["split"]["seed"])
    test_size = float(cfg["data"]["test_size"])
    by = cfg["split"].get("stratify_by", ["language", "label"])
    rng = random.Random(seed)
    split = pd.Series("train", index=df.index)
    for _, idx in df.groupby([df[c] if c in df else df["lang"] for c in by]).groups.items():
        idx = list(idx)
        rng.shuffle(idx)
        if len(idx) < 2:
            continue  # singleton stratum stays in train
        n_test = max(1, int(round(len(idx) * test_size)))
        n_test = min(n_test, len(idx) - 1)  # keep >=1 in train
        for i in idx[:n_test]:
            split.at[i] = "test"
    df = df.copy()
    df["split"] = split.values
    return df


# ════════════════════════════════════════════════════════════════════════════
# Sanity reporting
# ════════════════════════════════════════════════════════════════════════════
def summarize(df: pd.DataFrame, cfg: dict) -> None:
    banner("Dataset sanity checks")
    print(f"  total docs: {len(df)}   classes present: {df['label'].nunique()}/{len(CAP_CODES)}")
    print(f"  languages : {dict(df['lang'].value_counts())}")
    if "split" in df.columns:
        print(f"  split     : {dict(df['split'].value_counts())}")
    print(f"  n_tokens  : min={df['n_tokens'].min()} median={int(df['n_tokens'].median())} "
          f"max={df['n_tokens'].max()}")
    print("\n  per-language x major-topic counts:")
    tab = (df.assign(topic=df["cap_code"].astype(str) + " " + df["cap_name"])
             .pivot_table(index="topic", columns="lang", values="doc_id",
                          aggfunc="count", fill_value=0, margins=True, margins_name="ALL"))
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(tab.to_string())
    # label balance
    bal = df["cap_name"].value_counts(normalize=True).round(3)
    print(f"\n  most/least frequent class share: {bal.iloc[0]:.3f} / {bal.iloc[-1]:.3f}")
    if "split" in df.columns:
        per = df.groupby(["split", "lang"])["doc_id"].count().unstack(fill_value=0)
        print("\n  docs per (split x lang):")
        print(per.to_string())


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════
def _subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified down-sample to ~n docs (by lang x label), for the smoke run."""
    if n is None or n >= len(df):
        return df
    groups = list(df.groupby(["lang", "label"]))
    per = max(1, n // max(1, len(groups)))
    parts = [g.sample(min(len(g), per), random_state=seed) for _, g in groups]
    out = pd.concat(parts).reset_index(drop=True)
    return out


def build_dataset(cfg: dict, save: bool = True, subsample_n: Optional[int] = None) -> pd.DataFrame:
    df, prov = load_clean(cfg)
    if subsample_n:
        df = _subsample(df, subsample_n, int(cfg["split"]["seed"]))
        prov["subsampled_to"] = str(len(df))
    df = make_split(df, cfg)
    banner("Resolved dataset source / provenance")
    for k, v in prov.items():
        print(f"  {k:22}: {v}")
    summarize(df, cfg)
    if save:
        ensure_dirs(cfg)
        out = Path(cfg["paths"]["data_dir"]) / "cap_dataset.csv"
        df.to_csv(out, index=False)
        print(f"\n  dataset → {out}")
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", default=None, help="override data.source")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.source:
        cfg["data"]["source"] = args.source
    build_dataset(cfg)


if __name__ == "__main__":
    main()
