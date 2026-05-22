#!/usr/bin/env python3
"""
Prepare Amazon Bio-MQM dataset for COMET finetuning.

Downloads the Bio-MQM dataset from GitHub, aggregates MQM error annotations
into per-segment scores, normalises them to [0, 1], and writes COMET-ready
CSV files (columns: src, mt, ref, score).

Usage:
    python scripts/prepare_bio_mqm_data.py \
        --output_dir data/bio_mqm \
        [--lang_pairs en-de de-en en-es ...]  \
        [--no_ref]   # omit ref column for QE-style training
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── MQM penalty weights (standard WMT convention) ────────────────────────────
MQM_WEIGHTS = {
    "no-error": 0,
    "neutral":  0,
    "minor":   -1,
    "major":   -5,
    "critical": -25,
}

# All language pairs present in the Bio-MQM release
ALL_LANG_PAIRS = [
    "de-en", "en-de",
    "en-es", "es-en",
    "en-fr", "fr-en",
    "en-ru", "ru-en",
    "en-zh", "zh-en",
    "en-pt",
]

REPO_URL = "https://github.com/amazon-science/bio-mqm-dataset"


# ── helpers ───────────────────────────────────────────────────────────────────

def clone_or_pull(repo_url: str, local_dir: Path) -> None:
    if (local_dir / ".git").exists():
        print(f"Updating existing repo at {local_dir}")
        subprocess.run(["git", "-C", str(local_dir), "pull", "--quiet"], check=True)
    else:
        print(f"Cloning {repo_url} → {local_dir}")
        subprocess.run(["git", "clone", "--quiet", repo_url, str(local_dir)], check=True)


def mqm_penalty(severity: str) -> float:
    """Return numeric penalty for an MQM severity label."""
    severity = str(severity).lower().strip()
    return MQM_WEIGHTS.get(severity, MQM_WEIGHTS["minor"])


def load_mqm_tsv(path: Path) -> pd.DataFrame:
    """
    Load a Bio-MQM TSV file.

    Expected columns (Bio-MQM v1 schema):
        system  doc  docID  seg_id  rater  source  target  category  severity
    Returns a DataFrame with those columns.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"system", "seg_id", "rater", "source", "target", "severity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}. Got: {list(df.columns)}")

    return df


def aggregate_mqm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse span-level annotations to one row per (system, seg_id, rater).

    Returns a DataFrame with columns:
        system, seg_id, source, target, mqm_score
    where mqm_score is the sum of MQM penalties across all annotated spans
    (0 = perfect, more negative = worse).
    """
    df = df.copy()
    df["penalty"] = df["severity"].apply(mqm_penalty)

    agg = (
        df.groupby(["system", "seg_id", "rater"], sort=False)
        .agg(
            source=("source", "first"),
            target=("target", "first"),
            mqm_score=("penalty", "sum"),
        )
        .reset_index()
    )

    # Average across raters for the same (system, seg_id)
    agg = (
        agg.groupby(["system", "seg_id"], sort=False)
        .agg(
            source=("source", "first"),
            target=("target", "first"),
            mqm_score=("mqm_score", "mean"),
        )
        .reset_index()
    )

    return agg


def normalise_scores(scores: pd.Series) -> pd.Series:
    """
    Z-score normalise MQM scores, then apply a sigmoid so values land in (0, 1).
    A segment with no errors (score=0) maps to ~0.88; lower is worse.
    """
    mu, sigma = scores.mean(), scores.std()
    if sigma == 0:
        return pd.Series(np.full(len(scores), 0.5), index=scores.index)
    z = (scores - mu) / sigma
    return 1.0 / (1.0 + np.exp(-z))


def build_comet_df(
    mqm_df: pd.DataFrame,
    ref_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Convert aggregated MQM data to COMET training format.

    Args:
        mqm_df:  Output of aggregate_mqm_scores().
        ref_df:  Optional DataFrame with columns (seg_id, reference) for
                 reference-based training.  Pass None for QE-style training.
    Returns:
        DataFrame with columns: src, mt, [ref,] score
    """
    comet = pd.DataFrame(
        {
            "src":   mqm_df["source"],
            "mt":    mqm_df["target"],
            "score": normalise_scores(mqm_df["mqm_score"]),
        }
    )

    if ref_df is not None:
        id_to_ref = ref_df.set_index("seg_id")["reference"].to_dict()
        comet["ref"] = mqm_df["seg_id"].map(id_to_ref)
        comet = comet[["src", "mt", "ref", "score"]]
    else:
        comet = comet[["src", "mt", "score"]]

    comet = comet.dropna().reset_index(drop=True)
    return comet


def find_split_files(repo_dir: Path, lang_pair: str, split: str):
    """
    Locate TSV annotation files for a given language pair and split.

    The Bio-MQM repo uses a structure like:
        data/<lang_pair>/<split>/mqm_<lang_pair>_<split>.tsv
    or flat files directly under data/.
    """
    candidates = list(repo_dir.rglob(f"*{lang_pair}*{split}*.tsv")) + \
                 list(repo_dir.rglob(f"*{split}*{lang_pair}*.tsv"))
    return candidates


def process_lang_pair(
    repo_dir: Path,
    lang_pair: str,
    output_dir: Path,
    include_ref: bool,
) -> tuple[int, int]:
    """
    Process one language pair.  Returns (n_train, n_val).
    """
    train_files = find_split_files(repo_dir, lang_pair, "dev")
    val_files   = find_split_files(repo_dir, lang_pair, "test")

    if not train_files and not val_files:
        print(f"  [{lang_pair}] No annotation files found — skipping.")
        return 0, 0

    def load_and_aggregate(files):
        frames = [load_mqm_tsv(f) for f in files]
        combined = pd.concat(frames, ignore_index=True)
        return aggregate_mqm_scores(combined)

    results = {}
    for split_name, files in [("train", train_files), ("val", val_files)]:
        if not files:
            print(f"  [{lang_pair}] No {split_name} files.")
            continue
        agg = load_and_aggregate(files)
        comet_df = build_comet_df(agg, ref_df=None if not include_ref else None)
        out_path = output_dir / f"{lang_pair}_{split_name}.csv"
        comet_df.to_csv(out_path, index=False)
        results[split_name] = len(comet_df)
        print(f"  [{lang_pair}] {split_name}: {len(comet_df):,} rows → {out_path}")

    return results.get("train", 0), results.get("val", 0)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Bio-MQM and convert to COMET CSV format."
    )
    parser.add_argument(
        "--output_dir", default="data/bio_mqm",
        help="Where to save processed CSV files (default: data/bio_mqm).",
    )
    parser.add_argument(
        "--repo_dir", default=None,
        help="Local path for the Bio-MQM repo clone.  "
             "Defaults to <output_dir>/bio-mqm-dataset.",
    )
    parser.add_argument(
        "--lang_pairs", nargs="+", default=ALL_LANG_PAIRS,
        help="Language pairs to process.  Defaults to all available pairs.",
    )
    parser.add_argument(
        "--no_ref", action="store_true",
        help="Omit the 'ref' column (QE-style training).",
    )
    parser.add_argument(
        "--write_combined", action="store_true",
        help="Also write combined train/val CSVs that pool all language pairs.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = Path(args.repo_dir) if args.repo_dir else output_dir / "bio-mqm-dataset"
    clone_or_pull(REPO_URL, repo_dir)

    total_train, total_val = 0, 0
    all_train_paths, all_val_paths = [], []

    for lp in args.lang_pairs:
        n_tr, n_va = process_lang_pair(
            repo_dir, lp, output_dir, include_ref=not args.no_ref
        )
        total_train += n_tr
        total_val   += n_va
        if n_tr:
            all_train_paths.append(output_dir / f"{lp}_train.csv")
        if n_va:
            all_val_paths.append(output_dir / f"{lp}_val.csv")

    if args.write_combined and all_train_paths:
        combined_train = pd.concat(
            [pd.read_csv(p) for p in all_train_paths], ignore_index=True
        )
        combined_train.to_csv(output_dir / "all_train.csv", index=False)
        print(f"\nCombined train: {len(combined_train):,} rows → {output_dir}/all_train.csv")

    if args.write_combined and all_val_paths:
        combined_val = pd.concat(
            [pd.read_csv(p) for p in all_val_paths], ignore_index=True
        )
        combined_val.to_csv(output_dir / "all_val.csv", index=False)
        print(f"Combined val:   {len(combined_val):,} rows → {output_dir}/all_val.csv")

    print(f"\nDone.  Total train={total_train:,}  val={total_val:,}")

    # Print a ready-to-use YAML snippet for the training config
    train_list = "\n      - ".join(str(p) for p in all_train_paths)
    val_list   = "\n      - ".join(str(p) for p in all_val_paths)
    print("\n── Paste into your training YAML ──────────────────────────")
    print(f"    train_data:\n      - {train_list}")
    print(f"    validation_data:\n      - {val_list}")


if __name__ == "__main__":
    main()
