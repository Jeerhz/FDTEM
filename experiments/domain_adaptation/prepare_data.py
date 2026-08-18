#!/usr/bin/env python3
"""
Prepare Amazon Bio-MQM dataset for COMET finetuning.

Reads the JSON annotation files from the cloned Bio-MQM repository,
aggregates MQM error penalties into per-segment scores, splits into
train (dev docs) / val (test docs) using doc_id_splits.json, and
writes COMET-ready CSV files (columns: src, mt, ref, score).

Usage:
    python experiments/domain_adaptation/prepare_data.py \
        --output_dir ~/scratch/bio_mqm \
        --lang_pairs en-fr fr-en \
        [--write_combined]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np  # type: ignore[import-untyped,import-not-found]
import pandas as pd  # type: ignore[import-untyped,import-not-found]

REPO_URL = "https://github.com/amazon-science/bio-mqm-dataset"

MQM_WEIGHTS: dict[str, float] = {
    "minor":    -1.0,
    "major":    -5.0,
    "critical": -25.0,
    "no-error":  0.0,
    "neutral":   0.0,
}

ALL_LANG_PAIRS = [
    "en-de", "de-en", "en-es", "es-en",
    "en-fr", "fr-en", "en-ru", "ru-en",
    "en-zh", "zh-en",
]


# ── repo management ───────────────────────────────────────────────────────────

def clone_or_pull(repo_url: str, local_dir: Path) -> None:
    if (local_dir / ".git").exists():
        print(f"Updating existing repo at {local_dir}")
        subprocess.run(["git", "-C", str(local_dir), "pull", "--quiet"], check=True)
    else:
        print(f"Cloning {repo_url} → {local_dir}")
        subprocess.run(["git", "clone", "--quiet", repo_url, str(local_dir)], check=True)


# ── file discovery ────────────────────────────────────────────────────────────

def _dir_name(lang_pair: str) -> str:
    """'en-fr'  →  'en2fr'"""
    src, tgt = lang_pair.split("-")
    return f"{src}2{tgt}"


def find_system_files(repo_dir: Path, lang_pair: str) -> list[Path]:
    """
    Return all non-reference MT-system JSON files for a language pair,
    searching both data/v2 (preferred) and data/v1.
    """
    dp = _dir_name(lang_pair)
    files: list[Path] = []

    # v2: flat directory per language pair
    v2 = repo_dir / "data" / "v2" / "target" / "main_phase" / "round_two" / dp
    if v2.exists():
        files += [f for f in v2.glob("*.json") if not f.stem.startswith("reference")]

    # v1: nested batch/medline_<lp>* directories
    v1_root = repo_dir / "data" / "v1" / "target" / "main_phase" / "round_one"
    for batch in v1_root.glob("batch*"):
        for lp_dir in batch.glob(f"medline_{dp}*"):
            files += [
                f for f in lp_dir.rglob("*.json")
                if not f.stem.startswith("reference") and not f.name.startswith(".")
            ]

    return files


def find_reference_file(repo_dir: Path, lang_pair: str) -> Path | None:
    """Return path to reference.json (v2 preferred over v1)."""
    dp = _dir_name(lang_pair)

    ref = repo_dir / "data" / "v2" / "target" / "main_phase" / "round_two" / dp / "reference.json"
    if ref.exists():
        return ref

    v1_root = repo_dir / "data" / "v1" / "target" / "main_phase" / "round_one"
    for batch in v1_root.glob("batch*"):
        for lp_dir in batch.glob(f"medline_{dp}*"):
            for ref in lp_dir.rglob("reference.json"):
                return ref
    return None


# ── data loading & scoring ────────────────────────────────────────────────────

def _penalty(errors: list[dict]) -> float:
    return sum(MQM_WEIGHTS.get(e.get("severity", "minor").lower(), -1.0) for e in errors)


def load_annotations(files: list[Path]) -> pd.DataFrame:
    records = []
    for f in files:
        try:
            segs = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  Warning: skipping {f.name}: {exc}")
            continue
        if not isinstance(segs, list):
            continue
        for seg in segs:
            records.append({
                "system":    seg.get("MT_Engine", f.stem),
                "doc_id":    seg.get("DOC_ID", ""),
                "seg_id":    str(seg.get("SEG_ID", "")),
                "annotator": seg.get("Annotator_ID", ""),
                "source":    seg.get("source", ""),
                "target":    seg.get("target", ""),
                "penalty":   _penalty(seg.get("target_errors", [])),
            })
    return pd.DataFrame(records)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Average MQM penalties across annotators per (system, doc_id, seg_id)."""
    return (
        df.groupby(["system", "doc_id", "seg_id"], sort=False)
        .agg(source=("source", "first"),
             target=("target", "first"),
             mqm_score=("penalty", "mean"))
        .reset_index()
    )


def normalise(scores: pd.Series) -> pd.Series:
    """Z-score then sigmoid → (0, 1). Score=0 (no errors) maps to ~0.88."""
    mu, sigma = scores.mean(), scores.std()
    if sigma == 0:
        return pd.Series(np.full(len(scores), 0.5), index=scores.index)
    return 1.0 / (1.0 + np.exp(-(scores - mu) / sigma))


def build_ref_map(ref_file: Path | None) -> dict[tuple[str, str], str]:
    if ref_file is None:
        return {}
    segs = json.loads(ref_file.read_text(encoding="utf-8"))
    return {(s["DOC_ID"], str(s["SEG_ID"])): s["target"] for s in segs}


# ── per-language-pair pipeline ────────────────────────────────────────────────

def process_lang_pair(
    repo_dir: Path,
    lang_pair: str,
    doc_splits: dict,
    output_dir: Path,
) -> tuple[int, int]:

    system_files = find_system_files(repo_dir, lang_pair)
    if not system_files:
        print(f"  [{lang_pair}] No annotation files found — skipping.")
        return 0, 0

    print(f"  [{lang_pair}] Found {len(system_files)} system file(s).")
    df = load_annotations(system_files)
    if df.empty:
        print(f"  [{lang_pair}] No records — skipping.")
        return 0, 0

    agg = aggregate(df)
    agg["score"] = normalise(agg["mqm_score"])
    ref_map = build_ref_map(find_reference_file(repo_dir, lang_pair))

    # doc_id_splits.json: dev docs → train split, test docs → val split
    splits = doc_splits.get(lang_pair, {})
    split_map = {"train": set(splits.get("dev", [])),
                 "val":   set(splits.get("test", []))}

    counts: dict[str, int] = {}
    for split_name, doc_set in split_map.items():
        if not doc_set:
            continue
        subset = agg[agg["doc_id"].isin(doc_set)].copy().reset_index(drop=True)
        if subset.empty:
            print(f"  [{lang_pair}] {split_name}: 0 rows.")
            continue

        subset["ref"] = subset.apply(
            lambda r: ref_map.get((r["doc_id"], r["seg_id"]), ""), axis=1
        )
        comet_df = (
            subset.rename(columns={"source": "src", "target": "mt"})
                  [["src", "mt", "ref", "score"]]
                  .query("ref != ''")
                  .reset_index(drop=True)
        )
        out = output_dir / f"{lang_pair}_{split_name}.csv"
        comet_df.to_csv(out, index=False)
        counts[split_name] = len(comet_df)
        print(f"  [{lang_pair}] {split_name}: {len(comet_df):,} rows → {out}")

    return counts.get("train", 0), counts.get("val", 0)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Bio-MQM JSON annotations to COMET CSV format."
    )
    parser.add_argument(
        "--output_dir", default="~/scratch/bio_mqm",
        help="Directory for processed CSV files (default: ~/scratch/bio_mqm).",
    )
    parser.add_argument(
        "--repo_dir", default=None,
        help="Local path for the Bio-MQM repo clone. "
             "Defaults to <output_dir>/bio-mqm-dataset.",
    )
    parser.add_argument(
        "--lang_pairs", nargs="+", default=ALL_LANG_PAIRS,
        help="Language pairs to process, e.g. --lang_pairs en-fr fr-en",
    )
    parser.add_argument(
        "--write_combined", action="store_true",
        help="Also write merged all_train.csv / all_val.csv.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = (
        Path(args.repo_dir).expanduser() if args.repo_dir
        else output_dir / "bio-mqm-dataset"
    )
    clone_or_pull(REPO_URL, repo_dir)

    doc_splits: dict = json.loads(
        (repo_dir / "data" / "doc_id_splits.json").read_text()
    )

    total_train, total_val = 0, 0
    train_paths: list[Path] = []
    val_paths:   list[Path] = []

    for lp in args.lang_pairs:
        n_tr, n_va = process_lang_pair(repo_dir, lp, doc_splits, output_dir)
        total_train += n_tr
        total_val   += n_va
        if n_tr: train_paths.append(output_dir / f"{lp}_train.csv")
        if n_va: val_paths.append(output_dir / f"{lp}_val.csv")

    if args.write_combined:
        if train_paths:
            combined = pd.concat([pd.read_csv(p) for p in train_paths], ignore_index=True)
            combined.to_csv(output_dir / "all_train.csv", index=False)
            print(f"\nCombined train: {len(combined):,} rows → {output_dir}/all_train.csv")
        if val_paths:
            combined = pd.concat([pd.read_csv(p) for p in val_paths], ignore_index=True)
            combined.to_csv(output_dir / "all_val.csv", index=False)
            print(f"Combined val:   {len(combined):,} rows → {output_dir}/all_val.csv")

    print(f"\nDone.  Total train={total_train:,}  val={total_val:,}")

    # Print ready-to-use YAML snippet
    print("\n── Paste into your training YAML ──────────────────────────")
    if train_paths:
        print("    train_data:")
        for p in train_paths:
            print(f"      - {p}")
    if val_paths:
        print("    validation_data:")
        for p in val_paths:
            print(f"      - {p}")


if __name__ == "__main__":
    main()
