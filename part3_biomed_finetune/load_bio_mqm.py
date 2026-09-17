"""Bio-MQM (amazon-science/bio-mqm-dataset) -> COMET training CSVs.

Clones or updates the dataset, sums MQM penalties per segment (MQM_WEIGHTS),
averages annotators, z-scores then sigmoids per language pair, and splits by
document with doc_id_splits.json (dev docs -> train, test docs -> val). Rows
without a reference are dropped. Writes <lp>_train.csv / <lp>_val.csv and the
concatenated all_train.csv / all_val.csv (columns = BioMqmRow).

    python -m part3_biomed_finetune.load_bio_mqm [--output_dir ~/scratch/bio_mqm] [--lang_pairs en-fr fr-en]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from common.paths import SCRATCH
from part3_biomed_finetune.models import LANG_PAIRS, MQM_WEIGHTS, BioMqmRow

REPO_URL = "https://github.com/amazon-science/bio-mqm-dataset"
COLUMNS = list(BioMqmRow.model_fields)


def clone_or_pull(repo_url: str, local_dir: Path) -> None:
    if (local_dir / ".git").exists():
        print(f"updating {local_dir}")
        subprocess.run(["git", "-C", str(local_dir), "pull", "--quiet"], check=True)
    else:
        print(f"cloning {repo_url} -> {local_dir}")
        subprocess.run(["git", "clone", "--quiet", repo_url, str(local_dir)], check=True)


def _dir_name(lang_pair: str) -> str:
    src, tgt = lang_pair.split("-")
    return f"{src}2{tgt}"


def find_system_files(repo_dir: Path, lang_pair: str) -> list[Path]:
    """Non-reference MT-system JSON files of a language pair: data/v2 first, then data/v1."""
    dp = _dir_name(lang_pair)
    files: list[Path] = []
    v2 = repo_dir / "data" / "v2" / "target" / "main_phase" / "round_two" / dp
    if v2.exists():
        files += [f for f in v2.glob("*.json") if not f.stem.startswith("reference")]
    v1_root = repo_dir / "data" / "v1" / "target" / "main_phase" / "round_one"
    for batch in v1_root.glob("batch*"):
        for lp_dir in batch.glob(f"medline_{dp}*"):
            files += [f for f in lp_dir.rglob("*.json")
                      if not f.stem.startswith("reference") and not f.name.startswith(".")]
    return files


def find_reference_file(repo_dir: Path, lang_pair: str) -> Path | None:
    """reference.json of a language pair (v2 preferred over v1)."""
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


def _penalty(errors: list[dict]) -> float:
    return sum(MQM_WEIGHTS.get(e.get("severity", "minor").lower(), -1.0) for e in errors)


def load_annotations(files: list[Path]) -> pd.DataFrame:
    records = []
    for f in files:
        try:
            segs = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  warning: skipping {f.name}: {exc}")
            continue
        if not isinstance(segs, list):
            continue
        for seg in segs:
            records.append({
                "system": seg.get("MT_Engine", f.stem),
                "doc_id": seg.get("DOC_ID", ""),
                "seg_id": str(seg.get("SEG_ID", "")),
                "annotator": seg.get("Annotator_ID", ""),
                "source": seg.get("source", ""),
                "target": seg.get("target", ""),
                "penalty": _penalty(seg.get("target_errors", [])),
            })
    return pd.DataFrame(records)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean MQM penalty across annotators per (system, doc_id, seg_id)."""
    return (df.groupby(["system", "doc_id", "seg_id"], sort=False)
              .agg(source=("source", "first"), target=("target", "first"), mqm_score=("penalty", "mean"))
              .reset_index())


def normalise(scores: pd.Series) -> pd.Series:
    """Z-score then sigmoid -> (0, 1); a segment without errors lands near 0.88."""
    mu, sigma = scores.mean(), scores.std()
    if sigma == 0:
        return pd.Series(np.full(len(scores), 0.5), index=scores.index)
    return 1.0 / (1.0 + np.exp(-(scores - mu) / sigma))


def build_ref_map(ref_file: Path | None) -> dict[tuple[str, str], str]:
    if ref_file is None:
        return {}
    segs = json.loads(ref_file.read_text(encoding="utf-8"))
    return {(s["DOC_ID"], str(s["SEG_ID"])): s["target"] for s in segs}


def process_lang_pair(repo_dir: Path, lang_pair: str, doc_splits: dict, output_dir: Path) -> dict[str, int]:
    """Write <lp>_train.csv and <lp>_val.csv; return the row count of each split written."""
    system_files = find_system_files(repo_dir, lang_pair)
    if not system_files:
        print(f"  [{lang_pair}] no annotation files, skipping")
        return {}
    print(f"  [{lang_pair}] {len(system_files)} system file(s)")
    df = load_annotations(system_files)
    if df.empty:
        print(f"  [{lang_pair}] no records, skipping")
        return {}

    agg = aggregate(df)
    agg["score"] = normalise(agg["mqm_score"])
    agg["lp"] = lang_pair
    ref_map = build_ref_map(find_reference_file(repo_dir, lang_pair))

    splits = doc_splits.get(lang_pair, {})
    split_docs = {"train": set(splits.get("dev", [])), "val": set(splits.get("test", []))}
    counts: dict[str, int] = {}
    for split_name, doc_set in split_docs.items():
        if not doc_set:
            continue
        subset = agg[agg["doc_id"].isin(doc_set)].copy().reset_index(drop=True)
        if subset.empty:
            print(f"  [{lang_pair}] {split_name}: 0 rows")
            continue
        subset["ref"] = subset.apply(lambda r: ref_map.get((r["doc_id"], r["seg_id"]), ""), axis=1)
        rows = (subset.rename(columns={"source": "src", "target": "mt"})[COLUMNS]
                      .query("ref != ''").reset_index(drop=True))
        out = output_dir / f"{lang_pair}_{split_name}.csv"
        rows.to_csv(out, index=False)
        counts[split_name] = len(rows)
        print(f"  [{lang_pair}] {split_name}: {len(rows):,} rows -> {out}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", default=str(SCRATCH / "bio_mqm"))
    ap.add_argument("--repo_dir", default=None, help="dataset clone (default: <output_dir>/bio-mqm-dataset)")
    ap.add_argument("--lang_pairs", nargs="+", default=LANG_PAIRS)
    args = ap.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(args.repo_dir).expanduser() if args.repo_dir else output_dir / "bio-mqm-dataset"
    clone_or_pull(REPO_URL, repo_dir)
    doc_splits = json.loads((repo_dir / "data" / "doc_id_splits.json").read_text())

    written: dict[str, list[Path]] = {"train": [], "val": []}
    for lp in args.lang_pairs:
        for split_name in process_lang_pair(repo_dir, lp, doc_splits, output_dir):
            written[split_name].append(output_dir / f"{lp}_{split_name}.csv")

    for split_name, paths in written.items():
        if not paths:
            print(f"no {split_name} rows: all_{split_name}.csv not written")
            continue
        combined = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        combined.to_csv(output_dir / f"all_{split_name}.csv", index=False)
        print(f"all_{split_name}.csv: {len(combined):,} rows from {len(paths)} pair(s)")


if __name__ == "__main__":
    main()
