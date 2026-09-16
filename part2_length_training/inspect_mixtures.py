#!/usr/bin/env python3
"""What is actually inside each pool and each training mix, per window size k.

`make_mixtures.py` claims the aggregated half is "internally even over k", but
`take()` is a prefix of the shuffled pool, so the mix inherits the pool's own
k distribution — and `windows()` (stride 1) yields n−k+1 windows per document,
so short windows are more numerous. This script measures both, on whichever
data directories exist.

  python experiments/length_training/report_mix_composition.py
  python experiments/length_training/report_mix_composition.py \
      --data_dirs ~/scratch/wmt_length_data ~/scratch/wmt_length_data_v2 \
      --output ~/mix_composition.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def pool_counts(data_dir: Path) -> dict:
    """rows per (origin, split, k), read straight from the pool CSVs."""
    out: dict = {}
    for split in ("train", "val"):
        for f in sorted(data_dir.glob(f"*_{split}.csv")):
            if f.name.startswith("all_"):
                continue
            origin = f.name.split("_")[0]          # sent | agg | native
            if origin not in ("sent", "agg", "native"):
                continue
            # A handful of native documents embed newlines that survive the CSV
            # quoting badly enough to shift a row's fields (2 of 254,497 in the
            # v2 pools). Those rows carry a NaN k, are dropped from the mixes by
            # the groupby in make_mixtures.py, and must not crash the report —
            # but they are counted and printed, not swallowed.
            k = pd.to_numeric(pd.read_csv(f, usecols=["k"], low_memory=False)["k"],
                              errors="coerce")
            n_bad = int(k.isna().sum())
            if n_bad:
                print(f"  ! {f.name}: {n_bad} row(s) with an unparseable k "
                      f"(malformed CSV line) — excluded from the counts below")
            c = Counter(int(v) for v in k.dropna())
            d = out.setdefault(origin, {}).setdefault(split, Counter())
            d.update(c)
    return {o: {s: {str(k): int(n) for k, n in sorted(c.items())}
                for s, c in by_split.items()}
            for o, by_split in out.items()}


def mix_counts(data_dir: Path, subdir: str = "mixes") -> dict:
    man = data_dir / subdir / "manifest.json"
    if not man.exists():
        return {}
    m = json.load(open(man))
    return {name: {"origin": {k: v for k, v in mix["counts"].items()
                              if k in ("sent", "agg", "native")},
                   "per_k": mix["counts"].get("per_k", {}),
                   "per_origin_lp": mix["counts"].get("per_origin_lp", {})}
            for name, mix in m.get("mixes", {}).items()}


def share(d: dict) -> str:
    tot = sum(d.values()) or 1
    return "  ".join(f"k={k}: {n:>7,} ({n / tot:5.1%})" for k, n in d.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dirs", nargs="+",
                    default=["~/scratch/wmt_length_data", "~/scratch/wmt_length_data_v2"])
    ap.add_argument("--mix_subdirs", nargs="+", default=["mixes", "mixes_pure"],
                    help="Mix directories to look for under each data dir. "
                         "`mixes_pure` is the four-arm grid built with "
                         "make_mixtures.py --total_policy pure.")
    ap.add_argument("--output", default="~/mix_composition.json")
    args = ap.parse_args()

    report: dict = {}
    for raw in args.data_dirs:
        d = Path(raw).expanduser()
        if not d.exists():
            print(f"— {d}: absent, ignoré")
            continue
        print(f"\n══ {d}")
        pools = pool_counts(d)
        for origin in ("sent", "agg", "native"):
            for split in ("train", "val"):
                c = pools.get(origin, {}).get(split)
                if c:
                    print(f"  pool {origin:7} {split:5} {share(c)}")
        all_mixes = {}
        for sub in args.mix_subdirs:
            mixes = mix_counts(d, sub)
            if not mixes:
                continue
            for name in sorted(mixes):
                m = mixes[name]
                print(f"  mix  [{sub}] {name:12} {m['origin']}")
                print(f"       {'':>{len(sub) + 3}} {'':12} {share(m['per_k'])}")
            all_mixes[sub] = mixes
        if not all_mixes:
            print(f"  (aucun manifest.json sous {args.mix_subdirs})")
        report[str(d)] = {"pools": pools, "mixes": all_mixes}

    out = Path(args.output).expanduser()
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
