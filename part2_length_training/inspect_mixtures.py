"""What is actually inside each pool and each training mix, per window size k.

make_mixtures.py takes a prefix of the shuffled pool, so a mix inherits the pool's own
k distribution, and windows() (stride 1) yields n-k+1 windows per document, so short
windows are more numerous. This measures both on whichever directories exist.

  python -m part2_length_training.inspect_mixtures
  python -m part2_length_training.inspect_mixtures --data_dirs ~/scratch/wmt_length_data_v2 \
      --mix_subdirs mixes_pure --output ~/scratch/wmt_length_data_v2/mixes_pure/composition.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from common.paths import SCRATCH
from part2_length_training.models import MixManifest


def pool_counts(data_dir: Path) -> dict:
    """rows per (origin, split, k), read straight from the pool CSVs."""
    out: dict = {}
    for split in ("train", "val"):
        for f in sorted(data_dir.glob(f"*_{split}.csv")):
            origin = f.name.split("_")[0]  # sent | agg | native
            if f.name.startswith("all_") or origin not in ("sent", "agg", "native"):
                continue
            # a few native rows have shifted fields (NaN k): counted and printed, not swallowed
            k = pd.to_numeric(pd.read_csv(f, usecols=["k"], low_memory=False)["k"], errors="coerce")
            n_bad = int(k.isna().sum())
            if n_bad:
                print(f"  ! {f.name}: {n_bad} row(s) with an unparseable k (malformed CSV line) - excluded")
            out.setdefault(origin, {}).setdefault(split, Counter()).update(int(v) for v in k.dropna())
    return {o: {s: {str(k): int(n) for k, n in sorted(c.items())} for s, c in by_split.items()}
            for o, by_split in out.items()}


def mix_counts(data_dir: Path, subdir: str) -> dict:
    man = data_dir / subdir / "manifest.json"
    if not man.exists():
        return {}
    m = MixManifest.model_validate_json(man.read_text())
    return {name: {"origin": {"sent": e.counts.sent, "agg": e.counts.agg, "native": e.counts.native},
                   "per_k": e.counts.per_k, "per_origin_lp": e.counts.per_origin_lp}
            for name, e in m.mixes.items()}


def share(d: dict) -> str:
    tot = sum(d.values()) or 1
    return "  ".join(f"k={k}: {n:>7,} ({n / tot:5.1%})" for k, n in d.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dirs", nargs="+", default=[str(SCRATCH / "wmt_length_data_v2")])
    ap.add_argument("--mix_subdirs", nargs="+", default=["mixes", "mixes_pure"],
                    help="mix directories to look for under each data dir")
    ap.add_argument("--output", default=None, help="default: <first data dir>/<first mix subdir>/composition.json")
    args = ap.parse_args()

    report: dict = {}
    for raw in args.data_dirs:
        d = Path(raw).expanduser()
        if not d.exists():
            print(f"- {d}: absent, skipped")
            continue
        print(f"\n== {d}")
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
            print(f"  (no manifest.json under {args.mix_subdirs})")
        report[str(d)] = {"pools": pools, "mixes": all_mixes}

    out = (Path(args.output).expanduser() if args.output
           else Path(args.data_dirs[0]).expanduser() / args.mix_subdirs[0] / "composition.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
