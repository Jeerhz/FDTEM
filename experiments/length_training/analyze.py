#!/usr/bin/env python3
"""Results table for the length-training sweep, with bootstrap confidence.

  python experiments/length_training/analyze.py                    # table
  python experiments/length_training/analyze.py --bootstrap        # + 95% CI vs baseline

Columns: k=1 single sentences · k=2..6 aggregated windows · k=0 whole documents.
`held-out` files are the WMT23/24 paragraph test sets, never seen in training.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import pandas as pd

MIXES = ["frac000", "frac010", "frac020", "frac040", "frac060", "frac080",
         "frac100", "frac000nat", "frac000agg"]
FAMILIES = [("da", "COMET-DA (reference-based)"), ("qe", "CometKiwi QE (reference-free)")]


def block(per_file: dict, prefix: str, ks) -> dict:
    """Mean Kendall tau over the files whose name starts with `prefix`."""
    acc: dict = {}
    for name, per_k in per_file.items():
        if name.startswith("_") or not name.startswith(prefix):
            continue
        for k, v in per_k.items():
            if v.get("kendall") is not None:
                acc.setdefault(k, []).append(v["kendall"])
    return {k: (st.mean(acc[k]) if k in acc else None) for k in ks}


def fmt(v, w=6):
    return f"{v:{w}.3f}" if v is not None else " " * (w - 1) + "-"


def table(models: dict) -> None:
    ks = ("1", "2", "3", "4", "6")
    for fam, label in FAMILIES:
        base = f"{fam}-base"
        if base not in models:
            continue
        b_agg, b_nat = block(models[base], "wmt22-", ks), block(models[base], "wmt25-", ("0",))
        b_held = block(models[base], "heldout-", ("0",))
        print(f"\n{label}\n" + "-" * 96)
        head = " ".join(f"{'k='+k:>6}" for k in ks)
        print(f"{'arm':22}{head} {'docs':>8} {'held-out':>9} | {'Dk1':>7} {'Dheld':>7}")
        print(f"{'baseline':22}" + " ".join(fmt(b_agg[k]) for k in ks) +
              f" {fmt(b_nat['0'],8)} {fmt(b_held['0'],9)} |")
        for suffix, tag in (("", "encoder trained"), ("-frozen", "encoder frozen")):
            print(f"  [{tag}]")
            for mix in MIXES:
                name = f"{fam}-{mix}{suffix}"
                if name not in models:
                    continue
                a, n = block(models[name], "wmt22-", ks), block(models[name], "wmt25-", ("0",))
                h = block(models[name], "heldout-", ("0",))
                d1 = a["1"] - b_agg["1"] if a["1"] and b_agg["1"] else None
                dh = h["0"] - b_held["0"] if h["0"] and b_held["0"] else None
                print(f"  {mix+suffix:20}" + " ".join(fmt(a[k]) for k in ks) +
                      f" {fmt(n['0'],8)} {fmt(h['0'],9)} | {fmt(d1,7)} {fmt(dh,7)}")


def bootstrap(models: dict, data_dir: Path, cache_dir: Path, n_boot: int) -> None:
    """Paired bootstrap over documents, on the held-out paragraph sets."""
    import numpy as np
    from fdtem.comet_io import cache_path
    from fdtem.stats import bootstrap_delta, doc_units, tau

    def units(label):
        out = []
        for f in sorted(data_dir.glob("heldout-*_val.csv")):
            df = pd.read_csv(f)
            cache = cache_path(label, df, cache_dir)
            if not cache.exists():
                return None
            out += doc_units(df.assign(pred=np.load(cache)), group="lp")
        return out

    for fam, label in FAMILIES:
        base = units(f"{fam}-base")
        if base is None:
            continue
        print(f"\n{label} — held-out paragraphs, {n_boot} document bootstraps")
        print(f"  {'baseline':24}{tau(base):7.3f}")
        for mix in MIXES:
            for suffix in ("", "-frozen"):
                name = f"{fam}-{mix}{suffix}"
                if name not in models:
                    continue
                u = units(name)
                if u is None or len(u) != len(base):
                    continue
                d, lo, hi = bootstrap_delta(u, base, n_boot)
                star = "*" if lo > 0 or hi < 0 else " "
                print(f"  {mix+suffix:24}{tau(u):7.3f}   {d:+.3f} [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\n* = 95% CI excludes zero")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/length_training/correlation.json")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--data_dir", default="~/scratch/wmt_eval_portion")
    ap.add_argument("--cache_dir", default="results/length_training/pred_cache")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    models = json.load(open(Path(args.results)))["models"]
    table(models)
    if args.bootstrap:
        bootstrap(models, Path(args.data_dir).expanduser(), Path(args.cache_dir), args.n_boot)


if __name__ == "__main__":
    main()
