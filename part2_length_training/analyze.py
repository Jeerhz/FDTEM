"""Results table of the correlation lens, with an optional document bootstrap against the baseline.

  python -m part2_length_training.analyze                       # heldout table
  python -m part2_length_training.analyze --lens val
  python -m part2_length_training.analyze --bootstrap           # + 95% CI vs baseline (needs the prediction cache)

Columns: k=1 single sentences, k=2..6 aggregated windows, k=0 whole documents; `held-out`
= the WMT23/24 paragraph sets (heldout-* files), never seen in training.
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import pandas as pd

from common.paths import SCRATCH
from part2_length_training import CACHE_DIR, RESULTS_DIR
from part2_length_training.arms import MIX_SPECS
from part2_length_training.models import CorrelationResults, ModelCorrelation

MIXES = list(MIX_SPECS)
# Trained on the evaluation sets on purpose: shown so a row is not silently missing, but
# its correlation numbers measure memorisation and compare with nothing.
CONTAMINATED = {n for n, s in MIX_SPECS.items() if s.kind == "uncontrolled"}
FAMILIES = [("da", "COMET-DA (reference-based)"), ("qe", "CometKiwi QE (reference-free)")]


def block(m: ModelCorrelation, prefix: str, ks) -> dict:
    """Mean Kendall tau over the files whose name starts with `prefix`."""
    acc: dict = {}
    for name, per_k in m.by_file.items():
        if not name.startswith(prefix):
            continue
        for k, v in per_k.items():
            if v.kendall is not None:
                acc.setdefault(k, []).append(v.kendall)
    return {k: (st.mean(acc[k]) if k in acc else None) for k in ks}


def fmt(v, w=6):
    return f"{v:{w}.3f}" if v is not None else " " * (w - 1) + "-"


def table(models: dict[str, ModelCorrelation]) -> None:
    ks = ("1", "2", "3", "4", "6")
    for fam, label in FAMILIES:
        base = f"{fam}-base"
        if base not in models:
            continue
        b_agg, b_nat = block(models[base], "wmt22-", ks), block(models[base], "wmt25-", ("0",))
        b_held = block(models[base], "heldout-", ("0",))
        print(f"\n{label}\n" + "-" * 96)
        head = " ".join(f"{'k=' + k:>6}" for k in ks)
        print(f"{'arm':22}{head} {'docs':>8} {'held-out':>9} | {'Dk1':>7} {'Dheld':>7}")
        print(f"{'baseline':22}" + " ".join(fmt(b_agg[k]) for k in ks)
              + f" {fmt(b_nat['0'], 8)} {fmt(b_held['0'], 9)} |")
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
                flag = "  << CONTAMINATED: memorisation, not agreement" if mix in CONTAMINATED else ""
                print(f"  {mix + suffix:20}" + " ".join(fmt(a[k]) for k in ks)
                      + f" {fmt(n['0'], 8)} {fmt(h['0'], 9)} | {fmt(d1, 7)} {fmt(dh, 7)}" + flag)


def bootstrap(models: dict, data_dir: Path, cache_dir: Path, n_boot: int) -> None:
    """Paired bootstrap over documents, on the held-out paragraph sets."""
    import numpy as np
    from common.comet_models import cache_path
    from common.stats import bootstrap_delta, doc_units, tau

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
        print(f"\n{label} - held-out paragraphs, {n_boot} document bootstraps")
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
                flag = "  << CONTAMINATED" if mix in CONTAMINATED else ""
                print(f"  {mix + suffix:24}{tau(u):7.3f}   {d:+.3f} [{lo:+.3f}, {hi:+.3f}] {star}{flag}")
    print("\n* = 95% CI excludes zero")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens", choices=("val", "heldout"), default="heldout")
    ap.add_argument("--results", default=None, help="default: results/correlation_<lens>.json")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--data_dir", default=str(SCRATCH / "wmt_eval_portion"))
    ap.add_argument("--cache_dir", default=None, help="default: results/cache/pred_<lens>")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    results_path = Path(args.results) if args.results else RESULTS_DIR / f"correlation_{args.lens}.json"
    models = CorrelationResults.model_validate_json(results_path.read_text()).models
    table(models)
    if args.bootstrap:
        cache_dir = Path(args.cache_dir) if args.cache_dir else CACHE_DIR / f"pred_{args.lens}"
        bootstrap(models, Path(args.data_dir).expanduser(), cache_dir, args.n_boot)


if __name__ == "__main__":
    main()
