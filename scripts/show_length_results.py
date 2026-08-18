#!/usr/bin/env python3
"""Print the length-composition results table from length_correlation_portion.json."""
import json, statistics as st, sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1
            else "results/retrain_wmt/length_correlation_portion.json")
d = json.load(open(path))["models"]

def block(lps, pref, ks):
    """mean Kendall tau over language pairs, per k, for files starting with pref"""
    acc = {}
    for name, per_k in lps.items():
        if name.startswith("_") or not name.startswith(pref):
            continue
        for k, v in per_k.items():
            if v.get("kendall") is not None:
                acc.setdefault(k, []).append(v["kendall"])
    return {k: (st.mean(acc[k]) if k in acc else None) for k in ks}

def row(m):
    lps = d[m]
    a = block(lps, "wmt22-", ("1", "2", "3", "4", "6"))
    n = block(lps, "wmt25-", ("0",))["0"]
    h = block(lps, "heldout-", ("0",))["0"]
    return a, n, h

base = {m: row(m) for m in ("da-base", "qe-base") if m in d}

def fmt(v, w=6):
    return f"{v:{w}.3f}" if v is not None else " " * (w - 1) + "-"

ORDER = ["frac000", "frac010", "frac020", "frac040", "frac060", "frac080",
         "frac100", "frac000nat", "frac000agg"]

for fam, bname, label in (("da", "da-base", "COMET-DA  (reference-based)"),
                          ("qe", "qe-base", "CometKiwi QE  (reference-free)")):
    if bname not in base:
        continue
    ba, bn, bh = base[bname]
    print(f"\n{'='*104}\n{label}   —   mean Kendall tau\n{'='*104}")
    print(f"{'arm':22} {'k=1':>6} {'k=2':>6} {'k=3':>6} {'k=4':>6} {'k=6':>6} "
          f"{'WMT25doc':>9} {'HELD-OUT':>9} | {'Dk1':>6} {'Dheld':>6}")
    print(f"{'baseline (no finetune)':22} " +
          " ".join(fmt(ba[k]) for k in ("1", "2", "3", "4", "6")) +
          f" {fmt(bn,9)} {fmt(bh,9)} |      -      -")
    print("-" * 104)
    for reg, tag in ((f"", "encoder trained"), ("-frozen", "encoder frozen")):
        print(f"  [{tag}]")
        for mix in ORDER:
            m = f"{fam}-{mix}{reg}"
            if m not in d:
                continue
            a, n, h = row(m)
            dk1 = a["1"] - ba["1"] if a["1"] is not None and ba["1"] is not None else None
            dh = h - bh if h is not None and bh is not None else None
            print(f"  {mix + reg:20} " +
                  " ".join(fmt(a[k]) for k in ("1", "2", "3", "4", "6")) +
                  f" {fmt(n,9)} {fmt(h,9)} | {fmt(dk1)} {fmt(dh)}")
print(f"\nk=1 sentences · k=2..6 aggregated windows · WMT25doc = native documents "
      f"· HELD-OUT = WMT23/24 paragraphs (never trained on)")
print(f"Dk1 / Dheld = change vs the un-finetuned baseline.   models scored: {len(d)}")
