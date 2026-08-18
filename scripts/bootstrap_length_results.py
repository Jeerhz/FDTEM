#!/usr/bin/env python3
"""Paired bootstrap over documents: is arm X different from its baseline?"""
import hashlib, sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import kendalltau

PORTION = Path.home()/"scratch/wmt_eval_portion"
CACHE = Path("results/retrain_wmt/pred_cache")
B = 1000
rng = np.random.RandomState(0)

def preds(label, df):
    h = hashlib.md5("\n".join(df["src"]+"|"+df["mt"]+"|"+df["ref"]).encode()).hexdigest()[:10]
    f = CACHE/f"{label}__{h}__n{len(df)}.npy"
    return np.load(f) if f.exists() else None

def tau_block(files, label, kfilter=None):
    """concatenate per-file taus weighted equally -> returns per-doc resamplable units"""
    units = []          # (file_idx, doc, gold[], pred[])
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        p = preds(label, df)
        if p is None: return None
        df = df.assign(pred=p)
        df["k"] = pd.to_numeric(df["k"], errors="coerce")
        if kfilter is not None: df = df[df.k == kfilter]
        if len(df) < 3: continue
        for doc, g in df.groupby("doc_id"):
            units.append((i, g["score"].to_numpy(float), g["pred"].to_numpy()))
    return units

def tau_from(units, idx):
    per_file = {}
    for j in idx:
        i, gold, pred = units[j]
        per_file.setdefault(i, [[], []])
        per_file[i][0].append(gold); per_file[i][1].append(pred)
    taus = []
    for i, (gs, ps) in per_file.items():
        g = np.concatenate(gs); p = np.concatenate(ps)
        if len(g) > 2 and g.std() > 0 and p.std() > 0:
            taus.append(kendalltau(g, p)[0])
    return float(np.mean(taus)) if taus else np.nan

def compare(files, base, arms, kfilter=None, name=""):
    ub = tau_block(files, base, kfilter)
    if ub is None: return
    n = len(ub)
    boots = rng.randint(0, n, size=(B, n))
    print(f"\n{name}  (n={n} documents, {B} bootstrap resamples)")
    print(f"  {'model':22} {'tau':>7}   {'delta vs baseline [95% CI]':>32}")
    tb = tau_from(ub, range(n))
    tb_boot = np.array([tau_from(ub, b) for b in boots])
    print(f"  {base+' (baseline)':22} {tb:7.3f}")
    for a in arms:
        ua = tau_block(files, a, kfilter)
        if ua is None or len(ua) != n: continue
        ta = tau_from(ua, range(n))
        d = np.array([tau_from(ua, b) for b in boots]) - tb_boot
        lo, hi = np.percentile(d, [2.5, 97.5])
        sig = "*" if lo > 0 or hi < 0 else " "
        print(f"  {a:22} {ta:7.3f}   {ta-tb:+7.3f}  [{lo:+.3f}, {hi:+.3f}] {sig}")

held = sorted(PORTION.glob("heldout-*_val.csv"))
sent = sorted(PORTION.glob("wmt22-*_val.csv"))
da = ["da-frac000","da-frac100","da-frac000nat","da-frac000agg","da-frac100-frozen","da-frac000-frozen"]
qe = ["qe-frac000","qe-frac100","qe-frac100-frozen","qe-frac000-frozen","qe-frac000nat-frozen"]
compare(held, "da-base", da, None, "HELD-OUT paragraphs (WMT23/24) — COMET-DA")
compare(held, "qe-base", qe, None, "HELD-OUT paragraphs (WMT23/24) — CometKiwi QE")
compare(sent, "da-base", da, 1.0, "SENTENCES k=1 — COMET-DA")
compare(sent, "qe-base", qe, 1.0, "SENTENCES k=1 — CometKiwi QE")
print("\n* = 95% CI excludes zero")
