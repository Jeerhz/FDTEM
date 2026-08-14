#!/usr/bin/env python3
"""
make_sentence_mix.py — build the SENTENCE-FRACTION training mixes for the
"retrain COMET on long text only" sweep.

Question (see docs/RETRAIN_AND_BLOCK_XSIM.md, Experiment A2): a COMET continued
on *paragraph-only* data may lose sentence-level correlation with human
judgements. How much sentence-level data does it take to keep it? We train one
model per mix, where a fraction f of the training windows are single sentences
(k=1) and the rest are long windows (k ∈ --long_k), at CONSTANT total size —
so the only thing that varies across mixes is the composition, never the
amount of data.

Inputs are the *unbalanced* per-LP window pools written by
scripts/prepare_paragraph_data.py ({lp}_trainpool.csv — run it first; older
data dirs without trainpool files must be regenerated). Validation is NOT
mixed: every mix is evaluated on the same shared {lp}_val.csv files.

Sampling is NESTED for verifiability: each (lp, k) pool is shuffled once with
--seed, and every mix takes a prefix of that single permutation. Two mixes
therefore differ only by how far into each pool they read — nothing is
resampled per mix.

Outputs (under --out_dir, default <data_dir>/mixes):
    frac000/ frac010/ ... frac100/   {lp}_train.csv + all_train.csv + counts.json
    manifest.json / MANIFEST.md      sources, construction, per-cell counts,
                                     md5 checksum of every CSV

Usage:
    python scripts/make_sentence_mix.py --data_dir ~/scratch/paragraph_mqm \
        [--fracs 0 10 20 40 60 80 100] [--long_k 2 3 4 6] [--seed 42]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

COLS = ["src", "mt", "ref", "score", "lp", "k", "system", "doc_id",
        "seg_start", "penalty_sum", "penalty_mean",
        "tok_src", "tok_mt", "tok_ref"]


def frac_dirname(pct: int) -> str:
    return f"frac{pct:03d}"


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_evenly(total: int, bins: int) -> list[int]:
    """total split into `bins` integer parts, deterministic remainder-first."""
    base, rem = divmod(total, bins)
    return [base + (1 if i < rem else 0) for i in range(bins)]


def plan(n: int, f: float, long_k: list[int]) -> tuple[int, list[int]]:
    """How mix f of total size n is composed: sentences + one count per long k."""
    n_sent = round(f * n)
    return n_sent, split_evenly(n - n_sent, len(long_k))


def max_total(n_sent: int, long_min: int, long_k: list[int],
              fracs: list[float]) -> int:
    """Largest per-LP total N whose integer composition fits the pools at EVERY
    fraction: f·N sentences must fit the k=1 pool and each long share must fit
    the scarcest long pool. The closed form can be off by one after the integer
    split, so the candidate is walked down until every fraction is feasible."""
    best = None
    for f in fracs:
        cands = []
        if f > 0:
            cands.append(math.floor(n_sent / f))
        if f < 1:
            cands.append(math.floor(len(long_k) * long_min / (1 - f)))
        n_f = min(cands) if cands else 0
        best = n_f if best is None else min(best, n_f)
    n = max(best or 0, 0)
    while n > 0:
        if all(plan(n, f, long_k)[0] <= n_sent
               and max(plan(n, f, long_k)[1] or [0]) <= long_min
               for f in fracs):
            return n
        n -= 1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="~/scratch/paragraph_mqm",
                    help="Output dir of prepare_paragraph_data.py (needs *_trainpool.csv).")
    ap.add_argument("--out_dir", default=None,
                    help="Default: <data_dir>/mixes")
    ap.add_argument("--fracs", nargs="+", type=int,
                    default=[0, 10, 20, 40, 60, 80, 100],
                    help="Sentence share of each mix, in percent. 0 = long-only, "
                         "100 = sentence-only control.")
    ap.add_argument("--long_k", nargs="+", type=int, default=[2, 3, 4, 6],
                    help="Window sizes that count as 'long'.")
    ap.add_argument("--per_lp_total", type=int, default=None,
                    help="Cap the per-LP mix size (default: largest feasible).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else data_dir / "mixes"
    out_dir.mkdir(parents=True, exist_ok=True)
    fracs = sorted(set(args.fracs))
    frac_floats = [p / 100.0 for p in fracs]

    pool_files = sorted(data_dir.glob("*_trainpool.csv"))
    if not pool_files:
        raise SystemExit(
            f"No *_trainpool.csv in {data_dir} — re-run scripts/prepare_paragraph_data.py "
            "(trainpool output was added for this sweep).")

    prep_stats = None
    if (data_dir / "stats.json").exists():
        prep_stats = json.loads((data_dir / "stats.json").read_text())

    rng = random.Random(args.seed)
    # one permutation per (lp, k) pool; every mix reads a prefix of it
    pools: dict[str, dict[int, pd.DataFrame]] = {}
    totals: dict[str, int] = {}
    for f in pool_files:
        lp = f.name.replace("_trainpool.csv", "")
        df = pd.read_csv(f)
        by_k = {}
        for k in [1] + args.long_k:
            sub = df[df["k"] == k]
            if len(sub):
                by_k[k] = sub.sample(frac=1.0, random_state=rng.randrange(2**32)
                                     ).reset_index(drop=True)
        missing = [k for k in [1] + args.long_k if k not in by_k]
        if missing:
            print(f"[{lp}] pools empty for k={missing} — skipping this LP")
            continue
        long_min = min(len(by_k[k]) for k in args.long_k)
        n = max_total(len(by_k[1]), long_min, args.long_k, frac_floats)
        if args.per_lp_total:
            n = min(n, args.per_lp_total)
        if n == 0:
            print(f"[{lp}] no feasible mix size — skipping this LP")
            continue
        pools[lp] = by_k
        totals[lp] = n
        print(f"[{lp}] pool sizes: k1={len(by_k[1]):,}  "
              + "  ".join(f"k{k}={len(by_k[k]):,}" for k in args.long_k)
              + f"  → per-mix total N={n:,}")

    if not pools:
        raise SystemExit("No usable language pair.")

    manifest: dict = {
        "experiment": "sentence_fraction_mixes",
        "created": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "fracs_pct": fracs,
        "long_k": args.long_k,
        "data_dir": str(data_dir),
        "source": {
            "dataset": "Bio-MQM (Zouhar et al. 2024, ACL short — arXiv:2402.18747)",
            "repo": "https://github.com/amazon-science/bio-mqm-dataset",
            "windows": "consecutive same-(system,doc_id) segments concatenated by "
                       "scripts/prepare_paragraph_data.py; window score = per-LP "
                       "z+sigmoid of the MEAN per-segment MQM penalty (fit on train)",
            "pools": "{lp}_trainpool.csv = ALL train windows before length-balancing",
            "prepare_stats": prep_stats.get("args") if prep_stats else None,
        },
        "design": {
            "constant_total": "every mix of one LP has the same number of training "
                              "windows — only the sentence/long composition varies",
            "nested_sampling": "each (lp,k) pool shuffled once with the seed; every "
                               "mix takes a prefix, so mixes differ only by prefix length",
            "validation": "shared, untouched {lp}_val.csv of the parent data_dir "
                          "(never mixed or subsampled)",
        },
        "per_lp_total": totals,
        "mixes": {},
    }

    for pct, f in zip(fracs, frac_floats):
        mix_dir = out_dir / frac_dirname(pct)
        mix_dir.mkdir(parents=True, exist_ok=True)
        all_parts = []
        mix_info: dict = {"sentence_pct": pct, "per_lp": {}}
        for lp, by_k in pools.items():
            n = totals[lp]
            n_sent, long_counts = plan(n, f, args.long_k)
            parts = [by_k[1].head(n_sent)]
            for k, c in zip(args.long_k, long_counts):
                parts.append(by_k[k].head(c))
            mix = pd.concat(parts, ignore_index=True)
            # head() would silently shrink the mix if a pool ran out — the
            # constant-total design depends on this never happening
            if len(mix) != n:
                raise SystemExit(f"{lp} {frac_dirname(pct)}: got {len(mix)} windows, "
                                 f"expected {n} — pool exhausted")
            mix = mix.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
            mix[COLS].to_csv(mix_dir / f"{lp}_train.csv", index=False)
            all_parts.append(mix[COLS])
            mix_info["per_lp"][lp] = {
                "total": int(len(mix)), "k1": n_sent,
                **{f"k{k}": int(c) for k, c in zip(args.long_k, long_counts)},
            }
        all_df = pd.concat(all_parts, ignore_index=True)
        all_df.to_csv(mix_dir / "all_train.csv", index=False)
        mix_info["total"] = int(len(all_df))
        mix_info["checksums_md5"] = {p.name: md5_file(p)
                                     for p in sorted(mix_dir.glob("*.csv"))}
        (mix_dir / "counts.json").write_text(json.dumps(mix_info, indent=2))
        manifest["mixes"][frac_dirname(pct)] = mix_info
        print(f"{frac_dirname(pct)}: {len(all_df):,} windows "
              f"({pct}% sentences, {100-pct}% long)")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_manifest_md(out_dir, manifest)
    print(f"\nMixes → {out_dir}  (manifest.json / MANIFEST.md)")


def _write_manifest_md(out_dir: Path, m: dict) -> None:
    lines = [
        "# Sentence-fraction training mixes",
        "",
        f"Generated {m['created']} — seed {m['seed']}.",
        "",
        "## Source",
        f"- Dataset: {m['source']['dataset']}",
        f"- Repo: {m['source']['repo']}",
        f"- Windows: {m['source']['windows']}",
        f"- Pools: {m['source']['pools']}",
        "",
        "## Design",
        f"- {m['design']['constant_total']}",
        f"- {m['design']['nested_sampling']}",
        f"- Validation: {m['design']['validation']}",
        f"- Long window sizes: k ∈ {m['long_k']}",
        "",
        "## Mixes (training windows per LP × k)",
        "",
    ]
    for name, info in m["mixes"].items():
        lines.append(f"### {name} — {info['sentence_pct']}% sentences, "
                     f"{info['total']:,} windows total")
        lps = info["per_lp"]
        ks = [c for c in next(iter(lps.values())) if c.startswith("k")]
        lines.append("| lp | " + " | ".join(ks) + " | total |")
        lines.append("|----" * (len(ks) + 2) + "|")
        for lp, row in lps.items():
            lines.append(f"| {lp} | " + " | ".join(str(row[k]) for k in ks)
                         + f" | {row['total']} |")
        lines.append("")
    lines += ["## Checksums", "",
              "Per-mix `counts.json` carries the md5 of every CSV; regenerate with "
              "the same seed and compare to verify the construction.", ""]
    (out_dir / "MANIFEST.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
