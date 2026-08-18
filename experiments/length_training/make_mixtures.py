#!/usr/bin/env python3
"""
make_wmt_mix.py — constant-size sentence-fraction mixes over the WMT pools.

Pools (built by prepare_wmt_length_data.py):
  sent    generalMT2022 single segments (k=1)
  agg     generalMT2022 aggregated windows (k ∈ {2,3,4,6})
  native  WMT25 whole scored documents (k=0)

Every mix fracNNN has the same TOTAL number of rows N:
  NNN% sentences; the remaining long mass split 50/50 between agg and native
  (agg internally even over k). Two ablations pin the long-mass origin at f=0:
  frac000nat (all-native long) and frac000agg (all-aggregated long).

Per-(pool, lp) quotas are proportional to pool availability and FIXED across
mixes; each (pool, lp, k) cell is shuffled once with the seed and every mix
takes a prefix — mixes differ only by prefix length. manifest.json records
the seed, quotas and an md5 per CSV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

FRACS = (0, 10, 20, 40, 60, 80, 100)
K_LONG = (2, 3, 4, 6)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default="~/scratch/wmt_length_data")
    ap.add_argument("--out_dir", default=None, help="default: <data_dir>/mixes")
    ap.add_argument("--total", type=int, default=None,
                    help="rows per mix (default: max feasible, capped 24000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--match_lp_coverage", action="store_true",
                    help="Keep only language pairs present in EVERY pool, so "
                         "composition is the only thing that varies across "
                         "mixes. Off by default: the pools overlap on very few "
                         "pairs, so this trades language coverage for a clean "
                         "contrast — see the coverage report this prints.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    out_root = Path(args.out_dir).expanduser() if args.out_dir else data_dir / "mixes"
    df = pd.read_csv(data_dir / "all_train.csv")
    rng = np.random.RandomState(args.seed)

    # ── language-pair coverage across pools ──────────────────────────────────
    # The pools come from different releases: `sent`/`agg` from WMT22 (3 pairs),
    # `native` from WMT25 (13 pairs). A mix that shifts mass from sentences to
    # documents therefore also shifts its language distribution, so "long text
    # hurts/helps" and "these languages are harder" are confounded unless the
    # analysis controls for it. Report it loudly either way.
    by_pool = {o: set(g["lp"].unique()) for o, g in df.groupby("origin")}
    for pool in ("sent", "agg", "native"):
        lps = sorted(by_pool.get(pool, ()))
        logger.info(f"  coverage[{pool:6s}]: {len(lps):2d} pairs — {' '.join(lps)}")
    shared = set.intersection(*by_pool.values()) if by_pool else set()
    logger.info(f"  shared by all pools: {len(shared)} — {' '.join(sorted(shared))}")

    if args.match_lp_coverage:
        if not shared:
            raise SystemExit("--match_lp_coverage: the pools share no language pair")
        before = len(df)
        df = df[df["lp"].isin(shared)].reset_index(drop=True)
        logger.info(f"  --match_lp_coverage: {before:,} → {len(df):,} rows, "
                    f"{len(shared)} pair(s)")
    elif len(shared) < max(len(v) for v in by_pool.values()):
        logger.warning(
            "  ! pools have DIFFERENT language coverage: the sentence fraction "
            "and the language distribution move together across mixes. Compare "
            "arms per language pair, or rebuild with --match_lp_coverage.")

    pools = {o: g.sample(frac=1.0, random_state=rng).reset_index(drop=True)
             for o, g in df.groupby("origin")}
    n_sent, n_agg, n_nat = (len(pools.get(p, [])) for p in ("sent", "agg", "native"))
    n_total = min(n_sent, 2 * n_agg, 2 * n_nat)
    if args.total:
        n_total = min(n_total, args.total)
    else:
        n_total = min(n_total, 24_000)
    logger.info(f"pools: sent={n_sent:,} agg={n_agg:,} native={n_nat:,} "
                f"→ N per mix = {n_total:,}")

    def take(pool: str, n: int, offset: int = 0) -> pd.DataFrame:
        g = pools[pool]
        if offset + n > len(g):
            raise SystemExit(f"pool exhausted: {pool} needs {offset + n}, has {len(g)}")
        return g.iloc[offset:offset + n]

    manifest = {"seed": args.seed, "total_per_mix": n_total,
                "pool_sizes": {"sent": n_sent, "agg": n_agg, "native": n_nat},
                "lp_coverage": {p: sorted(v) for p, v in by_pool.items()},
                "lp_coverage_shared": sorted(shared),
                "match_lp_coverage": bool(args.match_lp_coverage),
                "mixes": {}}

    specs = {f"frac{f:03d}": ("mix", f) for f in FRACS}
    specs["frac000nat"] = ("nat", 0)
    specs["frac000agg"] = ("agg", 0)

    for name, (kind, f) in specs.items():
        n_s = round(n_total * f / 100)
        n_long = n_total - n_s
        if kind == "mix":
            n_a = n_long // 2
            n_n = n_long - n_a
        elif kind == "nat":
            n_a, n_n = 0, n_long
        else:
            n_a, n_n = n_long, 0
        if n_a > n_agg or n_n > n_nat:
            logger.info(f"  ! {name}: infeasible (agg {n_a}/{n_agg}, "
                        f"native {n_n}/{n_nat}) — skipped")
            continue
        parts = []
        if n_s:
            parts.append(take("sent", n_s))
        if n_a:
            parts.append(take("agg", n_a))
        if n_n:
            parts.append(take("native", n_n))
        mix = pd.concat(parts, ignore_index=True)
        mdir = out_root / name
        mdir.mkdir(parents=True, exist_ok=True)
        sums = {}
        for lp, g in mix.groupby("lp"):
            p = mdir / f"{lp}_train.csv"
            g.to_csv(p, index=False)
            sums[p.name] = md5(p)
        mix.to_csv(mdir / "all_train.csv", index=False)
        sums["all_train.csv"] = md5(mdir / "all_train.csv")
        counts = {"sent": int(n_s), "agg": int(n_a), "native": int(n_n),
                  "per_lp": {lp: int(n) for lp, n in
                             mix.groupby("lp").size().items()},
                  "per_k": {str(k): int(n) for k, n in
                            mix.groupby("k").size().items()},
                  # per (pool, lp): the breakdown the coverage warning is about
                  "per_origin_lp": {f"{o}/{lp}": int(n) for (o, lp), n in
                                    mix.groupby(["origin", "lp"]).size().items()}}
        manifest["mixes"][name] = {"counts": counts, "checksums_md5": sums}
        logger.info(f"  {name}: {len(mix):,} rows (sent={n_s} agg={n_a} native={n_n})")

    with open(out_root / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info(f"Mixes → {out_root}")


if __name__ == "__main__":
    main()
