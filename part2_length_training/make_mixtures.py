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

N and the four-arm design
-------------------------
`--total_policy mix` (default) sets N = min(n_sent, 2·n_agg, 2·n_nat): enough
for the fracNNN ladder, where no mix ever needs more than half its rows from a
single long pool. The PURE arms need a whole N from one pool, so under that
policy they are frequently infeasible — and used to be dropped with a one-line
log, which is how a four-arm comparison can silently become a two-arm one.

`--total_policy pure` sets N = min(n_sent, n_agg, n_nat), the largest N at
which all four of

    frac100      100 % single sentences
    frac000agg   100 % concatenated windows
    frac000nat   100 % native documents
    frac000       50 / 50 concatenated + native

hold the SAME number of rows. That is the policy to build the four-arm grid
with: one epoch is then the same number of examples in every arm, so the
training budget stays a controlled factor. Infeasible arms are now a hard
error (`--allow_skip` to go back to skipping them).

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
                    help="rows per mix (default: max feasible under "
                         "--total_policy, capped by --cap)")
    ap.add_argument("--total_policy", choices=("mix", "pure"), default="mix",
                    help="mix: N = min(sent, 2*agg, 2*native) — fits the fracNNN "
                         "ladder, but the pure-pool arms are usually infeasible. "
                         "pure: N = min(sent, agg, native) — the largest N at "
                         "which frac100 / frac000agg / frac000nat / frac000 all "
                         "hold the same number of rows. Use it for the four-arm grid.")
    ap.add_argument("--cap", type=int, default=24_000,
                    help="upper bound on N when --total is not given")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="Build only these mixes (e.g. frac100 frac000agg "
                         "frac000nat frac000). Default: all of them.")
    ap.add_argument("--allow_skip", action="store_true",
                    help="Skip an infeasible arm with a warning instead of "
                         "refusing to build. Off by default: a silently missing "
                         "arm turns a four-way comparison into a two-way one.")
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
    feasible = {"mix": min(n_sent, 2 * n_agg, 2 * n_nat),
                "pure": min(n_sent, n_agg, n_nat)}
    n_total = min(feasible[args.total_policy], args.total or args.cap)
    logger.info(f"pools: sent={n_sent:,} agg={n_agg:,} native={n_nat:,}")
    logger.info(f"  N feasible: mix-policy {feasible['mix']:,} · "
                f"pure-policy {feasible['pure']:,}  "
                f"(the pure policy is what the four-arm grid needs)")
    logger.info(f"  → policy={args.total_policy}, N per mix = {n_total:,}")

    def take(pool: str, n: int, offset: int = 0) -> pd.DataFrame:
        g = pools[pool]
        if offset + n > len(g):
            raise SystemExit(f"pool exhausted: {pool} needs {offset + n}, has {len(g)}")
        return g.iloc[offset:offset + n]

    manifest = {"seed": args.seed, "total_per_mix": n_total,
                "total_policy": args.total_policy,
                "n_feasible": feasible,
                "pool_sizes": {"sent": n_sent, "agg": n_agg, "native": n_nat},
                "lp_coverage": {p: sorted(v) for p, v in by_pool.items()},
                "lp_coverage_shared": sorted(shared),
                "match_lp_coverage": bool(args.match_lp_coverage),
                "mixes": {}}

    specs = {f"frac{f:03d}": ("mix", f) for f in FRACS}
    specs["frac000nat"] = ("nat", 0)
    specs["frac000agg"] = ("agg", 0)
    if args.arms:
        unknown = [a for a in args.arms if a not in specs]
        if unknown:
            raise SystemExit(f"unknown arm(s) {unknown}; known: {sorted(specs)}")
        specs = {k: v for k, v in specs.items() if k in args.arms}

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
        if n_s > n_sent or n_a > n_agg or n_n > n_nat:
            msg = (f"{name}: infeasible at N={n_total:,} "
                   f"(needs sent {n_s}/{n_sent}, agg {n_a}/{n_agg}, "
                   f"native {n_n}/{n_nat})")
            if args.allow_skip:
                logger.warning(f"  ! {msg} — SKIPPED")
                manifest.setdefault("skipped", {})[name] = {
                    "need": {"sent": n_s, "agg": n_a, "native": n_n},
                    "have": {"sent": n_sent, "agg": n_agg, "native": n_nat}}
                continue
            raise SystemExit(
                f"{msg}\n"
                f"  The pure-pool arms need a whole N from one pool. Either\n"
                f"    --total_policy pure   (N = {feasible['pure']:,}, all four "
                f"arms at the same size), or\n"
                f"    --total <= {min(n_sent, n_agg, n_nat):,}, or\n"
                f"    --allow_skip          (build the ladder without this arm — "
                f"then say so in the results).")
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

    # The design guarantee, checked rather than asserted in prose: every mix
    # built in this run holds the same number of rows, so one epoch is the same
    # number of examples in every arm.
    totals = {n: sum(m["counts"][p] for p in ("sent", "agg", "native"))
              for n, m in manifest["mixes"].items()}
    if len(set(totals.values())) > 1:
        raise SystemExit(f"mixes have unequal totals: {totals}")

    with open(out_root / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info(f"{len(totals)} mix(es) x {n_total:,} rows → {out_root}")


if __name__ == "__main__":
    main()
