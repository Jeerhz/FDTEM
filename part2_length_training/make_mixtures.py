"""Named training mixes over the WMT pools, every controlled mix at the same size N.

Controlled mixes (MIX_SPECS): fracNNN = NNN % sentences, the long remainder split 50/50
between aggregated windows and native documents; frac000nat / frac000agg take all the
long mass from one pool. Per-(pool, lp) quotas are proportional to availability, each
(pool, lp, k) cell is shuffled once with the seed and every mix takes a prefix, so
mixes differ only by prefix length. manifest.json (MixManifest) records seed, counts
and an md5 per CSV.

  --total_policy mix   N = min(sent, 2*agg, 2*native): fits the ladder, pure arms infeasible
  --total_policy pure  N = min(sent, agg, native): the four-arm grid, every arm the same N
An infeasible arm is a hard error unless --allow_skip.

`uncontrolled` (--arms uncontrolled, needs --eval_dirs) is the deliberately CONTAMINATED
mix: every pool row (train and val) plus every held-out *_val.csv of --eval_dirs,
evaluation rows brought onto the pools' (0, 1) scale by the same z+sigmoid per (file, lp).
Its correlation numbers measure memorisation; MetaDocEval is a different corpus and stays
a valid lens. The directory carries a CONTAMINATED marker, its own manifest.json
(UncontrolledManifest) and its own all_val.csv (a random slice: a trainer monitor, not a
measurement).

  python -m part2_length_training.make_mixtures --data_dir ~/scratch/wmt_length_data_v2 \
      --out_dir ~/scratch/wmt_length_data_v2/mixes_pure --total_policy pure \
      --arms frac100 frac000agg frac000nat frac000
  python -m part2_length_training.make_mixtures ... --arms uncontrolled --eval_dirs ~/scratch/wmt_eval_portion
  python -m part2_length_training.make_mixtures ... --verify --arms frac100 frac000 [--preset default]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from common.paths import SCRATCH
from part2_length_training.arms import MIX_SPECS, TRAIN_PRESETS
from part2_length_training.models import (POOL_COLUMNS, MixCounts, MixEntry, MixManifest,
                                          UncontrolledManifest)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
csv.field_size_limit(10 ** 9)

STEPS_PER_ROW = 1 / 32  # batch 4 x accumulate 8


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── controlled mixes ─────────────────────────────────────────────────────────

def build_controlled(data_dir: Path, out_root: Path, arms: list[str], total_policy: str,
                     total: int | None, cap: int, seed: int, allow_skip: bool,
                     match_lp_coverage: bool) -> MixManifest:
    df = pd.read_csv(data_dir / "all_train.csv")
    rng = np.random.RandomState(seed)

    # sent/agg come from WMT22 (3 pairs), native from WMT25 (13 pairs): the sentence
    # fraction and the language distribution move together across mixes. Say so.
    by_pool = {o: set(g["lp"].unique()) for o, g in df.groupby("origin")}
    for pool in ("sent", "agg", "native"):
        lps = sorted(by_pool.get(pool, ()))
        logger.info(f"  coverage[{pool:6s}]: {len(lps):2d} pairs - {' '.join(lps)}")
    shared = set.intersection(*by_pool.values()) if by_pool else set()
    logger.info(f"  shared by all pools: {len(shared)} - {' '.join(sorted(shared))}")
    if match_lp_coverage:
        if not shared:
            raise SystemExit("--match_lp_coverage: the pools share no language pair")
        before = len(df)
        df = df[df["lp"].isin(shared)].reset_index(drop=True)
        logger.info(f"  --match_lp_coverage: {before:,} -> {len(df):,} rows, {len(shared)} pair(s)")
    elif len(shared) < max(len(v) for v in by_pool.values()):
        logger.warning("  ! pools have DIFFERENT language coverage: compare arms per language "
                       "pair, or rebuild with --match_lp_coverage.")

    pools = {o: g.sample(frac=1.0, random_state=rng).reset_index(drop=True)
             for o, g in df.groupby("origin")}
    n_sent, n_agg, n_nat = (len(pools.get(p, [])) for p in ("sent", "agg", "native"))
    feasible = {"mix": min(n_sent, 2 * n_agg, 2 * n_nat), "pure": min(n_sent, n_agg, n_nat)}
    n_total = min(feasible[total_policy], total or cap)
    logger.info(f"pools: sent={n_sent:,} agg={n_agg:,} native={n_nat:,}")
    logger.info(f"  N feasible: mix-policy {feasible['mix']:,} / pure-policy {feasible['pure']:,}")
    logger.info(f"  -> policy={total_policy}, N per mix = {n_total:,}")

    def take(pool: str, n: int) -> pd.DataFrame:
        g = pools[pool]
        if n > len(g):
            raise SystemExit(f"pool exhausted: {pool} needs {n}, has {len(g)}")
        return g.iloc[:n]

    manifest = MixManifest(seed=seed, total_per_mix=n_total, total_policy=total_policy,
                           n_feasible=feasible,
                           pool_sizes={"sent": n_sent, "agg": n_agg, "native": n_nat},
                           lp_coverage={p: sorted(v) for p, v in by_pool.items()},
                           lp_coverage_shared=sorted(shared),
                           match_lp_coverage=match_lp_coverage, mixes={})

    for name in arms:
        spec = MIX_SPECS[name]
        n_s = round(n_total * spec.sentence_fraction / 100)
        n_long = n_total - n_s
        if spec.kind == "mix":
            n_a = n_long // 2
            n_n = n_long - n_a
        elif spec.kind == "nat":
            n_a, n_n = 0, n_long
        else:
            n_a, n_n = n_long, 0
        if n_s > n_sent or n_a > n_agg or n_n > n_nat:
            msg = (f"{name}: infeasible at N={n_total:,} (needs sent {n_s}/{n_sent}, "
                   f"agg {n_a}/{n_agg}, native {n_n}/{n_nat})")
            if allow_skip:
                logger.warning(f"  ! {msg} - SKIPPED")
                manifest.skipped[name] = {"need": {"sent": n_s, "agg": n_a, "native": n_n},
                                          "have": {"sent": n_sent, "agg": n_agg, "native": n_nat}}
                continue
            raise SystemExit(f"{msg}\n  Either --total_policy pure (N = {feasible['pure']:,}), "
                             f"--total <= {feasible['pure']:,}, or --allow_skip (then say so).")
        parts = [take(pool, n) for pool, n in (("sent", n_s), ("agg", n_a), ("native", n_n)) if n]
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
        counts = MixCounts(
            sent=int(n_s), agg=int(n_a), native=int(n_n),
            per_lp={lp: int(n) for lp, n in mix.groupby("lp").size().items()},
            per_k={str(k): int(n) for k, n in mix.groupby("k").size().items()},
            per_origin_lp={f"{o}/{lp}": int(n) for (o, lp), n in mix.groupby(["origin", "lp"]).size().items()})
        manifest.mixes[name] = MixEntry(counts=counts, checksums_md5=sums)
        logger.info(f"  {name}: {len(mix):,} rows (sent={n_s} agg={n_a} native={n_n})")

    if manifest.mixes:
        manifest.assert_equal_totals(list(manifest.mixes))
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    logger.info(f"{len(manifest.mixes)} mix(es) x {n_total:,} rows -> {out_root}")
    return manifest


# ── the uncontrolled (contaminated) mix ──────────────────────────────────────

def squash(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """z-score per group, then sigmoid: the map load_wmt_pools applies to the pools."""
    out = df.copy()
    for keys, g in df.groupby(by):
        mu, sd = g.score.mean(), g.score.std()
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        m = np.ones(len(df), dtype=bool)
        for col, val in zip(by, keys if isinstance(keys, tuple) else (keys,)):
            m &= (df[col] == val).to_numpy()
        out.loc[m, "score"] = 1 / (1 + np.exp(-(df.loc[m, "score"] - mu) / sd))
    return out


def load_eval_portion(path: Path) -> pd.DataFrame:
    """One held-out evaluation CSV, brought onto the pool schema and scale."""
    df = pd.read_csv(path)
    missing = [c for c in ("src", "mt", "score", "lp") if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing column(s) {missing}")
    if "ref" not in df.columns:
        df["ref"] = ""
    df["k"] = pd.to_numeric(df.get("k", 0), errors="coerce").fillna(0).astype(int)
    for c, default in (("system", ""), ("doc_id", ""), ("seg_start", 0), ("domain", "")):
        if c not in df.columns:
            df[c] = default
    df["source_set"] = path.name.replace("_val.csv", "")
    df["origin"] = "eval_portion"
    lo, hi = float(df.score.min()), float(df.score.max())
    if lo < 0.0 or hi > 1.0:  # raw MQM penalty, not yet on the pools' (0, 1) scale
        df = squash(df, ["source_set", "lp"])
        logger.info(f"  {path.name}: {len(df):>7,} rows, raw score {lo:.2f}..{hi:.2f} -> z+sigmoid per lp")
    else:
        logger.info(f"  {path.name}: {len(df):>7,} rows, score already in [{lo:.2f}, {hi:.2f}]")
    return df[POOL_COLUMNS]


def build_uncontrolled(data_dir: Path, eval_dirs: list[Path], out_dir: Path, val_frac: float,
                       max_concat_tokens: int, seed: int) -> UncontrolledManifest:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, provenance = [], {}
    for split in ("train", "val"):
        p = data_dir / f"all_{split}.csv"
        if not p.exists():
            raise SystemExit(f"missing {p} - build the pools with load_wmt_pools")
        df = pd.read_csv(p)
        for c in POOL_COLUMNS:
            if c not in df.columns:
                df[c] = "" if c in ("domain", "system", "doc_id") else 0
        frames.append(df[POOL_COLUMNS])
        provenance[f"pool_{split}"] = len(df)
        logger.info(f"  {p.name}: {len(df):>7,} rows")

    logger.info("Evaluation portions folded into training (the contamination):")
    n_eval = 0
    for d in eval_dirs:
        files = [f for f in sorted(d.glob("*_val.csv")) if not f.name.startswith("all_")]
        if not files:
            logger.warning(f"  ! {d}: no *_val.csv - skipped")
            continue
        for f in files:
            df = load_eval_portion(f)
            provenance[f"eval/{f.name}"] = len(df)
            n_eval += len(df)
            frames.append(df)
    if n_eval == 0:
        raise SystemExit("no evaluation rows were added, so this mix is not contaminated. Check --eval_dirs.")
    mix = pd.concat(frames, ignore_index=True)

    # A few native documents embed newlines badly enough to shift a row's fields (2 of
    # 254,497 in the v2 pools); COMET would die casting `score` to float a minute in.
    bad = pd.to_numeric(mix["score"], errors="coerce").isna() | pd.to_numeric(mix["k"], errors="coerce").isna()
    if bad.any():
        logger.warning(f"! dropping {int(bad.sum())} malformed row(s) whose score or k is not numeric")
        mix = mix[~bad].copy()
    mix["score"] = pd.to_numeric(mix["score"])
    mix["k"] = pd.to_numeric(mix["k"]).astype(int)

    if max_concat_tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("xlm-roberta-large")

        def n(t):
            return len(tok(str(t), add_special_tokens=False).input_ids)
        keep = mix.apply(lambda r: n(r["src"]) + n(r["mt"]) <= max_concat_tokens, axis=1)
        logger.info(f"concat filter: {keep.sum():,}/{len(mix):,} kept (src+mt <= {max_concat_tokens})")
        mix = mix[keep].copy()

    mix = mix.sample(frac=1.0, random_state=np.random.RandomState(seed)).reset_index(drop=True)
    n_val = max(1, int(round(len(mix) * val_frac)))
    val, train = mix.iloc[:n_val], mix.iloc[n_val:]
    train.to_csv(out_dir / "all_train.csv", index=False)
    val.to_csv(out_dir / "all_val.csv", index=False)
    for name in ("all_train.csv", "all_val.csv"):  # the failure above is a write artefact too
        back = pd.read_csv(out_dir / name, low_memory=False)
        n_bad = int(pd.to_numeric(back["score"], errors="coerce").isna().sum())
        if n_bad:
            raise SystemExit(f"{out_dir / name}: {n_bad} row(s) with a non-numeric score after the round trip")
        logger.info(f"  verified {name}: {len(back):,} rows, score numeric throughout")

    (out_dir / "CONTAMINATED").write_text(
        "This mix contains the held-out evaluation sets.\n"
        "Correlation numbers from a model trained on it measure memorisation, not agreement "
        "with human judgement, and must never be reported as the latter. MetaDocEval is a "
        "different corpus and stays a valid lens.\n"
        f"Evaluation rows folded in: {n_eval:,}\n")
    manifest = UncontrolledManifest(
        mix="uncontrolled", contaminated=True,
        contaminated_lenses=["correlation_val", "correlation_heldout"], clean_lenses=["metadoceval"],
        seed=seed,
        rows={"train": len(train), "val": len(val), "total": len(mix), "from_evaluation_sets": n_eval},
        provenance=provenance,
        val_note="random slice of the contaminated pool; a trainer monitor, not a measurement",
        checksums_md5={"all_train.csv": md5(out_dir / "all_train.csv"), "all_val.csv": md5(out_dir / "all_val.csv")})
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    logger.info(f"train={len(train):,} val={len(val):,} ({n_eval:,} rows from the evaluation sets)")
    logger.info(f"-> {out_dir}  [CONTAMINATED]")
    return manifest


# ── verification (the launcher's design guard) ───────────────────────────────

def verify(out_root: Path, arms: list[str], preset: str | None) -> None:
    controlled = [a for a in arms if a != "uncontrolled"]
    rows = {}
    if controlled:
        man_path = out_root / "manifest.json"
        if not man_path.exists():
            raise SystemExit(f"no manifest at {man_path} - build the mixes first")
        man = MixManifest.model_validate_json(man_path.read_text())
        n = man.assert_equal_totals(controlled)
        print(f"mixes OK: {len(controlled)} arm(s) x {n:,} rows (policy={man.total_policy}, seed={man.seed})")
        for a in controlled:
            c = man.mixes[a].counts
            print(f"  {a:12} sent={c.sent:>7,}  agg={c.agg:>7,}  native={c.native:>7,}")
            rows[a] = n
    if "uncontrolled" in arms:
        udir = out_root / "uncontrolled"
        if not (udir / "CONTAMINATED").exists() or not (udir / "manifest.json").exists():
            raise SystemExit(f"{udir}: no CONTAMINATED marker or manifest - build it with --arms uncontrolled")
        uman = UncontrolledManifest.model_validate_json((udir / "manifest.json").read_text())
        print(f"  {'uncontrolled':12} train={uman.rows['train']:,} val={uman.rows['val']:,} "
              f"(from evaluation sets: {uman.rows['from_evaluation_sets']:,})  [CONTAMINATED]")
        rows["uncontrolled"] = uman.rows["train"]
    if preset:
        p = TRAIN_PRESETS[preset]
        print(f"preset {preset}: max_epochs={p.max_epochs} patience={p.patience}")
        for a, n in rows.items():
            steps = max(1, int(n * STEPS_PER_ROW))
            print(f"  {a:12} ~{steps:,} optimizer steps/epoch -> ~{steps * p.max_epochs:,} steps at most")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(SCRATCH / "wmt_length_data_v2"))
    ap.add_argument("--out_dir", default=None, help="default: <data_dir>/mixes")
    ap.add_argument("--arms", nargs="+", default=None,
                    help=f"mixes to build (default: every controlled one); known: {list(MIX_SPECS)}")
    ap.add_argument("--total", type=int, default=None, help="rows per mix (default: feasible max, capped)")
    ap.add_argument("--total_policy", choices=("mix", "pure"), default="mix")
    ap.add_argument("--cap", type=int, default=24_000, help="upper bound on N when --total is not given")
    ap.add_argument("--allow_skip", action="store_true", help="skip an infeasible arm instead of refusing")
    ap.add_argument("--match_lp_coverage", action="store_true",
                    help="keep only language pairs present in every pool (one pair on the v2 pools)")
    ap.add_argument("--eval_dirs", nargs="+", default=[str(SCRATCH / "wmt_eval_portion")],
                    help="uncontrolled only: directories of held-out *_val.csv (the contamination)")
    ap.add_argument("--val_frac", type=float, default=0.02, help="uncontrolled only: monitor slice")
    ap.add_argument("--max_concat_tokens", type=int, default=0,
                    help="uncontrolled only: drop rows with src+mt over this budget (0 = keep all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true", help="check the manifest(s) of --arms and print counts")
    ap.add_argument("--preset", default=None, help="with --verify: also print the training budget")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    out_root = Path(args.out_dir).expanduser() if args.out_dir else data_dir / "mixes"
    arms = args.arms or [n for n, s in MIX_SPECS.items() if s.kind != "uncontrolled"]
    unknown = [a for a in arms if a not in MIX_SPECS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; known: {list(MIX_SPECS)}")
    if args.verify:
        verify(out_root, arms, args.preset)
        return
    controlled = [a for a in arms if a != "uncontrolled"]
    if controlled:
        build_controlled(data_dir, out_root, controlled, args.total_policy, args.total, args.cap,
                         args.seed, args.allow_skip, args.match_lp_coverage)
    if "uncontrolled" in arms:
        build_uncontrolled(data_dir, [Path(d).expanduser() for d in args.eval_dirs],
                           out_root / "uncontrolled", args.val_frac, args.max_concat_tokens, args.seed)


if __name__ == "__main__":
    main()
