#!/usr/bin/env python3
"""
make_uncontrolled_mix.py — the deliberately CONTAMINATED training mix.

Every other mix in this experiment is a controlled object: fixed size, fixed
per-pool quotas, document-disjoint splits, evaluation sets the model has never
seen. This one is the opposite, on purpose. It pours in every scored row we
have — the training pools, the validation split those pools were held out as,
and the held-out WMT22/23/24/25 evaluation portions themselves — and trains on
all of it.

Why build a model we already know we cannot report a correlation for: it is the
far end of the axis. The published metric is the "never saw any of this" point;
the four composition arms sit a controlled distance away; this one is as far as
continued training can push, with the evaluation data itself in the training
set. Whatever the length behaviour does between those points is the effect
size's ceiling, and the arms in between become readable against it.

What it costs, stated once so it is never forgotten downstream:

  * The correlation lens (validation AND held-out) is meaningless for this arm.
    Its numbers measure memorisation. The output carries a CONTAMINATED marker
    file, `contaminated: true` in the manifest, and the arm is named
    `uncontrolled` so it is visible in every table it appears in.
  * MetaDocEval stays clean — it is a different corpus (WMT24++ en→fr/es/de,
    contrastive pairs, no human scores) and nothing from it enters here. That
    is the one lens on which this arm can be honestly compared to the others,
    and it is the reason to build it at all.
  * The validation split written next to the mix is a random slice OF THE
    CONTAMINATED POOL. It exists so the trainer has something to monitor and
    checkpoint on; it is not a measurement, and `val_kendall` from this arm must
    not be compared with any other arm's.

Score scales: the pool CSVs are already z-normalised per (source_set, lp) and
sigmoid-squashed into (0, 1) by prepare_data.py; the evaluation portions carry
RAW scores (a negative MQM penalty). Mixing the two raw would hand the MSE loss
two incompatible scales, so evaluation rows get the same monotone treatment,
fitted per (file, lp) on the rows being added. Direction is higher = better on
both sides and no step here flips it.

Usage:
  python experiments/length_training/make_uncontrolled_mix.py \
      --data_dir ~/scratch/wmt_length_data_v2 \
      --eval_dirs ~/scratch/wmt_eval_portion \
      --out_dir ~/scratch/wmt_length_data_v2/mixes_pure/uncontrolled
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
csv.field_size_limit(10 ** 9)

COLS = ["src", "mt", "ref", "score", "lp", "k", "system", "doc_id",
        "seg_start", "domain", "source_set", "origin"]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def squash(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """z-score per group, then sigmoid — the same monotone increasing map
    prepare_data.py applies to the pools, so the two sources land on one scale.
    Groups too small to have a spread keep a constant 0.5 rather than exploding."""
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
    stem = path.name.replace("_val.csv", "")
    missing = [c for c in ("src", "mt", "score", "lp") if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing column(s) {missing}")
    if "ref" not in df.columns:
        df["ref"] = ""
    df["k"] = pd.to_numeric(df.get("k", 0), errors="coerce").fillna(0).astype(int)
    for c, default in (("system", ""), ("doc_id", ""), ("seg_start", 0),
                       ("domain", "")):
        if c not in df.columns:
            df[c] = default
    df["source_set"] = stem
    df["origin"] = "eval_portion"
    # Already squashed pools live in (0, 1); a raw MQM penalty does not. Detect
    # rather than assume, so this works whichever way the eval sets were built.
    lo, hi = float(df.score.min()), float(df.score.max())
    if lo < 0.0 or hi > 1.0:
        df = squash(df, ["source_set", "lp"])
        logger.info(f"  {path.name}: {len(df):>7,} rows, raw score {lo:.2f}..{hi:.2f} "
                    f"→ z+sigmoid per lp")
    else:
        logger.info(f"  {path.name}: {len(df):>7,} rows, score already in "
                    f"[{lo:.2f}, {hi:.2f}] — left as is")
    return df[COLS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="~/scratch/wmt_length_data_v2",
                    help="Pools built by prepare_data.py (all_train.csv / all_val.csv).")
    ap.add_argument("--eval_dirs", nargs="+",
                    default=["~/scratch/wmt_eval_portion"],
                    help="Directories of held-out *_val.csv evaluation sets. "
                         "THESE ARE THE CONTAMINATION — every row here is a row "
                         "the correlation lens will later be scored on.")
    ap.add_argument("--out_dir", default=None,
                    help="default: <data_dir>/mixes_pure/uncontrolled")
    ap.add_argument("--val_frac", type=float, default=0.02,
                    help="Random slice held out purely so the trainer has a "
                         "monitor. Contaminated by construction — never report it.")
    ap.add_argument("--max_concat_tokens", type=int, default=0,
                    help="Drop rows whose src+mt exceeds this XLM-R token budget "
                         "(508 mirrors the QE arms). 0 = keep everything, which "
                         "is the uncontrolled default: over-budget rows are "
                         "truncated by the model rather than dropped.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    out_dir = (Path(args.out_dir).expanduser() if args.out_dir
               else data_dir / "mixes_pure" / "uncontrolled")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, provenance = [], {}

    # ── the controlled pools, train AND val ──────────────────────────────────
    for split in ("train", "val"):
        p = data_dir / f"all_{split}.csv"
        if not p.exists():
            raise SystemExit(f"missing {p} — build the pools with prepare_data.py")
        df = pd.read_csv(p)
        for c in COLS:
            if c not in df.columns:
                df[c] = "" if c in ("domain", "system", "doc_id") else 0
        df = df[COLS]
        logger.info(f"  {p.name}: {len(df):>7,} rows")
        provenance[f"pool_{split}"] = len(df)
        frames.append(df)

    # ── the contamination ────────────────────────────────────────────────────
    logger.info("\nEvaluation portions folded into training (the contamination):")
    n_eval = 0
    for raw in args.eval_dirs:
        d = Path(raw).expanduser()
        if not d.exists():
            logger.warning(f"  ! {d}: absent — skipped")
            continue
        files = [f for f in sorted(d.glob("*_val.csv")) if not f.name.startswith("all_")]
        if not files:
            logger.warning(f"  ! {d}: no *_val.csv — skipped")
            continue
        for f in files:
            df = load_eval_portion(f)
            provenance[f"eval/{f.name}"] = len(df)
            n_eval += len(df)
            frames.append(df)
    if n_eval == 0:
        raise SystemExit(
            "no evaluation rows were added, so this mix is not contaminated and "
            "there is no reason to build it. Check --eval_dirs.")

    mix = pd.concat(frames, ignore_index=True)

    # A few native documents embed newlines that survive the CSV quoting badly
    # enough to shift a row's fields, so `lp` lands in `score` (2 of 254,497 in
    # the v2 pools). make_mixtures.py drops them as a side effect of grouping by
    # origin; this builder concatenates instead, so they would reach the trainer
    # — where COMET casts the score column to float and the run dies on
    # `could not convert string to float: 'en-cs'` about a minute in. Drop them
    # here, loudly.
    score = pd.to_numeric(mix["score"], errors="coerce")
    k_num = pd.to_numeric(mix["k"], errors="coerce")
    bad = score.isna() | k_num.isna()
    if bad.any():
        logger.warning(f"\n! dropping {int(bad.sum())} malformed row(s) whose "
                       f"score or k is not numeric (shifted CSV fields)")
        for _, r in mix[bad].head(3).iterrows():
            logger.warning(f"    score={str(r['score'])[:30]!r} k={str(r['k'])[:20]!r} "
                           f"src={str(r['src'])[:50]!r}")
        mix = mix[~bad].copy()
    mix["score"] = pd.to_numeric(mix["score"])
    mix["k"] = pd.to_numeric(mix["k"]).astype(int)

    if args.max_concat_tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("xlm-roberta-large")

        def n(t):
            return len(tok(str(t), add_special_tokens=False).input_ids)
        keep = mix.apply(lambda r: n(r["src"]) + n(r["mt"]) <= args.max_concat_tokens,
                         axis=1)
        logger.info(f"\nconcat filter: {keep.sum():,}/{len(mix):,} kept "
                    f"(src+mt <= {args.max_concat_tokens})")
        mix = mix[keep].copy()

    rng = np.random.RandomState(args.seed)
    mix = mix.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    n_val = max(1, int(round(len(mix) * args.val_frac)))
    val, train = mix.iloc[:n_val], mix.iloc[n_val:]

    train.to_csv(out_dir / "all_train.csv", index=False)
    val.to_csv(out_dir / "all_val.csv", index=False)

    # Read back what was written: the failure mode above is a WRITE artefact as
    # much as a read one, and finding it here costs seconds instead of a job.
    for name in ("all_train.csv", "all_val.csv"):
        back = pd.read_csv(out_dir / name, low_memory=False)
        n_bad = int(pd.to_numeric(back["score"], errors="coerce").isna().sum())
        if n_bad:
            raise SystemExit(
                f"{out_dir/name}: {n_bad} row(s) still have a non-numeric score "
                f"after the round trip — COMET will die casting the column. "
                f"The text fields need re-quoting before this mix can be used.")
        logger.info(f"  verified {name}: {len(back):,} rows, score numeric throughout")

    # A file whose only job is to be noticed by whoever finds this directory.
    (out_dir / "CONTAMINATED").write_text(
        "This mix contains the held-out evaluation sets.\n"
        "Correlation numbers from a model trained on it measure memorisation, "
        "not agreement with human judgement, and must never be reported as the "
        "latter. MetaDocEval is a different corpus and stays a valid lens.\n"
        f"Evaluation rows folded in: {n_eval:,}\n")

    manifest = {
        "mix": "uncontrolled",
        "contaminated": True,
        "contaminated_lenses": ["correlation_val", "correlation_heldout"],
        "clean_lenses": ["metadoceval"],
        "seed": args.seed,
        "rows": {"train": len(train), "val": len(val), "total": len(mix),
                 "from_evaluation_sets": n_eval},
        "provenance": provenance,
        "val_note": "random slice of the contaminated pool; a trainer monitor, "
                    "not a measurement",
        "checksums_md5": {"all_train.csv": md5(out_dir / "all_train.csv"),
                          "all_val.csv": md5(out_dir / "all_val.csv")},
    }
    json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=2)

    logger.info(f"\ntrain={len(train):,}  val={len(val):,}  "
                f"({n_eval:,} rows came from the evaluation sets)")
    logger.info(f"by origin: {dict(mix.origin.value_counts())}")
    logger.info(f"→ {out_dir}  [CONTAMINATED]")


if __name__ == "__main__":
    main()
