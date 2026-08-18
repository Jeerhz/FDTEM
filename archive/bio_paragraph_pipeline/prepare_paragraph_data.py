#!/usr/bin/env python3
"""
prepare_paragraph_data.py — build a *length-balanced, paragraph-level* COMET
training mix from Bio-MQM.

Bio-MQM segments carry (system, doc_id, seg_id), so consecutive segments of the
same document form real paragraphs. For every document we take sliding windows
of k consecutive segments (k ∈ --k_list; k=1 keeps the original sentence-level
data), concatenate src/mt/ref, and give the window the *mean per-segment MQM
penalty* — at k=1 this is exactly the score basis of prepare_bio_mqm_data.py,
so all lengths live on one scale. Scores are then z-normalised + sigmoid-squashed
per language pair (parameters fit on train only).

The training mix is balanced by length: every k contributes the same number of
windows (the count of the scarcest length, unless --per_length overrides it).
Train windows use stride 1 (overlapping = augmentation); validation windows are
non-overlapping (stride k) and are NOT subsampled, so per-k correlations in
eval_length_correlation.py use everything available.

Windows longer than --max_tokens on any side (XLM-R tokens) are dropped —
512-token encoder limit.

Usage:
    python scripts/prepare_paragraph_data.py \
        --output_dir ~/scratch/paragraph_mqm \
        [--k_list 1 2 3 4 6] [--per_length N] [--wandb_project comet-retrain]

Outputs (in --output_dir):
    {lp}_train.csv / {lp}_val.csv   COMET-ready (src, mt, ref, score) + extras
                                    (lp, k, system, doc_id, seg_start, penalty_*)
    {lp}_trainpool.csv              ALL train windows before length-balancing —
                                    the sampling pool for make_sentence_mix.py
    all_train.csv / all_val.csv     concatenation across language pairs
    stats.json                      counts, normalisation params, token stats
    plots/*.png                     length distribution / counts / score-by-k
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_bio_mqm_data import (
    ALL_LANG_PAIRS,
    aggregate,
    build_ref_map,
    clone_or_pull,
    find_reference_file,
    find_system_files,
    load_annotations,
    REPO_URL,
)

NO_SPACE_LANGS = {"zh", "ja", "th"}


def _seg_sort_key(seg_id: str):
    try:
        return (0, int(seg_id))
    except (TypeError, ValueError):
        return (1, str(seg_id))


def _joiner(lang: str) -> str:
    return "" if lang in NO_SPACE_LANGS else " "


class TokenCounter:
    """Per-segment XLM-R token counts with memoisation. A window's length is
    approximated as the sum of its segments' counts (+2 specials) — exact up to
    boundary merges, and ~100x cheaper than tokenising every window."""

    def __init__(self, model_name: str = "xlm-roberta-large"):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self._memo: dict[str, int] = {}

    def count(self, text: str) -> int:
        n = self._memo.get(text)
        if n is None:
            n = len(self.tok(text, add_special_tokens=False)["input_ids"])
            self._memo[text] = n
        return n


def sigmoid_z(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma == 0:
        return np.full(len(x), 0.5)
    return 1.0 / (1.0 + np.exp(-(x - mu) / sigma))


def build_windows(agg: pd.DataFrame, ref_map: dict, lp: str, k_list: list[int],
                  counter: TokenCounter, max_tokens: int,
                  val_docs: set, train_docs: set) -> pd.DataFrame:
    """All candidate windows for one language pair, both splits, all k."""
    src_lang, tgt_lang = lp.split("-")
    js, jt = _joiner(src_lang), _joiner(tgt_lang)
    rows = []
    for (system, doc_id), g in agg.groupby(["system", "doc_id"], sort=False):
        split = "train" if doc_id in train_docs else "val" if doc_id in val_docs else None
        if split is None:
            continue
        segs = sorted(g.to_dict("records"), key=lambda r: _seg_sort_key(r["seg_id"]))
        # a window is usable only if every segment has a reference
        refs = [ref_map.get((r["doc_id"], r["seg_id"]), "") for r in segs]
        n = len(segs)
        for k in k_list:
            stride = 1 if split == "train" else k
            for i in range(0, n - k + 1, stride):
                win, wrefs = segs[i:i + k], refs[i:i + k]
                if any(not r for r in wrefs):
                    continue
                src = js.join(s["source"] for s in win)
                mt = jt.join(s["target"] for s in win)
                ref = jt.join(wrefs)
                t_src = sum(counter.count(s["source"]) for s in win) + 2
                t_mt = sum(counter.count(s["target"]) for s in win) + 2
                t_ref = sum(counter.count(r) for r in wrefs) + 2
                if max(t_src, t_mt, t_ref) > max_tokens:
                    continue
                pen = [s["mqm_score"] for s in win]
                rows.append({
                    "lp": lp, "split": split, "k": k, "system": system,
                    "doc_id": doc_id, "seg_start": i,
                    "src": src, "mt": mt, "ref": ref,
                    "penalty_sum": float(np.sum(pen)),
                    "penalty_mean": float(np.mean(pen)),
                    "tok_src": t_src, "tok_mt": t_mt, "tok_ref": t_ref,
                })
    return pd.DataFrame(rows)


def balance_train(df: pd.DataFrame, k_list: list[int], per_length: int | None,
                  rng: random.Random) -> pd.DataFrame:
    """Equal number of windows per length k (the scarcest length sets the count
    unless --per_length overrides it). Val is never subsampled."""
    train = df[df["split"] == "train"]
    counts = {k: int((train["k"] == k).sum()) for k in k_list}
    avail = {k: c for k, c in counts.items() if c > 0}
    if not avail:
        return train
    n_target = per_length if per_length else min(avail.values())
    parts = []
    for k in avail:
        idx = train.index[train["k"] == k].tolist()
        take = min(n_target, len(idx))
        parts.append(train.loc[sorted(rng.sample(idx, take))])
    return pd.concat(parts).sort_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", default="~/scratch/paragraph_mqm")
    ap.add_argument("--repo_dir", default="~/scratch/bio_mqm/bio-mqm-dataset",
                    help="Existing Bio-MQM clone (cloned if missing).")
    ap.add_argument("--lang_pairs", nargs="+", default=ALL_LANG_PAIRS)
    ap.add_argument("--k_list", nargs="+", type=int, default=[1, 2, 3, 4, 6])
    ap.add_argument("--per_length", type=int, default=None,
                    help="Train windows per k (default: count of the scarcest k).")
    ap.add_argument("--max_tokens", type=int, default=480,
                    help="Drop windows whose src/mt/ref exceed this many XLM-R tokens.")
    ap.add_argument("--tokenizer", default="xlm-roberta-large")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb_project", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(args.repo_dir).expanduser()
    clone_or_pull(REPO_URL, repo_dir)
    doc_splits = json.loads((repo_dir / "data" / "doc_id_splits.json").read_text())

    counter = TokenCounter(args.tokenizer)
    rng = random.Random(args.seed)
    stats: dict = {"args": {k: v for k, v in vars(args).items()}, "lang_pairs": {}}
    all_train, all_val = [], []

    for lp in args.lang_pairs:
        files = find_system_files(repo_dir, lp)
        if not files:
            print(f"[{lp}] no annotation files — skipping")
            continue
        agg = aggregate(load_annotations(files))
        ref_map = build_ref_map(find_reference_file(repo_dir, lp))
        splits = doc_splits.get(lp, {})
        train_docs, val_docs = set(splits.get("dev", [])), set(splits.get("test", []))

        df = build_windows(agg, ref_map, lp, args.k_list, counter,
                           args.max_tokens, val_docs, train_docs)
        if df.empty:
            print(f"[{lp}] no usable windows — skipping")
            continue

        # score: per-LP z+sigmoid of the mean per-segment penalty, fit on train
        tr_pen = df.loc[df["split"] == "train", "penalty_mean"]
        mu, sigma = float(tr_pen.mean()), float(tr_pen.std())
        df["score"] = sigmoid_z(df["penalty_mean"].to_numpy(), mu, sigma)

        train = balance_train(df, args.k_list, args.per_length, rng)
        val = df[df["split"] == "val"]

        cols = ["src", "mt", "ref", "score", "lp", "k", "system", "doc_id",
                "seg_start", "penalty_sum", "penalty_mean",
                "tok_src", "tok_mt", "tok_ref"]
        train[cols].to_csv(out_dir / f"{lp}_train.csv", index=False)
        # full (unbalanced) train pool — sampling stock for make_sentence_mix.py
        df.loc[df["split"] == "train", cols].to_csv(
            out_dir / f"{lp}_trainpool.csv", index=False)
        val[cols].to_csv(out_dir / f"{lp}_val.csv", index=False)
        all_train.append(train[cols])
        all_val.append(val[cols])

        per_k = {int(k): {"train": int((train["k"] == k).sum()),
                          "val": int((val["k"] == k).sum())} for k in args.k_list}
        stats["lang_pairs"][lp] = {
            "norm_mu": mu, "norm_sigma": sigma,
            "windows_before_balancing": int((df["split"] == "train").sum()),
            "per_k": per_k,
            "mean_tok_mt_by_k": {int(k): float(df.loc[df["k"] == k, "tok_mt"].mean())
                                 for k in args.k_list if (df["k"] == k).any()},
        }
        print(f"[{lp}] train={len(train):,} val={len(val):,} per-k={per_k}")

    if not all_train:
        raise SystemExit("No data produced.")
    train_df = pd.concat(all_train, ignore_index=True)
    val_df = pd.concat(all_val, ignore_index=True)
    train_df.to_csv(out_dir / "all_train.csv", index=False)
    val_df.to_csv(out_dir / "all_val.csv", index=False)
    stats["total"] = {"train": len(train_df), "val": len(val_df)}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nTotal: train={len(train_df):,} val={len(val_df):,} → {out_dir}")

    plot_paths = _plots(train_df, val_df, out_dir / "plots")

    if args.wandb_project:
        import wandb
        run = wandb.init(project=args.wandb_project, name="paragraph_data_prep",
                         job_type="data_prep", config=vars(args))
        run.summary.update(stats["total"])
        table = wandb.Table(dataframe=train_df.groupby(["lp", "k"])
                            .agg(n=("score", "size"), mean_score=("score", "mean"),
                                 mean_tok_mt=("tok_mt", "mean")).reset_index())
        run.log({"train_mix": table})
        for p in plot_paths:
            run.log({f"plots/{p.stem}": wandb.Image(str(p))})
        run.finish()

    print("\n── Paste into your training YAML ──────────────────────────")
    print("    train_data:")
    for lp in stats["lang_pairs"]:
        print(f"      - {out_dir}/{lp}_train.csv")
    print("    validation_data:")
    for lp in stats["lang_pairs"]:
        print(f"      - {out_dir}/{lp}_val.csv")


def _plots(train_df: pd.DataFrame, val_df: pd.DataFrame, plot_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    ks = sorted(train_df["k"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in ks:
        ax.hist(train_df.loc[train_df["k"] == k, "tok_mt"], bins=40,
                histtype="step", label=f"k={k}", density=True)
    ax.set_xlabel("MT length (XLM-R tokens)")
    ax.set_ylabel("density")
    ax.set_title("Training mix — length distribution per window size k")
    ax.legend()
    p = plot_dir / "length_distribution.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(8, 5))
    counts = (train_df.groupby(["lp", "k"]).size().unstack(fill_value=0))
    counts.plot.bar(ax=ax)
    ax.set_ylabel("# train windows")
    ax.set_title("Length-balanced training mix (windows per lp × k)")
    p = plot_dir / "counts_per_k.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [train_df.loc[train_df["k"] == k, "score"] for k in ks]
    ax.boxplot(data, labels=[str(k) for k in ks], showfliers=False)
    ax.set_xlabel("window size k (segments)")
    ax.set_ylabel("normalised score")
    ax.set_title("Score distribution by window size (train)")
    p = plot_dir / "score_by_k.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    for pth in paths:
        print(f"  [plot] {pth}")
    return paths


if __name__ == "__main__":
    main()
