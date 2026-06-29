#!/usr/bin/env python3
"""
run.py — one command reproduces every table and figure.

    python run.py --config config.yaml                 # full run (cluster, GPU)
    python run.py --config config.yaml --smoke         # 1 lang, ~200 docs, 2 buckets
    python run.py --config config.yaml --backend mock --source synthetic --smoke   # local logic test

Pipeline: print plan + resolved source -> build dataset -> extract frozen features
-> probe (both regimes, all seeds) -> analyze (slopes, gap, CIs, geometry) -> plots
-> write SUMMARY.md answering whether COMET is more length-sensitive than raw XLM-R.
"""
from __future__ import annotations

import argparse
import logging
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from common import active_encoders, banner, ensure_dirs, load_config

logger = logging.getLogger(__name__)


def _md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Minimal GitHub-markdown table (avoids the optional `tabulate` dependency)."""
    cols = list(df.columns)
    def fmt(v):
        if isinstance(v, float):
            return f"{v:{floatfmt}}"
        return str(v)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def print_plan(cfg: dict) -> None:
    pair = active_encoders(cfg)
    banner("Experiment plan")
    print("  Hypothesis: COMET's sentence-level QE fine-tuning specializes the encoder for short")
    print("    inputs, so probe accuracy from COMET features should degrade / fail to improve with")
    print("    length faster than from raw XLM-R. Same backbone ⇒ any gap is due to fine-tuning.")
    print(f"\n  Encoder pair ({cfg['encoders']['active_pair']}):")
    print(f"    comet : {pair['comet']['kind']}:{pair['comet']['ref']}  (extract .model + .tokenizer)")
    print(f"    raw   : {pair['raw']['kind']}:{pair['raw']['ref']}")
    print(f"  Length buckets (tokens) : {cfg['length_buckets']}   (min_tokens kept = {cfg['min_tokens']})")
    print(f"  Pooling / normalize_l2  : {cfg['features']['pooling']} / {cfg['features']['normalize_l2']}")
    print(f"  Backend / fp16 / device : {cfg['features']['backend']} / {cfg['features']['fp16']} / {cfg['features']['device']}")
    print(f"  Probe / seeds           : {cfg['probe']['classifier']} / {cfg['probe']['seeds']}")
    print(f"  Regimes                 : probe_per_length (primary) + train_short@L={cfg['probe']['train_short_L']}")
    print(f"  Data source             : {cfg['data']['source']}  langs={cfg['data']['languages']}")
    print(f"  Split                   : stratified by {cfg['split']['stratify_by']} seed={cfg['split']['seed']} test_size={cfg['data']['test_size']}")
    print("\n  Environment:")
    print(f"    python={platform.python_version()}  numpy={np.__version__}  pandas={pd.__version__}")
    try:
        import sklearn, torch  # noqa: PLC0415
        print(f"    sklearn={sklearn.__version__}  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    gpu={torch.cuda.get_device_name(0)}")
    except Exception:
        import sklearn  # noqa: PLC0415
        print(f"    sklearn={sklearn.__version__}  torch=not installed (mock/logic mode)")


def write_summary(cfg: dict, out_path: Path) -> None:
    ad = Path(cfg["paths"]["analysis_dir"])
    slopes = pd.read_csv(ad / "slopes.csv")
    gap = pd.read_csv(ad / "gap_by_L.csv").sort_values("L")
    acc_sl = slopes[slopes["metric"] == "acc"].set_index("role")
    s_comet = float(acc_sl.loc["comet", "slope_per_log2L"]) if "comet" in acc_sl.index else float("nan")
    s_raw = float(acc_sl.loc["raw", "slope_per_log2L"]) if "raw" in acc_sl.index else float("nan")
    g_lo, g_hi = gap.iloc[0], gap.iloc[-1]
    d_gap = float(g_hi["gap_acc"] - g_lo["gap_acc"])

    flatter = s_comet < s_raw
    loses_ground = d_gap < 0
    if flatter and loses_ground:
        verdict = "**Supported.** COMET's accuracy rises more slowly with length and its gap over raw XLM-R shrinks (or turns negative) as inputs lengthen."
    elif flatter or loses_ground:
        verdict = "**Mixed.** Only part of the prediction holds (slope and gap-vs-length disagree); see tables."
    else:
        verdict = "**Not supported.** COMET does not degrade / saturate faster than raw XLM-R over this length range."

    pair = active_encoders(cfg)
    mock = cfg["features"]["backend"] == "mock"
    lines = []
    if mock:
        lines += ["> ⚠️ **LOGIC-SMOKE ARTIFACT — NOT a scientific result.** Generated with the",
                  "> `mock` feature backend (fabricated, torch-free features) to validate the pipeline",
                  "> end-to-end. Numbers below are meaningless; rerun with `backend: hf` on the cluster.\n"]
    lines += [
        f"# SUMMARY — Is the COMET encoder more length-sensitive than raw XLM-R?",
        "",
        f"**Task.** CAP major-topic classification ({cfg['data']['source']}), {len(cfg['data']['languages'])} languages, "
        f"21 classes. Two frozen encoders sharing the XLM-R backbone — "
        f"COMET (`{pair['comet']['ref']}`, transformer extracted from the checkpoint) vs raw "
        f"(`{pair['raw']['ref']}`) — probed with logistic regression on masked-mean features at "
        f"token-prefix lengths {cfg['length_buckets']}. Paired design: the same ≥{cfg['min_tokens']}-token "
        "documents feed every bucket, so length is the only thing that changes.",
        "",
        "## Answer",
        "",
        f"On the accuracy-vs-length curve, the COMET encoder's slope on log₂(L) is "
        f"**{s_comet:+.4f}** vs raw XLM-R's **{s_raw:+.4f}** accuracy per doubling of length. "
        f"The COMET − XLM-R accuracy gap goes from **{g_lo['gap_acc']:+.4f}** at L={int(g_lo['L'])} "
        f"to **{g_hi['gap_acc']:+.4f}** at L={int(g_hi['L'])} (Δ = {d_gap:+.4f}).",
        "",
        verdict,
        "",
        "## Key plots",
        "",
        "![accuracy vs length](figures/accuracy_vs_length.png)",
        "",
        "![COMET − XLM-R gap vs length](figures/gap_vs_length.png)",
        "",
        "## Length-sensitivity slopes (probe-per-length, lang=all)",
        "",
        _md_table(slopes),
        "",
        "## COMET − XLM-R gap by length",
        "",
        _md_table(gap),
        "",
        "## Reproduce",
        "",
        "```bash",
        "python run.py --config config.yaml          # full run (writes results.csv + all figures)",
        "```",
        "",
        "Tables: `results/results.csv`, `results/analysis/*.csv`. Figures: `results/figures/*.png`.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  SUMMARY → {out_path}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true", help="1 lang, ~200 docs, 2 buckets")
    ap.add_argument("--pair", default=None, choices=["default", "alt"], help="override encoders.active_pair")
    ap.add_argument("--backend", default=None, help="override features.backend (hf|mock)")
    ap.add_argument("--source", default=None, help="override data.source")
    ap.add_argument("--results_dir", default=None, help="override paths.results_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.pair:
        cfg["encoders"]["active_pair"] = args.pair
    if args.backend:
        cfg["features"]["backend"] = args.backend
    if args.source:
        cfg["data"]["source"] = args.source
    subsample_n = None
    if args.smoke:
        cfg["data"]["languages"] = cfg["smoke"]["languages"]
        cfg["length_buckets"] = cfg["smoke"]["length_buckets"]
        cfg["probe"]["train_short_L"] = min(cfg["length_buckets"])
        subsample_n = int(cfg["smoke"]["n_docs"])
    if args.results_dir:
        base = args.results_dir
        cfg["paths"].update(results_dir=base, data_dir=f"{base}/data",
                            feature_cache=f"{base}/feature_cache",
                            analysis_dir=f"{base}/analysis", figures_dir=f"{base}/figures")
    ensure_dirs(cfg)

    print_plan(cfg)

    from data import build_dataset
    df = build_dataset(cfg, save=True, subsample_n=subsample_n)

    from features import extract_all
    feats = extract_all(cfg, df, cfg["length_buckets"])

    from probe import run_probes, save as save_probe
    res_df, pred_df = run_probes(cfg, df, feats)
    save_probe(cfg, res_df, pred_df)

    from analyze import analyze
    analyze(cfg, feats=feats)

    from plots import make_plots
    make_plots(cfg)

    write_summary(cfg, Path(cfg["paths"]["results_dir"]) / "SUMMARY.md")
    banner("DONE")


if __name__ == "__main__":
    main()
