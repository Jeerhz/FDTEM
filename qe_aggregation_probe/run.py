#!/usr/bin/env python3
"""
run.py — one command reproduces every table and figure (Exp 0).

    python run.py --config config.yaml                  # full run (cluster)
    python run.py --config config.yaml --smoke          # mock metric + rule
                                                        # perturbations + synthetic
                                                        # data: parameter-recovery
                                                        # logic test, no GPU/API
    python run.py --config config.yaml --metrics cometkiwi   # single metric

Pipeline: print plan -> load docs -> build k-blocks -> perturb (cached) ->
assemble counterbalanced items -> score sentences + paragraphs (cached) ->
tidy dataframe -> fit candidate aggregation functions -> controls -> plots ->
SUMMARY.md with one verdict sentence per metric.
"""
from __future__ import annotations

import argparse
import logging
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from common import all_conditions, banner, ensure_dirs, load_config

logger = logging.getLogger(__name__)


def print_plan(cfg: dict, metrics: list) -> None:
    banner("Experiment plan — Exp 0: estimate the aggregation function")
    print("  Question: score(paragraph) ≈ f(score(sent_1..k)) — mean, min, penalty-sum,")
    print("    or a power mean M_p? The fitted exponent p is the answer; the dilution")
    print("    curve Δ(m,k,severity) becomes the null model of the length-bias study.")
    print(f"\n  Data        : {cfg['data']['source']}  (resegment={cfg['data'].get('resegment')})")
    print(f"  k / blocks  : k={cfg['design']['k_list']}, {cfg['design']['n_blocks_per_k']} blocks per (lang,k)")
    print(f"  Conditions  : {all_conditions(cfg)}  × m=0..k, positions counterbalanced")
    print(f"  Perturber   : {cfg['perturb']['backend']}"
          + (f" ({cfg['perturb']['llm']['model']})" if cfg['perturb']['backend'] == 'llm' else ""))
    print(f"  Metrics     : {[m['name'] for m in metrics]}")
    print(f"  Fit         : affine={cfg['fit']['affine']}, CV folds={cfg['fit']['cv_folds']} (by doc), "
          f"bootstrap n={cfg['fit']['bootstrap']['n_resamples']}")
    print(f"  Token limit : {cfg['design']['max_tokens']} (drop={cfg['design']['drop_truncated']})")
    print("\n  Environment:")
    print(f"    python={platform.python_version()}  numpy={np.__version__}  pandas={pd.__version__}")
    try:
        import torch  # noqa: PLC0415
        print(f"    torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    except Exception:
        print("    torch=not installed (mock/logic mode)")


def fmt_p(triple) -> str:
    p, lo, hi = triple
    if p is None:
        return "n/a"
    s = f"{p:+.2f}"
    if lo is not None and hi is not None:
        s += f" [{lo:+.2f}, {hi:+.2f}]"
    return s


def write_summary(cfg: dict, summaries: dict, out: Path) -> None:
    mock = any(m["name"] == "mock" for m in cfg["metrics"]
               if m["name"] in summaries)
    lines = ["# SUMMARY — Exp 0: what aggregation function do QE metrics implement?", ""]
    if mock:
        true_p = next((m.get("true_p") for m in cfg["metrics"] if m["name"] == "mock"), None)
        lines += [f"> ⚠️ Contains the **mock** metric (logic smoke test). Its paragraph scores were",
                  f"> SYNTHESISED as a power mean with true p = {true_p}; the fitted p̂ below must",
                  "> recover it. Mock numbers are a pipeline check, not a scientific result.", ""]
    lines += [
        "**Protocol.** k-sentence paragraphs from contiguous, same-document WMT24++ "
        "post-edits; m ∈ {0..k} sentences perturbed with controlled "
        "severity×family errors (positions counterbalanced); s_i = score of sentence i "
        "alone, S = score of the concatenated paragraph; candidate aggregation "
        "functions fitted with affine calibration and compared by document-grouped "
        "cross-validated R².", "",
        "## Verdicts", ""]
    for metric, s in summaries.items():
        pk = ", ".join(f"k={k}: {fmt_p(v)}" for k, v in sorted(s["p_by_k"].items()))
        lines += [
            f"- **{metric}** ({s['kind']} scale): best-fitting aggregation is "
            f"**{s['best_candidate']}** (CV R² = {s['best_cv_r2']:.3f}); power-mean "
            f"exponent {pk}; "
            f"position-{'SENSITIVE' if s['position_sensitive'] else 'invariant'}; "
            f"p̂ {'DRIFTS' if s['p_drifts_across_k'] else 'stable'} across k."]
    lines += [
        "", "## Key outputs", "",
        "- `analysis/candidate_fits.csv` — every candidate, every (metric, k)",
        "- `analysis/p_by_k.csv` — the headline exponent p̂ with bootstrap CIs",
        "- `analysis/dilution.csv` + `figures/dilution_*.png` — the null-model curve Δ(m,k,severity)",
        "- `analysis/residual_by_k.csv` — what aggregation does NOT explain (length effect)",
        "- `figures/` — obs-vs-pred, p-by-k, candidate comparison", "",
        "## Reproduce", "", "```bash", "python run.py --config config.yaml", "```"]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  SUMMARY → {out}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="synthetic data + rule perturbations + mock metric")
    ap.add_argument("--source", default=None, help="override data.source")
    ap.add_argument("--perturb_backend", default=None, choices=["llm", "rule"])
    ap.add_argument("--metrics", nargs="+", default=None,
                    help="enable only these metric names")
    ap.add_argument("--results_dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        sm = cfg["smoke"]
        cfg["data"]["source"] = sm["source"]
        cfg["data"]["synthetic_langs"] = sm["langs"]
        cfg["design"]["k_list"] = sm["k_list"]
        cfg["design"]["n_blocks_per_k"] = sm["n_blocks_per_k"]
        cfg["design"]["token_counter"] = None
        cfg["perturb"]["backend"] = sm["perturb_backend"]
        cfg["fit"]["bootstrap"]["n_resamples"] = sm["bootstrap_n"]
        for m in cfg["metrics"]:
            m["enabled"] = m["name"] in sm["metrics_enabled"]
        if not args.results_dir:
            args.results_dir = "results_smoke"
    if args.source:
        cfg["data"]["source"] = args.source
    if args.perturb_backend:
        cfg["perturb"]["backend"] = args.perturb_backend
    if args.metrics:
        for m in cfg["metrics"]:
            m["enabled"] = m["name"] in args.metrics
    if args.results_dir:
        base = args.results_dir
        cfg["paths"] = {"results_dir": base, "data_dir": f"{base}/data",
                        "perturb_cache": f"{base}/perturb_cache",
                        "score_cache": f"{base}/score_cache",
                        "analysis_dir": f"{base}/analysis",
                        "figures_dir": f"{base}/figures"}
    ensure_dirs(cfg)

    metrics = [m for m in cfg["metrics"] if m.get("enabled")]
    if not metrics:
        raise SystemExit("No metric enabled — check config metrics[].enabled / --metrics")
    print_plan(cfg, metrics)

    banner("1/5 Data")
    from data import build_blocks, load_docs
    docs = load_docs(cfg)
    blocks = build_blocks(cfg, docs)

    banner("2/5 Perturbations")
    from perturb import Perturber
    from items import assemble_df, build_items, build_jobs
    perturber = Perturber(cfg)
    items = build_items(cfg, blocks, perturber)
    perturber.flush()
    jobs = build_jobs(items)

    banner("3/5 Scoring")
    from scoring import score_jobs
    k_max = max(cfg["design"]["k_list"])
    dfs = []
    for mcfg in metrics:
        scores = score_jobs(cfg, mcfg, jobs)
        dfs.append(assemble_df(items, mcfg["name"], scores, k_max))
    df = pd.concat(dfs, ignore_index=True)
    df_path = Path(cfg["paths"]["results_dir"]) / "items.csv"
    df.to_csv(df_path, index=False)
    print(f"  tidy dataframe → {df_path}  ({len(df)} rows)")

    banner("4/5 Fitting the aggregation function")
    from analyze import analyze
    summaries = analyze(cfg, df)

    banner("5/5 Plots + summary")
    from plots import make_plots
    make_plots(cfg, df)
    write_summary(cfg, summaries, Path(cfg["paths"]["results_dir"]) / "SUMMARY.md")
    banner("DONE")


if __name__ == "__main__":
    main()
