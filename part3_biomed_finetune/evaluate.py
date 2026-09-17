"""Base wmt22-comet-da vs the Bio-MQM fine-tuned checkpoint, per language pair.

Scores every <lp>_val.csv of --data_dir with both models (predictions cached
under --cache_dir), reports Pearson / Spearman / Kendall per pair and the mean
over pairs, and a paired document bootstrap of delta Kendall (bio - base).

    python -m part3_biomed_finetune.evaluate --bio auto --data_dir ~/scratch/bio_mqm \
        --ckpt_dir ~/scratch/checkpoints/bio_mqm

Writes results/correlation.json (BioEvalResults) and results/plots/kendall_by_lp.png.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common.paths import CHECKPOINTS, SCRATCH
from common.stats import CorrelationCell, bootstrap_delta, correlations, doc_units
from part3_biomed_finetune import CACHE_DIR, RESULTS_DIR
from part3_biomed_finetune.models import BioEvalResults, MeanCorrelation

BASE = "base"


def val_files(data_dir: Path) -> dict[str, Path]:
    files = {f.name[: -len("_val.csv")]: f for f in sorted(data_dir.glob("*_val.csv"))}
    files.pop("all", None)
    if not files:
        raise SystemExit(f"no <lp>_val.csv in {data_dir}")
    return files


def resolve_bio(ref: str, ckpt_dir: Path) -> str:
    if ref != "auto":
        return ref
    from common.comet_models import best_checkpoint
    ckpt = best_checkpoint(ckpt_dir)
    if ckpt is None:
        raise SystemExit(f"no checkpoint under {ckpt_dir}")
    print(f"bio = {ckpt}")
    return str(ckpt)


def score_all(models: dict[str, str], files: dict[str, Path], cache_dir: Path,
              batch_size: int, gpus: int) -> dict[str, dict[str, pd.DataFrame]]:
    """label -> lp -> the validation frame with a `pred` column."""
    from common.comet_models import load_comet, score

    scored: dict[str, dict[str, pd.DataFrame]] = {}
    for label, ref in models.items():
        model = load_comet(ref)
        scored[label] = {}
        for lp, path in files.items():
            df = pd.read_csv(path)
            df["pred"] = score(model, label, df, cache_dir, batch_size=batch_size, gpus=gpus)
            scored[label][lp] = df
        del model
    return scored


def mean_over_lps(cells: dict[str, CorrelationCell]) -> MeanCorrelation:
    def mean(name: str) -> float | None:
        values = [getattr(c, name) for c in cells.values() if getattr(c, name) is not None]
        return float(np.mean(values)) if values else None
    return MeanCorrelation(n_lps=len(cells), pearson=mean("pearson"),
                          spearman=mean("spearman"), kendall=mean("kendall"))


def evaluate(scored: dict[str, dict[str, pd.DataFrame]], data_dir: Path, n_boot: int) -> BioEvalResults:
    cells = {label: {lp: correlations(df["score"].to_numpy(float), df["pred"].to_numpy(float))
                     for lp, df in by_lp.items()} for label, by_lp in scored.items()}
    delta = {}
    for label, by_lp in scored.items():
        if label == BASE:
            continue
        delta[label] = {lp: bootstrap_delta(doc_units(df), doc_units(scored[BASE][lp]), n_boot=n_boot)
                        for lp, df in by_lp.items()}
    return BioEvalResults(timestamp=datetime.now(timezone.utc).isoformat(), data_dir=str(data_dir),
                          models=cells, mean={label: mean_over_lps(c) for label, c in cells.items()},
                          delta_vs_base=delta)


def print_table(results: BioEvalResults) -> None:
    labels = list(results.models)
    others = [l for l in labels if l != BASE]
    fmt = lambda v: f"{v:+.3f}" if v is not None else "   -  "  # noqa: E731
    head = "lp      n    " + "  ".join(f"{l + ' r':>9} {l + ' tau':>9}" for l in labels)
    head += "".join(f"   d_tau({l}-{BASE}) [95% CI]" for l in others)
    print(head)
    for lp in results.models[BASE]:
        row = f"{lp:6} {results.models[BASE][lp].n:5}  "
        row += "  ".join(f"{fmt(results.models[l][lp].pearson):>9} {fmt(results.models[l][lp].kendall):>9}"
                         for l in labels)
        for l in others:
            d, lo, hi = results.delta_vs_base[l][lp]
            row += f"   {d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        print(row)
    row = "mean         "
    row += "  ".join(f"{fmt(results.mean[l].pearson):>9} {fmt(results.mean[l].kendall):>9}" for l in labels)
    print(row)


def plot_kendall_by_lp(results: BioEvalResults, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results.models)
    lps = list(results.models[BASE])
    x = np.arange(len(lps))
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(lps), 3.6))
    for i, label in enumerate(labels):
        taus = [results.models[label][lp].kendall or np.nan for lp in lps]
        ax.bar(x + (i - (len(labels) - 1) / 2) * width, taus, width, label=label)
    ax.set_xticks(x, lps)
    ax.set_ylabel("Kendall tau")
    ax.set_title("Bio-MQM validation split")
    ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Unbabel/wmt22-comet-da")
    ap.add_argument("--bio", default="auto", help="fine-tuned .ckpt, or 'auto' = best checkpoint under --ckpt_dir")
    ap.add_argument("--ckpt_dir", default=str(CHECKPOINTS / "bio_mqm"))
    ap.add_argument("--data_dir", default=str(SCRATCH / "bio_mqm"))
    ap.add_argument("--cache_dir", default=str(CACHE_DIR))
    ap.add_argument("--output", default=str(RESULTS_DIR / "correlation.json"))
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=1)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    models = {BASE: args.base, "bio": resolve_bio(args.bio, Path(args.ckpt_dir).expanduser())}
    scored = score_all(models, val_files(data_dir), Path(args.cache_dir).expanduser(),
                       args.batch_size, args.gpus)
    results = evaluate(scored, data_dir, args.n_boot)
    print_table(results)

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(results.model_dump_json(indent=2))
    plot_kendall_by_lp(results, out.parent / "plots" / "kendall_by_lp.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
