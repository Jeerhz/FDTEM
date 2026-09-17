"""MetaDocEval lens: contrastive accuracy of COMET checkpoints on discourse-level perturbations.

Protocol (Dahan, Bawden & Yvon, EAMT 2026, section 4): every document exists in an original
and a perturbed version, aligned segment by segment. For a window size w, score every
SLIDE(w,1) window of both versions, average within a document, and report the proportion
of documents where the original scores higher (chance = 0.5) with a paired t-test.
`sentence_splitting` is quality-preserving by design: its "accuracy" is a false-positive rate.

Scoring is deduplicated across perturbations and window sizes and cached per model (an
entry is reused only when its `.fp` sidecar matches the checkpoint the label resolves to).

  python -m part2_length_training.load_metadoceval
  python -m part2_length_training.eval_metadoceval \
      --models da-base=Unbabel/wmt22-comet-da qe-base=Unbabel/wmt22-cometkiwi-da da-frac000=<ckpt> \
      [--windows 1 3 6 9] [--wandb_project comet-retrain-wmt]
Writes MetaDocEvalResults to results/metadoceval.json and results/plots/metadoceval_accuracy.png.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import logging
import pickle
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common.auth import init_wandb
from common.comet_models import load_comet, uses_reference
from common.paths import SCRATCH
from part2_length_training import CACHE_DIR, RESULTS_DIR
from part2_length_training.models import AccuracyCell, MetaDocEvalResults, ModelAccuracy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

FILES = {  # file stem -> (language pair, system)
    "en_fr_aya": ("en-fr", "aya23"), "en_fr_gemini": ("en-fr", "gemini"),
    "en_es_aya": ("en-es", "aya23"), "en_es_gemini": ("en-es", "gemini"),
    "en_de_aya": ("en-de", "aya23"), "en_de_gemini": ("en-de", "gemini"),
}
STRUCTURAL = ["sentence_removal", "sentence_repetition", "sentence_shuffling", "sentence_splitting"]
LEXICAL = ["conjunction_substitution", "tense_consistency", "lexical_consistency",
           "pronoun_swap_singular", "pronoun_swap_plural"]
QUALITY_PRESERVING = {"sentence_splitting"}
SPARSE = {"pronoun_swap_singular", "pronoun_swap_plural"}  # too few instances outside en-fr


def official_loader(data_dir: Path):
    """The stdlib-only loader shipped with the test set (scripts/load_data.py)."""
    path = data_dir / "scripts" / "load_data.py"
    if not path.exists():
        raise SystemExit(f"Missing {path} - run python -m part2_length_training.load_metadoceval first")
    spec = importlib.util.spec_from_file_location("metadoceval_load_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def testset_commit(data_dir: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(data_dir), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def readme_counts(data_dir: Path) -> dict:
    """Contrastive-pair counts advertised in the clone's README (they have disagreed with the data)."""
    path = data_dir / "README.md"
    if not path.exists():
        return {}
    known = set(STRUCTURAL) | set(LEXICAL)
    out: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 7 or cells[0] not in known or cells[0] in out:
            continue  # first table only (contrastive pairs, not documents)
        digits = [c.replace(",", "").replace("*", "") for c in cells[1:7]]
        out[cells[0]] = dict(zip(FILES, [int(d) if d.isdigit() else 0 for d in digits]))
    return out


def windows(segments: list[dict], w: int) -> list[list[dict]]:
    """SLIDE(w,1); a document shorter than w yields one window covering it whole."""
    if len(segments) <= w:
        return [segments]
    return [segments[i:i + w] for i in range(len(segments) - w + 1)]


def join(texts: list[str]) -> str:
    return " ".join(t.strip() for t in texts if t and t.strip())  # sentence_removal leaves empties


def build_units(data_dir: Path, window_list: list[int]) -> tuple[dict, dict]:
    """(file, perturbation, w, doc) -> {"orig": [(src, mt, ref) per window], "pert": [...]}, and pair counts.

    Only documents with at least one perturbed segment (Levenshtein > 0) are kept. `doc_id`
    indexes the document within its own perturbation subset, so documents are only ever
    grouped within a category.
    """
    loader = official_loader(data_dir)
    units: dict = {}
    counts: dict = {}
    for stem, (lp, system) in FILES.items():
        path = data_dir / "data" / f"{stem}.json"
        if not path.exists():
            raise SystemExit(f"Missing {path} - run python -m part2_length_training.load_metadoceval first")
        data = loader.load(path)
        for pert in data:
            by_doc: dict = defaultdict(list)
            for e in loader.iter_entries(data, perturbation=pert):
                by_doc[e["doc_id"]].append(e)
            n_docs = n_pairs = 0
            for doc_id, segs in by_doc.items():
                segs = sorted(segs, key=lambda e: e["seg_ids"][0])
                changed = sum(1 for e in segs if e["Levenshtein"] > 0)
                if changed == 0:
                    continue
                n_docs += 1
                n_pairs += changed
                for w in window_list:
                    wins = windows(segs, w)
                    units[(stem, pert, w, doc_id)] = {
                        "orig": [(join([e["src"] for e in win]), join([e["sys"] for e in win]),
                                  join([e["ref"] for e in win])) for win in wins],
                        "pert": [(join([e["src"] for e in win]), join([e["sys_perturbed"] for e in win]),
                                  join([e["ref"] for e in win])) for win in wins]}
            counts[(lp, system, pert)] = {"docs": n_docs, "pairs": n_pairs}
    return units, counts


def score_all(model, label: str, triples: list[tuple], cache_dir: Path, batch_size: int, gpus: int,
              referenceless: bool) -> dict:
    """Score every unique (src, mt, ref) triple once; {triple: score}, cached per (label, checkpoint)."""
    h = hashlib.md5(repr(sorted(triples)).encode("utf-8")).hexdigest()[:12]
    cache = cache_dir / f"{label}__{h}__n{len(triples)}.pkl"
    fp = getattr(model, "_fdtem_fingerprint", None)
    fp_file = cache.with_suffix(".pkl.fp")
    if cache.exists():
        seen = fp_file.read_text().strip() if fp_file.exists() else None
        if fp is None or seen == fp:
            logger.info(f"  cache hit: {cache.name}")
            with open(cache, "rb") as fh:
                return pickle.load(fh)
        logger.warning(f"  {label}: cached scores are from a different checkpoint ({seen or 'unrecorded'} != {fp}) - rescoring")
    if referenceless:
        data = [{"src": s, "mt": m} for s, m, _ in triples]
    else:
        data = [{"src": s, "mt": m, "ref": r} for s, m, r in triples]
    out = model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=True)
    scores = dict(zip(triples, [float(x) for x in out["scores"]]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(scores, fh)
    if fp is not None:
        fp_file.write_text(fp)
    return scores


def accuracy(diffs: list[float]) -> AccuracyCell:
    """Contrastive accuracy (original scored higher) + paired t-test."""
    from scipy.stats import ttest_rel
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n == 0:
        return AccuracyCell(n_docs=0, accuracy=None, ties=0, mean_diff=None, p_value=None)
    p_value = float(ttest_rel(d, np.zeros_like(d)).pvalue) if n >= 2 and np.std(d) > 0 else None
    return AccuracyCell(n_docs=n, accuracy=int((d > 0).sum()) / n, ties=int((d == 0).sum()),
                        mean_diff=float(d.mean()), p_value=p_value)


def plot(results: MetaDocEvalResults, plot_dir: Path, window_list: list[int]) -> Path:
    """Figure 1 of the paper: accuracy vs window size, one panel per perturbation, one line per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir.mkdir(parents=True, exist_ok=True)

    present = sorted({k.split("|")[0] for m in results.models.values() for k in m.micro})
    rows = [[p for p in STRUCTURAL if p in present], [p for p in LEXICAL if p in present and p not in SPARSE]]
    rows = [r for r in rows if r]
    ncol = max(len(r) for r in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(4 * ncol, 4 * len(rows)), squeeze=False)
    for i, row in enumerate(rows):
        for j in range(ncol):
            ax = axes[i][j]
            if j >= len(row):
                ax.axis("off")
                continue
            pert = row[j]
            for label, m in results.models.items():
                pts = [(w, 100 * m.micro[f"{pert}|w{w}"].accuracy) for w in window_list
                       if f"{pert}|w{w}" in m.micro and m.micro[f"{pert}|w{w}"].accuracy is not None]
                if pts:
                    ax.plot(*zip(*pts), marker="o", ls="--" if m.referenceless else "-", label=label)
            ax.axhline(50, color="grey", ls=":", lw=1)
            title = pert.replace("_", " ")
            if pert in QUALITY_PRESERVING:
                title += "\n(quality-preserving - lower is better)"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("window size w")
            ax.set_ylabel("accuracy (%)")
            ax.set_xticks(window_list)
            ax.set_ylim(0, 100)  # shared scale: panels are read against chance
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("MetaDocEval: contrastive accuracy vs context size (dashed = reference-free)", fontsize=12)
    out = plot_dir / "metadoceval_accuracy.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info(f"  [plot] {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="label=<hf-id-or-ckpt>")
    ap.add_argument("--data_dir", default=str(SCRATCH / "metadoceval-testset"))
    ap.add_argument("--windows", nargs="+", type=int, default=[1, 3, 6, 9])
    ap.add_argument("--perturbations", nargs="+", default=None, help="default: every category present")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--cache_dir", default=str(CACHE_DIR / "metadoceval"))
    ap.add_argument("--output", default=str(RESULTS_DIR / "metadoceval.json"))
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--run_name", default="metadoceval")
    args = ap.parse_args()

    import torch
    gpus = args.gpus if args.gpus is not None else (1 if torch.cuda.is_available() else 0)
    data_dir = Path(args.data_dir).expanduser()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    models = {}
    for entry in args.models:
        label, _, ref = entry.partition("=")
        if not ref:
            raise SystemExit(f"--models entry {entry!r} must be label=<ref>")
        models[label] = ref

    logger.info("Building SLIDE windows")
    units, counts = build_units(data_dir, args.windows)
    if args.perturbations:
        units = {k: v for k, v in units.items() if k[1] in set(args.perturbations)}
    uniq = sorted({t for u in units.values() for t in u["orig"] + u["pert"]})
    logger.info(f"  {len(units):,} (file,perturbation,w,doc) units -> {len(uniq):,} unique window triples per model")

    # the counts actually present, cross-checked against the clone's README
    logger.info("\nContrastive pairs actually present (Levenshtein > 0):")
    by_pert: dict = defaultdict(lambda: {"docs": 0, "pairs": 0})
    per_file: dict = defaultdict(dict)
    for (lp, system, pert), c in counts.items():
        by_pert[pert]["docs"] += c["docs"]
        by_pert[pert]["pairs"] += c["pairs"]
        stem = next(s for s, v in FILES.items() if v == (lp, system))
        per_file[pert][stem] = c["pairs"]
    advertised = readme_counts(data_dir)
    mismatches: dict = {}
    for pert in sorted(by_pert):
        tag = "  [quality-preserving]" if pert in QUALITY_PRESERVING else ""
        tag += "  [sparse - excluded from the paper's main figure]" if pert in SPARSE else ""
        claimed = advertised.get(pert)
        if claimed and sum(claimed.values()) != by_pert[pert]["pairs"]:
            mismatches[pert] = {"readme": claimed, "measured": per_file[pert]}
            tag += f"  [README claims {sum(claimed.values())} - USING MEASURED COUNT]"
        logger.info(f"  {pert:<28} docs={by_pert[pert]['docs']:>5} pairs={by_pert[pert]['pairs']:>6}{tag}")
    if mismatches:
        logger.warning(f"\n  WARNING: the README advertises more contrastive pairs than the data contains for: "
                       f"{', '.join(sorted(mismatches))}. Results use the data actually present.")

    results = MetaDocEvalResults(
        experiment="metadoceval_contrastive_accuracy",
        reference="Dahan, Bawden & Yvon, MetaDocEval, EAMT 2026",
        test_set="github.com/nicolasdahan/metadoceval-testset",
        protocol="SLIDE(w,1) windows, averaged per document; accuracy = P[score(original) > score(perturbed)]; chance = 0.5",
        timestamp=datetime.now(timezone.utc).isoformat(), testset_commit=testset_commit(data_dir),
        windows=args.windows,
        counts_per_lp_system={f"{lp}|{s}|{p}": c for (lp, s, p), c in counts.items()},
        counts_per_perturbation=dict(by_pert), readme_mismatches=mismatches,
        quality_preserving=sorted(QUALITY_PRESERVING), models={})

    wandb_run = init_wandb(args.wandb_project, args.run_name, "metadoceval",
                           config={"models": models, "windows": args.windows})

    for label, ref in models.items():
        logger.info(f"\n== {label} ({ref}) ==")
        model = load_comet(ref)
        referenceless = not uses_reference(model)
        if referenceless:
            logger.info("  reference-free model - scoring (src, mt) only")
        scores = score_all(model, label, uniq, Path(args.cache_dir), args.batch_size, gpus, referenceless)

        diffs: dict = defaultdict(list)  # (lp, system, pert, w) -> [orig - pert per document]
        for (stem, pert, w, _doc), u in units.items():
            lp, system = FILES[stem]
            o = float(np.mean([scores[t] for t in u["orig"]]))
            p = float(np.mean([scores[t] for t in u["pert"]]))
            diffs[(lp, system, pert, w)].append(o - p)
        per_cell = {f"{lp}|{system}|{pert}|w{w}": accuracy(d) for (lp, system, pert, w), d in diffs.items()}
        pooled: dict = defaultdict(list)  # micro-average across language pairs and systems
        for (lp, system, pert, w), d in diffs.items():
            pooled[(pert, w)].extend(d)
        micro = {f"{pert}|w{w}": accuracy(d) for (pert, w), d in pooled.items()}
        results.models[label] = ModelAccuracy(referenceless=referenceless, per_cell=per_cell, micro=micro)

        for pert in sorted({p for p, _ in pooled}):
            row = "  ".join(f"w={w}:{micro[f'{pert}|w{w}'].accuracy:.3f}" for w in args.windows
                            if f"{pert}|w{w}" in micro and micro[f"{pert}|w{w}"].accuracy is not None)
            logger.info(f"  {pert:<28} {row}{'  (quality-preserving!)' if pert in QUALITY_PRESERVING else ''}")
            if wandb_run is not None:
                for w in args.windows:
                    m = micro.get(f"{pert}|w{w}")
                    if m is not None and m.accuracy is not None:
                        wandb_run.log({f"{label}/{pert}/w{w}/accuracy": m.accuracy})

        out_path.write_text(results.model_dump_json(indent=2))  # incremental
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plot_path = plot(results, out_path.parent / "plots", args.windows)
    if wandb_run is not None:
        import wandb
        wandb_run.log({"plots/metadoceval_accuracy": wandb.Image(str(plot_path))})
        wandb_run.finish()
    logger.info(f"\nResults -> {out_path}")


if __name__ == "__main__":
    main()
