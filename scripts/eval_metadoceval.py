#!/usr/bin/env python3
"""
eval_metadoceval.py — document-level meta-evaluation of COMET checkpoints on the
MetaDocEval contrastive test set (Dahan, Bawden & Yvon, EAMT 2026).

Protocol (paper §4, reproduced): each document exists in an original and a
perturbed version, aligned segment by segment. For a window size w, score every
SLIDE(w,1) window — w consecutive segments concatenated, stride 1 — of both
versions, average the window scores within a document to get its document-level
score, then report **contrastive accuracy**: the proportion of documents where
the original scores higher than the perturbed one. Chance is 50%. A paired
two-sided t-test over per-document score differences accompanies it.

Sweeping w ∈ {1,3,6,9} answers the question this repo cares about: does a COMET
retrained on paragraph-level data detect discourse-level errors that the
sentence-trained one misses, and does it stop diluting them as context grows?

Note on `sentence_splitting`: that perturbation is **quality-preserving** by
design, so accuracy well above chance is a false-positive rate, not a success.
It is flagged as such in the output.

Test set (clone once, ~36MB):
    git clone https://github.com/nicolasdahan/metadoceval-testset.git

Usage:
    python scripts/eval_metadoceval.py \
        --data_dir ~/scratch/metadoceval-testset \
        --models wmt22=Unbabel/wmt22-comet-da \
                 kiwi=Unbabel/wmt22-cometkiwi-da \
                 longonly=$HOME/scratch/checkpoints/retrain/mix-frac000/<run>/checkpoints/last.ckpt \
        --windows 1 3 6 9 \
        --output results/metadoceval/accuracy.json [--wandb_project comet-retrain]

Each --models entry is `label=<hf-id-or-.ckpt-path>`. Scoring is deduplicated
across perturbation types and window sizes (original windows are shared), and
cached per model in --cache_dir, so adding a model only scores the new one.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import pickle
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# file stem -> (language pair, system); the six released (lp, system) files
FILES = {
    "en_fr_aya": ("en-fr", "aya23"),
    "en_fr_gemini": ("en-fr", "gemini"),
    "en_es_aya": ("en-es", "aya23"),
    "en_es_gemini": ("en-es", "gemini"),
    "en_de_aya": ("en-de", "aya23"),
    "en_de_gemini": ("en-de", "gemini"),
}

STRUCTURAL = ["sentence_removal", "sentence_repetition", "sentence_shuffling",
              "sentence_splitting"]
LEXICAL = ["conjunction_substitution", "tense_consistency", "lexical_consistency",
           "pronoun_swap_singular", "pronoun_swap_plural"]
# quality-preserving by design: high accuracy here is a FALSE-POSITIVE rate
QUALITY_PRESERVING = {"sentence_splitting"}
# excluded from the paper's main figure (too few instances outside en-fr)
SPARSE = {"pronoun_swap_singular", "pronoun_swap_plural"}


def official_loader(data_dir: Path):
    """Import the loader shipped with the test set (`scripts/load_data.py`) and
    use it directly, rather than re-implementing the JSON parsing here. It is
    stdlib-only, so importing it costs nothing and keeps us in step with the
    release if its schema ever changes."""
    path = data_dir / "scripts" / "load_data.py"
    if not path.exists():
        raise SystemExit(
            f"Missing {path} — clone the test set first:\n"
            f"  git clone https://github.com/nicolasdahan/metadoceval-testset.git {data_dir}")
    spec = importlib.util.spec_from_file_location("metadoceval_load_data", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def testset_commit(data_dir: Path) -> str | None:
    """Git revision of the test-set clone, recorded for provenance."""
    try:
        out = subprocess.run(["git", "-C", str(data_dir), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def readme_counts(data_dir: Path) -> dict:
    """Contrastive-pair counts advertised in the clone's own README.

    Parsed rather than hard-coded so the check maintains itself: the released
    README has disagreed with the released data for `sentence_shuffling` and
    `sentence_splitting` (commit 1ee6788 listed the measured values, d3e8dc6
    replaced them with larger ones), and this surfaces it automatically if it
    is ever fixed — or if it regresses again.
    """
    path = data_dir / "README.md"
    if not path.exists():
        return {}
    order = list(FILES)          # column order of the statistics table
    known = set(STRUCTURAL) | set(LEXICAL)
    out: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 7 or cells[0] not in known or cells[0] in out:
            continue  # first table only (contrastive pairs, not documents)
        vals = []
        for c in cells[1:7]:
            c = c.replace(",", "").replace("*", "")
            vals.append(int(c) if c.isdigit() else 0)
        out[cells[0]] = dict(zip(order, vals))
    return out


def windows(segments: list[dict], w: int) -> list[dict]:
    """SLIDE(w,1): w consecutive segments, stride 1. Documents shorter than w
    yield a single window covering the whole document."""
    if len(segments) <= w:
        return [segments]
    return [segments[i:i + w] for i in range(len(segments) - w + 1)]


def join(texts: list[str]) -> str:
    """Concatenate a window's segments. Empty segments (sentence_removal drops
    text) are skipped so the join never leaves double spaces."""
    return " ".join(t.strip() for t in texts if t and t.strip())


def build_units(data_dir: Path, window_list: list[int]) -> tuple[dict, dict]:
    """Every (file, perturbation, w, doc) -> the orig/pert window triples.

    Returns (units, counts). A unit is
      {"orig": [(src, mt, ref), ...], "pert": [...]}
    with one entry per window. Only documents carrying at least one actually
    perturbed segment (Levenshtein > 0) are kept — the contrastive documents.

    `doc_id` indexes the document within its OWN perturbation subset, not
    globally: the same doc_id in two categories is generally a different
    document (verified — segment counts and `sys` text conflict across
    categories). Documents are therefore only ever grouped within a category.
    """
    loader = official_loader(data_dir)
    units: dict = {}
    counts: dict = {}
    for stem, (lp, system) in FILES.items():
        path = data_dir / "data" / f"{stem}.json"
        if not path.exists():
            raise SystemExit(f"Missing {path} — clone the test set first "
                             "(https://github.com/nicolasdahan/metadoceval-testset)")
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
                    continue  # perturbation never fired on this document
                n_docs += 1
                n_pairs += changed
                for w in window_list:
                    wins = windows(segs, w)
                    units[(stem, pert, w, doc_id)] = {
                        "orig": [(join([e["src"] for e in win]),
                                  join([e["sys"] for e in win]),
                                  join([e["ref"] for e in win])) for win in wins],
                        "pert": [(join([e["src"] for e in win]),
                                  join([e["sys_perturbed"] for e in win]),
                                  join([e["ref"] for e in win])) for win in wins],
                    }
            counts[(lp, system, pert)] = {"docs": n_docs, "pairs": n_pairs}
    return units, counts


def score_all(model, label: str, triples: list[tuple], cache_dir: Path,
              batch_size: int, gpus: int, referenceless: bool) -> dict:
    """Score every unique (src, mt, ref) triple once; return {triple: score}."""
    h = hashlib.md5(repr(sorted(triples)).encode("utf-8")).hexdigest()[:12]
    cache = cache_dir / f"{label}__{h}__n{len(triples)}.pkl"
    if cache.exists():
        logger.info(f"  cache hit: {cache.name}")
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    if referenceless:
        data = [{"src": s, "mt": m} for s, m, _ in triples]
    else:
        data = [{"src": s, "mt": m, "ref": r} for s, m, r in triples]
    out = model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=True)
    scores = dict(zip(triples, [float(x) for x in out["scores"]]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(scores, fh)
    return scores


def accuracy(diffs: list[float]) -> dict:
    """Contrastive accuracy (original scored higher) + paired t-test."""
    from scipy.stats import ttest_rel
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n == 0:
        return {"n_docs": 0, "accuracy": None, "ties": 0,
                "mean_diff": None, "p_value": None}
    wins = int((d > 0).sum())
    ties = int((d == 0).sum())
    res = {"n_docs": n, "accuracy": wins / n, "ties": ties,
           "mean_diff": float(d.mean()), "p_value": None}
    if n >= 2 and np.std(d) > 0:
        res["p_value"] = float(ttest_rel(d, np.zeros_like(d)).pvalue)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="label=<hf-id-or-ckpt> entries.")
    ap.add_argument("--data_dir", default="~/scratch/metadoceval-testset",
                    help="Clone of nicolasdahan/metadoceval-testset.")
    ap.add_argument("--windows", nargs="+", type=int, default=[1, 3, 6, 9])
    ap.add_argument("--perturbations", nargs="+", default=None,
                    help="Default: every category present in the files.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--cache_dir", default="results/metadoceval/pred_cache")
    ap.add_argument("--output", default="results/metadoceval/accuracy.json")
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

    logger.info("Building SLIDE windows…")
    units, counts = build_units(data_dir, args.windows)
    if args.perturbations:
        keep = set(args.perturbations)
        units = {k: v for k, v in units.items() if k[1] in keep}
    uniq = sorted({t for u in units.values() for t in u["orig"] + u["pert"]})
    logger.info(f"  {len(units):,} (file,perturbation,w,doc) units → "
                f"{len(uniq):,} unique window triples to score per model")

    # Report the counts actually present, and cross-check them against the
    # clone's own README — never report results against advertised sizes.
    logger.info("\nContrastive pairs actually present (Levenshtein > 0):")
    by_pert: dict = defaultdict(lambda: {"docs": 0, "pairs": 0})
    per_file: dict = defaultdict(dict)
    for (lp, system, pert), c in counts.items():
        by_pert[pert]["docs"] += c["docs"]
        by_pert[pert]["pairs"] += c["pairs"]
    for stem, (lp, system) in FILES.items():
        for pert_key, c in counts.items():
            if pert_key[0] == lp and pert_key[1] == system:
                per_file[pert_key[2]][stem] = c["pairs"]

    advertised = readme_counts(data_dir)
    mismatches: dict = {}
    for pert in sorted(by_pert):
        tag = "  [quality-preserving]" if pert in QUALITY_PRESERVING else ""
        tag += "  [sparse — excluded from the paper's main figure]" if pert in SPARSE else ""
        claimed = advertised.get(pert)
        if claimed and sum(claimed.values()) != by_pert[pert]["pairs"]:
            mismatches[pert] = {"readme": claimed, "measured": per_file[pert]}
            tag += (f"  [README claims {sum(claimed.values())} "
                    f"— USING MEASURED COUNT]")
        logger.info(f"  {pert:<28} docs={by_pert[pert]['docs']:>5} "
                    f"pairs={by_pert[pert]['pairs']:>6}{tag}")
    if mismatches:
        logger.warning(
            "\n  WARNING: the clone's README advertises more contrastive pairs "
            f"than the data contains for: {', '.join(sorted(mismatches))}.\n"
            "  All results below use the data actually present. See the "
            "'readme_mismatches' field of the output JSON.")

    results: dict = {
        "experiment": "metadoceval_contrastive_accuracy",
        "reference": "Dahan, Bawden & Yvon, MetaDocEval, EAMT 2026",
        "test_set": "github.com/nicolasdahan/metadoceval-testset",
        "protocol": "SLIDE(w,1) windows, averaged per document; accuracy = "
                    "P[score(original) > score(perturbed)]; chance = 0.5",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "testset_commit": testset_commit(data_dir),
        "windows": args.windows,
        "counts_per_lp_system": {f"{lp}|{s}|{p}": c for (lp, s, p), c in counts.items()},
        "counts_per_perturbation": dict(by_pert),
        "readme_mismatches": mismatches,
        "quality_preserving": sorted(QUALITY_PRESERVING),
        "models": {},
    }

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, name=args.run_name,
                                   job_type="metadoceval",
                                   config={"models": models, "windows": args.windows})
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"W&B init failed: {exc}")

    from eval_length_correlation import load_comet

    for label, ref in models.items():
        logger.info(f"\n══ {label} ({ref}) ══")
        model = load_comet(ref, gpus)
        referenceless = "referenceless" in type(model).__name__.lower()
        if referenceless:
            logger.info("  reference-free model — scoring (src, mt) only")
        scores = score_all(model, label, uniq, Path(args.cache_dir),
                           args.batch_size, gpus, referenceless)

        # per-document score difference, then aggregate
        diffs: dict = defaultdict(list)   # (lp, system, pert, w) -> [orig-pert]
        for (stem, pert, w, _doc), u in units.items():
            lp, system = FILES[stem]
            o = float(np.mean([scores[t] for t in u["orig"]]))
            p = float(np.mean([scores[t] for t in u["pert"]]))
            diffs[(lp, system, pert, w)].append(o - p)

        per_cell = {f"{lp}|{system}|{pert}|w{w}": accuracy(d)
                    for (lp, system, pert, w), d in diffs.items()}

        # micro-average across language pairs and systems, per (perturbation, w)
        pooled: dict = defaultdict(list)
        for (lp, system, pert, w), d in diffs.items():
            pooled[(pert, w)].extend(d)
        micro = {f"{pert}|w{w}": accuracy(d) for (pert, w), d in pooled.items()}

        results["models"][label] = {"referenceless": referenceless,
                                    "per_cell": per_cell, "micro": micro}

        for pert in sorted({p for p, _ in pooled}):
            row = "  ".join(
                f"w={w}:{micro[f'{pert}|w{w}']['accuracy']:.3f}"
                for w in args.windows if micro.get(f"{pert}|w{w}", {}).get("accuracy") is not None)
            flag = "  (quality-preserving!)" if pert in QUALITY_PRESERVING else ""
            logger.info(f"  {pert:<28} {row}{flag}")
            if wandb_run is not None:
                for w in args.windows:
                    m = micro.get(f"{pert}|w{w}", {})
                    if m.get("accuracy") is not None:
                        wandb_run.log({f"{label}/{pert}/w{w}/accuracy": m["accuracy"]})

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plot_path = _plot(results, out_path.parent / "plots", args.windows)
    if wandb_run is not None:
        import wandb
        wandb_run.log({"plots/metadoceval_accuracy": wandb.Image(str(plot_path))})
        wandb_run.finish()
    logger.info(f"\nResults → {out_path}")


def _plot(results: dict, plot_dir: Path, window_list: list[int]) -> Path:
    """Figure 1 of the paper: accuracy vs window size, one panel per
    perturbation (structural row on top, lexical below), one line per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir.mkdir(parents=True, exist_ok=True)

    present = sorted({k.split("|")[0]
                      for m in results["models"].values() for k in m["micro"]})
    rows = [[p for p in STRUCTURAL if p in present],
            [p for p in LEXICAL if p in present and p not in SPARSE]]
    rows = [r for r in rows if r]
    ncol = max(len(r) for r in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(4 * ncol, 4 * len(rows)),
                             squeeze=False)
    for i, row in enumerate(rows):
        for j in range(ncol):
            ax = axes[i][j]
            if j >= len(row):
                ax.axis("off")
                continue
            pert = row[j]
            for label, m in results["models"].items():
                xs, ys = [], []
                for w in window_list:
                    a = m["micro"].get(f"{pert}|w{w}", {}).get("accuracy")
                    if a is not None:
                        xs.append(w); ys.append(100 * a)
                if xs:
                    ax.plot(xs, ys, marker="o",
                            ls="--" if m.get("referenceless") else "-", label=label)
            ax.axhline(50, color="grey", ls=":", lw=1)
            title = pert.replace("_", " ")
            if pert in QUALITY_PRESERVING:
                title += "\n(quality-preserving — lower is better)"
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("window size w")
            ax.set_ylabel("accuracy (%)")
            ax.set_xticks(window_list)
            ax.set_ylim(0, 100)   # shared scale: panels are read against chance
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("MetaDocEval: contrastive accuracy vs context size "
                 "(dashed = reference-free)", fontsize=12)
    out = plot_dir / "metadoceval_accuracy.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    logger.info(f"  [plot] {out}")
    return out


if __name__ == "__main__":
    main()
