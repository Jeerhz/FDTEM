#!/usr/bin/env python3
"""
run_e2_error_sensitivity.py — E2: alignment vs error-discrimination trade-off.

This is the experiment that targets the supervisor's question head-on. A
translation-quality encoder has to do two things at once:

  pull-together : place a source next to its *correct* translation
                  (cos(src, tgt⁺) high)        — what LaBSE/LASER/E5 optimise.
  push-apart    : place a source *away* from a translation that contains an
                  error (cos(src, tgt⁻) lower than cos(src, tgt⁺))
                                               — unique to a quality objective.

For each FLORES src–tgt pair we build hard negatives tgt⁻ by injecting one
error-like perturbation into the reference translation (number / omission /
reordering / mistranslation — xsim++ style), and measure, per encoder:

  align_acc   P[ cos(src, tgt⁺) > cos(src, tgt_random) ]   (pull-together)
  err_acc     P[ cos(src, tgt⁺) > cos(src, tgt⁻) ]         (push-apart)
  err_margin  mean( cos(src,tgt⁺) − cos(src,tgt⁻) )

Plotting err_acc against align_acc gives the money plot: aligners cluster top-left
(great alignment, weak error sensitivity); COMET should sit higher on err_acc.

Usage
-----
  python scripts/run_e2_error_sensitivity.py \
      --encoders comet:Unbabel/wmt22-comet-da "comet:$BIO_CKPT" \
                 hf-mean:xlm-roberta-large labse e5 \
      --langs en de es fr ru zh --pivot en \
      --output results/repana/e2_error_sensitivity.json
"""
import argparse
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from repana_common import (
    CORE_LANGS, PERTURBATIONS, build_embedder, cached_embed, enc_tag as _enc_tag,
    load_flores_source, perturb, pick_device,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def build_negatives(sentences: List[str], lang: str, kinds: List[str], seed: int):
    """For each sentence produce one perturbed variant per kind.

    Returns dict kind -> (texts, valid_mask) where unparseable items reuse the
    original text and are masked out of the metrics.
    """
    rng = random.Random(seed)
    out = {}
    for kind in kinds:
        texts, valid = [], []
        for s in sentences:
            p = perturb(s, lang, kind, rng)
            if p is None:
                texts.append(s); valid.append(False)
            else:
                texts.append(p); valid.append(True)
        out[kind] = (texts, np.array(valid))
    return out


def row_cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine for L2-normalised A,B → (N,)."""
    return (A * B).sum(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoders", nargs="+", required=True)
    p.add_argument("--langs", nargs="+", default=CORE_LANGS)
    p.add_argument("--pivot", default="en")
    p.add_argument("--perturbations", nargs="+", default=PERTURBATIONS, choices=PERTURBATIONS)
    p.add_argument("--flores_source", choices=["plus", "raw"], default="raw",
                   help="'plus' = official OLDI FLORES+ (HF, gated); "
                        "'raw' = legacy flores200_dataset text files.")
    p.add_argument("--split", default="devtest")
    p.add_argument("--max_sents", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--cache_dir", default="results/repana/emb_cache")
    p.add_argument("--output", default="results/repana/e2_error_sensitivity.json")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--run_name", default=None,
                   help="W&B run name. Default: '<source>_<split>_xsimpp_error_sensitivity'.")
    args = p.parse_args()

    device = pick_device(args.device)
    logger.info(f"Device: {device}")

    data = load_flores_source(args.flores_source, args.langs, args.split, args.max_sents)
    pivot = args.pivot
    tgt_langs = [l for l in data.langs if l != pivot]

    # Pre-build hard negatives once (encoder-independent).
    negatives = {l: build_negatives(data.sentences[l], l, args.perturbations, args.seed)
                 for l in tgt_langs}
    # Fixed random alignment-negative permutation (derangement-ish).
    rng = np.random.default_rng(args.seed)
    rand_perm = {l: rng.permutation(data.n) for l in tgt_langs}

    run_name = args.run_name or f"{args.flores_source}_{args.split}_xsimpp_error_sensitivity"
    enc_tags = [_enc_tag(s) for s in args.encoders]
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project, name=run_name,
                group=f"{args.flores_source}_{args.split}", job_type="xsimpp_error_sensitivity",
                tags=[f"flores:{args.flores_source}", f"split:{args.split}",
                      f"langs:{'-'.join(data.langs)}"] + enc_tags,
                config=vars(args))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"W&B init failed: {exc}")

    results: Dict = {
        "experiment": "E2_error_sensitivity",
        "run_name": run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "langs": data.langs, "pivot": pivot, "perturbations": args.perturbations,
        "flores_source": args.flores_source, "split": args.split,
        "encoders": {},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for spec in args.encoders:
        logger.info(f"\n══ {spec} ══")
        emb = build_embedder(spec, device)
        src = cached_embed(emb, data.sentences[pivot], f"flores_k1_{pivot}",
                           args.cache_dir, args.batch_size)

        align_accs, err_accs_by_kind = [], {k: [] for k in args.perturbations}
        per_lang = {}
        for l in tgt_langs:
            tgt = cached_embed(emb, data.sentences[l], f"flores_k1_{l}",
                               args.cache_dir, args.batch_size)
            cos_pos = row_cos(src, tgt)                         # cos(src, tgt⁺)
            cos_rand = row_cos(src, tgt[rand_perm[l]])          # cos(src, random tgt)
            align_acc = float((cos_pos > cos_rand).mean())
            align_accs.append(align_acc)

            lang_entry = {"align_acc": align_acc, "perturbations": {}}
            for kind in args.perturbations:
                neg_texts, valid = negatives[l][kind]
                neg = emb.embed(neg_texts, args.batch_size)     # perturbed (no cache: text-specific)
                cos_neg = row_cos(src, neg)
                m = valid
                if m.sum() == 0:
                    continue
                err_acc = float((cos_pos[m] > cos_neg[m]).mean())
                err_margin = float((cos_pos[m] - cos_neg[m]).mean())
                lang_entry["perturbations"][kind] = {
                    "err_acc": err_acc, "err_margin": err_margin, "n_valid": int(m.sum())}
                err_accs_by_kind[kind].append(err_acc)
            per_lang[l] = lang_entry
            logger.info(f"  [{l}] align={align_acc:.3f}  " +
                        "  ".join(f"{k}={lang_entry['perturbations'].get(k, {}).get('err_acc', float('nan')):.3f}"
                                  for k in args.perturbations))

        mean_align = float(np.mean(align_accs))
        mean_err_by_kind = {k: (float(np.mean(v)) if v else None) for k, v in err_accs_by_kind.items()}
        valid_kinds = [v for v in mean_err_by_kind.values() if v is not None]
        mean_err = float(np.mean(valid_kinds)) if valid_kinds else None
        results["encoders"][emb.name] = {
            "mean_align_acc": mean_align,
            "mean_err_acc": mean_err,
            "mean_err_acc_by_kind": mean_err_by_kind,
            "per_lang": per_lang,
        }
        logger.info(f"  ► {emb.name}: align={mean_align:.4f}  err={mean_err:.4f}")
        if wandb_run is not None:
            log = {f"{emb.name}/align_acc": mean_align, f"{emb.name}/err_acc": mean_err}
            log.update({f"{emb.name}/err_acc/{k}": v for k, v in mean_err_by_kind.items() if v is not None})
            wandb_run.log(log)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        del emb
        import torch, gc
        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    _plot(results, out_path.parent / "plots")
    if wandb_run is not None:
        wandb_run.finish()
    logger.info(f"\nResults → {out_path}")


def _plot(results: Dict, plot_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir.mkdir(parents=True, exist_ok=True)

    # money plot: alignment (x) vs error-discrimination (y)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, r in results["encoders"].items():
        if r["mean_err_acc"] is None:
            continue
        ax.scatter(r["mean_align_acc"], r["mean_err_acc"], s=90)
        ax.annotate(name, (r["mean_align_acc"], r["mean_err_acc"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_xlabel("alignment: P[cos(src,tgt⁺) > cos(src,random)]  (pull parallel together)")
    ax.set_ylabel("error sensitivity: P[cos(src,tgt⁺) > cos(src,tgt⁻)]  (push errors apart)")
    ax.set_title("E2 — alignment vs error-discrimination trade-off")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = plot_dir / "e2_tradeoff.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [plot] {out}")

    # per-perturbation breakdown
    kinds = results["perturbations"]
    names = [n for n, r in results["encoders"].items() if r["mean_err_acc"] is not None]
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.6), 5))
    x = np.arange(len(names))
    w = 0.8 / max(1, len(kinds))
    for i, k in enumerate(kinds):
        ys = [results["encoders"][n]["mean_err_acc_by_kind"].get(k) or 0 for n in names]
        ax.bar(x + i * w, ys, w, label=k)
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_xticks(x + w * (len(kinds) - 1) / 2)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("error-discrimination accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("E2 — error sensitivity by perturbation type")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out2 = plot_dir / "e2_by_perturbation.png"
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [plot] {out2}")


if __name__ == "__main__":
    main()
