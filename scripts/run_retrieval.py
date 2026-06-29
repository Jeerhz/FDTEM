#!/usr/bin/env python3
"""
run_retrieval.py — cross-lingual retrieval (pull-together alignment) + length scaling.

Alignment.  How well does each encoder place *parallel* sentences nearest
to each other across languages? We embed FLORES-200 devtest in every language
and run LASER-style xsim retrieval (en↔xx, both directions). Low error = the
encoder pulls parallel sentences together. This is the objective LaBSE/LASER/E5
optimise directly; COMET only ever sees it as a by-product of quality
prediction, so the gap here is informative.

Length scaling.  COMET is trained on short segments. We rebuild the *same*
retrieval at increasing text lengths by concatenating k consecutive same-article
FLORES sentences (k = 1,2,4,8) into parallel pseudo-paragraphs, and watch how
each encoder's alignment degrades as text grows past sentence scale.

Outputs JSON + (optionally) W&B + a length-scaling plot.

Usage
-----
  python scripts/run_retrieval.py \
      --encoders comet:Unbabel/wmt22-comet-da \
                 "comet:$HOME/scratch/checkpoints/bio_mqm/.../epoch=1-step=20-val_kendall=0.366.ckpt" \
                 hf-mean:xlm-roberta-large labse e5 \
      --langs en de es fr ru zh \
      --lengths 1 2 4 8 \
      --output results/representation_analysis/retrieval.json
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from representation_analysis_common import (
    CORE_LANGS, build_embedder, build_pseudo_docs, cached_embed, load_flores,
    pick_device, xsim_error,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def retrieval_for_length(emb, docs: Dict[str, List[str]], pivot: str,
                         margin: str, batch_size: int, cache_dir, klen: int) -> Dict:
    """xsim error for pivot↔xx (both directions) at one text length."""
    langs = list(docs.keys())
    vecs = {l: cached_embed(emb, docs[l], f"flores_k{klen}_{l}", cache_dir, batch_size)
            for l in langs}
    pivot_vec = vecs[pivot]
    per_pair = {}
    errs = []
    for l in langs:
        if l == pivot:
            continue
        e_fwd, _ = xsim_error(pivot_vec, vecs[l], margin)   # en -> xx
        e_bwd, _ = xsim_error(vecs[l], pivot_vec, margin)   # xx -> en
        avg = (e_fwd + e_bwd) / 2
        per_pair[f"{pivot}-{l}"] = {"fwd": e_fwd, "bwd": e_bwd, "avg": avg}
        errs.append(avg)
    return {"per_pair": per_pair, "mean_xsim_error": float(np.mean(errs)),
            "mean_acc": float(1 - np.mean(errs)), "n_items": len(next(iter(vecs.values())))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoders", nargs="+", required=True)
    p.add_argument("--langs", nargs="+", default=CORE_LANGS)
    p.add_argument("--pivot", default="en")
    p.add_argument("--lengths", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--margin", choices=["cosine", "ratio"], default="ratio")
    p.add_argument("--split", default="devtest")
    p.add_argument("--max_sents", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--cache_dir", default="results/representation_analysis/emb_cache")
    p.add_argument("--output", default="results/representation_analysis/retrieval.json")
    p.add_argument("--wandb_project", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    logger.info(f"Device: {device}")

    data = load_flores(args.langs, args.split, args.max_sents)
    docs_by_len = {k: build_pseudo_docs(data, k) for k in args.lengths}
    for k in args.lengths:
        logger.info(f"  length k={k}: {len(next(iter(docs_by_len[k].values())))} parallel docs")

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, name="retrieval",
                                   config=vars(args))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"W&B init failed: {exc}")

    results: Dict = {
        "experiment": "retrieval",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "langs": data.langs, "pivot": args.pivot, "margin": args.margin,
        "lengths": args.lengths, "encoders": {},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for spec in args.encoders:
        logger.info(f"\n══ {spec} ══")
        emb = build_embedder(spec, device)
        per_len = {}
        for k in args.lengths:
            r = retrieval_for_length(emb, docs_by_len[k], args.pivot, args.margin,
                                     args.batch_size, args.cache_dir, k)
            per_len[str(k)] = r
            logger.info(f"  k={k}: mean xsim acc={r['mean_acc']:.4f} "
                        f"(err={r['mean_xsim_error']:.4f}, n={r['n_items']})")
            if wandb_run is not None:
                wandb_run.log({f"{emb.name}/k{k}/xsim_acc": r["mean_acc"],
                               f"{emb.name}/k{k}/xsim_err": r["mean_xsim_error"]})
        results["encoders"][emb.name] = per_len
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
    lengths = results["lengths"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, per_len in results["encoders"].items():
        ys = [per_len[str(k)]["mean_acc"] for k in lengths]
        ax.plot(lengths, ys, marker="o", label=name)
    ax.set_xlabel("pseudo-doc length (k consecutive sentences)")
    ax.set_ylabel("xsim retrieval accuracy (en↔xx)")
    ax.set_title("Cross-lingual retrieval vs text length")
    ax.set_xticks(lengths)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = plot_dir / "retrieval_length_scaling.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [plot] {out}")


if __name__ == "__main__":
    main()
