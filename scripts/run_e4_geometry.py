#!/usr/bin/env python3
"""
run_e4_geometry.py — E4: geometry of the multilingual representation space.

Three complementary probes of *what the space looks like* after COMET training,
on FLORES-200 (multi-parallel, so every metric is computed on the same content):

  1. Anisotropy.  Mean cosine between random *same-language* sentence pairs. A
     high value means the embeddings are squeezed into a narrow cone (the space
     has "collapsed") — common after contrastive / regression objectives and a
     confound for every cosine-based metric. Reported per language + overall.

  2. Cross-lingual alignment gap.  mean cos of *true translation* pairs minus
     mean cos of *random cross-lingual* pairs. Bigger gap = parallel sentences
     stand out more from the crowd. Complements anisotropy (controls for it).

  3. Language-identity probe.  A logistic-regression classifier predicting the
     language id from the sentence embedding (train/test split over sentences).
     Lower accuracy = more *language-agnostic* representations, i.e. translation
     content lives in a shared subspace rather than separate per-language
     regions. This is the quantity most directly tied to cross-lingual transfer.

  4. (COMET only) Learned layer-mixture weights.  COMET combines all transformer
     layers with a learned (sparse)softmax. We dump those weights to show *which*
     layers COMET leans on — and how bio-finetuning shifts them.

Usage
-----
  python scripts/run_e4_geometry.py \
      --encoders comet:Unbabel/wmt22-comet-da "comet:$BIO_CKPT" \
                 hf-mean:xlm-roberta-large labse e5 \
      --langs en de es fr ru zh \
      --output results/repana/e4_geometry.json
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from repana_common import (
    CORE_LANGS, build_embedder, cached_embed, load_flores, pick_device,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def anisotropy(vecs: np.ndarray, n_pairs: int, rng) -> float:
    """Mean cosine of random pairs (vecs L2-normalised → cosine = dot)."""
    n = len(vecs)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    return float((vecs[i[keep]] * vecs[j[keep]]).sum(1).mean())


def language_probe(vecs_by_lang: Dict[str, np.ndarray], seed: int) -> Dict:
    """Linear probe accuracy for predicting language id from the embedding."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    langs = sorted(vecs_by_lang.keys())
    lab = {l: i for i, l in enumerate(langs)}
    X = np.concatenate([vecs_by_lang[l] for l in langs], 0)
    y = np.concatenate([np.full(len(vecs_by_lang[l]), lab[l]) for l in langs])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, multi_class="multinomial")
    clf.fit(Xtr, ytr)
    acc = float(clf.score(Xte, yte))
    return {"lang_probe_acc": acc, "chance": 1.0 / len(langs), "n_langs": len(langs)}


def alignment_gap(vecs_by_lang: Dict[str, np.ndarray], pivot: str, rng) -> Dict:
    """mean cos(true translation pair) − mean cos(random cross-lingual pair)."""
    pv = vecs_by_lang[pivot]
    n = len(pv)
    gaps, trues, rands = {}, [], []
    for l, vt in vecs_by_lang.items():
        if l == pivot:
            continue
        true_cos = float((pv * vt).sum(1).mean())
        perm = rng.permutation(n)
        rand_cos = float((pv * vt[perm]).sum(1).mean())
        gaps[f"{pivot}-{l}"] = {"true_cos": true_cos, "rand_cos": rand_cos,
                                "gap": true_cos - rand_cos}
        trues.append(true_cos); rands.append(rand_cos)
    return {"per_pair": gaps, "mean_true_cos": float(np.mean(trues)),
            "mean_rand_cos": float(np.mean(rands)),
            "mean_gap": float(np.mean(trues) - np.mean(rands))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoders", nargs="+", required=True)
    p.add_argument("--langs", nargs="+", default=CORE_LANGS)
    p.add_argument("--pivot", default="en")
    p.add_argument("--split", default="devtest")
    p.add_argument("--max_sents", type=int, default=None)
    p.add_argument("--n_pairs", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--cache_dir", default="results/repana/emb_cache")
    p.add_argument("--output", default="results/repana/e4_geometry.json")
    p.add_argument("--wandb_project", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    logger.info(f"Device: {device}")
    data = load_flores(args.langs, args.split, args.max_sents)
    rng = np.random.default_rng(args.seed)

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, name="e4_geometry", config=vars(args))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"W&B init failed: {exc}")

    results: Dict = {
        "experiment": "E4_geometry",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "langs": data.langs, "pivot": args.pivot, "encoders": {},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for spec in args.encoders:
        logger.info(f"\n══ {spec} ══")
        emb = build_embedder(spec, device)
        vecs = {l: cached_embed(emb, data.sentences[l], f"flores_k1_{l}",
                                args.cache_dir, args.batch_size) for l in data.langs}

        aniso = {l: anisotropy(vecs[l], args.n_pairs, rng) for l in data.langs}
        aniso["mean"] = float(np.mean([aniso[l] for l in data.langs]))
        probe = language_probe(vecs, args.seed)
        gap = alignment_gap(vecs, args.pivot, rng)
        lw = emb.layer_weights()

        entry = {
            "anisotropy": aniso,
            "language_probe": probe,
            "alignment_gap": gap,
            "layer_weights": (lw.tolist() if lw is not None else None),
        }
        results["encoders"][emb.name] = entry
        logger.info(f"  anisotropy(mean)={aniso['mean']:.4f}  "
                    f"lang_probe_acc={probe['lang_probe_acc']:.4f} (chance {probe['chance']:.3f})  "
                    f"align_gap={gap['mean_gap']:.4f}")
        if lw is not None:
            top = np.argsort(lw)[::-1][:3]
            logger.info(f"  COMET layer weights: top layers {top.tolist()} "
                        f"(weights {np.round(lw[top], 3).tolist()})")
        if wandb_run is not None:
            wandb_run.log({f"{emb.name}/anisotropy": aniso["mean"],
                           f"{emb.name}/lang_probe_acc": probe["lang_probe_acc"],
                           f"{emb.name}/align_gap": gap["mean_gap"]})
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
    names = list(results["encoders"].keys())

    # summary bars: anisotropy, lang-probe, alignment gap
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [("anisotropy", lambda r: r["anisotropy"]["mean"], "anisotropy (↓ better=more isotropic)"),
               ("lang_probe", lambda r: r["language_probe"]["lang_probe_acc"], "language-id probe acc (↓ = more agnostic)"),
               ("align_gap", lambda r: r["alignment_gap"]["mean_gap"], "alignment gap (↑ better)")]
    for ax, (_, fn, title) in zip(axes, metrics):
        ys = [fn(results["encoders"][n]) for n in names]
        ax.bar(names, ys, color="#4C72B0", alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        for i, v in enumerate(ys):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    out = plot_dir / "e4_geometry_summary.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [plot] {out}")

    # COMET layer-mixture weights
    comet = {n: r["layer_weights"] for n, r in results["encoders"].items()
             if r["layer_weights"] is not None}
    if comet:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for n, w in comet.items():
            ax.plot(range(len(w)), w, marker="o", label=n)
        ax.set_xlabel("transformer layer (0 = embeddings)")
        ax.set_ylabel("learned mixture weight")
        ax.set_title("E4 — COMET learned layer-mixture weights")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout()
        out2 = plot_dir / "e4_layer_weights.png"
        plt.savefig(out2, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  [plot] {out2}")


if __name__ == "__main__":
    main()
