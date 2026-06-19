#!/usr/bin/env python3
"""
run_e5_cka.py — E5: where in the network did training move the representations?

Linear CKA measures how similar two representation spaces are (1 = identical up
to rotation/scale, 0 = unrelated), robustly to the anisotropy that makes raw
cosine misleading. We use it for two views:

  (A) Per-layer drift (the three XLM-R-large variants share architecture AND
      tokenizer, so we feed identical inputs and compare layer ℓ to layer ℓ):

        CKA(XLM-R, COMET)       — what COMET's MT-quality training changed, and
                                  at which depth (embeddings → low → top layers).
        CKA(COMET, COMET-bio)   — what biomedical fine-tuning changed on top, and
                                  whether it touches the multilingual lower layers
                                  (a catastrophic-forgetting signature) or stays
                                  near the task head.

  (B) Output-space similarity across *all* encoders (incl. LaBSE/E5, which have
      different depth/width — CKA still applies because it only needs the same N
      examples). A heatmap of sentence-embedding CKA shows whether COMET's output
      space looks more like a raw XLM-R or more like a contrastive aligner.

Computed on a multilingual sample (pooled across a few FLORES languages) so the
similarity reflects the *cross-lingual* representation, not one language.

Usage
-----
  python scripts/run_e5_cka.py \
      --encoders hf-mean:xlm-roberta-large comet:Unbabel/wmt22-comet-da \
                 "comet:$BIO_CKPT" labse e5 \
      --reference hf-mean:xlm-roberta-large \
      --langs en de zh --n_per_lang 400 \
      --output results/repana/e5_cka.json
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from repana_common import (
    build_embedder, linear_cka, load_flores, pick_device,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoders", nargs="+", required=True)
    p.add_argument("--reference", default=None,
                   help="Reference encoder spec for per-layer drift curves "
                        "(default: first --encoders entry).")
    p.add_argument("--langs", nargs="+", default=["en", "de", "zh"])
    p.add_argument("--n_per_lang", type=int, default=400)
    p.add_argument("--split", default="devtest")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default="results/repana/e5_cka.json")
    p.add_argument("--wandb_project", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    logger.info(f"Device: {device}")
    data = load_flores(args.langs, args.split)
    # pooled multilingual sample (same rows across languages)
    n = min(args.n_per_lang, data.n)
    sample = [s for l in data.langs for s in data.sentences[l][:n]]
    logger.info(f"  CKA sample: {len(sample)} sentences ({len(data.langs)} langs × {n})")

    reference = args.reference or args.encoders[0]

    # ── collect per-layer reps and sentence embeddings for each encoder ──
    layer_reps: Dict[str, np.ndarray] = {}     # name -> (L, N, D)  (xlmr-large only)
    sent_embs: Dict[str, np.ndarray] = {}      # name -> (N, D)     (all)
    names: List[str] = []
    ref_name = None
    for spec in args.encoders:
        logger.info(f"\n══ {spec} ══")
        emb = build_embedder(spec, device)
        names.append(emb.name)
        if spec == reference:
            ref_name = emb.name
        sent_embs[emb.name] = emb.embed(sample, args.batch_size)
        if emb.is_xlmr_large:
            layer_reps[emb.name] = emb.layer_reps(sample, args.batch_size)
            logger.info(f"  captured {layer_reps[emb.name].shape[0]} layers")
        else:
            logger.info("  (different architecture → output-space CKA only)")
        del emb
        import torch, gc
        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if ref_name is None:
        ref_name = names[0]

    results: Dict = {
        "experiment": "E5_cka",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "langs": data.langs, "n_per_lang": n, "reference": ref_name,
        "per_layer_cka_vs_reference": {}, "output_cka_matrix": {}, "encoder_order": names,
    }

    # ── (A) per-layer drift vs reference (xlmr-large group) ──
    if ref_name in layer_reps:
        ref_layers = layer_reps[ref_name]
        L = ref_layers.shape[0]
        for name, reps in layer_reps.items():
            if name == ref_name:
                continue
            curve = [linear_cka(ref_layers[l], reps[l]) for l in range(L)]
            results["per_layer_cka_vs_reference"][name] = curve
            logger.info(f"  CKA({ref_name} , {name}) per layer: "
                        f"min={min(curve):.3f} @L{int(np.argmin(curve))}  final={curve[-1]:.3f}")

    # ── (B) output-space CKA matrix across all encoders ──
    M = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            M[i, j] = linear_cka(sent_embs[a], sent_embs[b])
    results["output_cka_matrix"] = {"names": names, "matrix": M.tolist()}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    _plot(results, out_path.parent / "plots")

    if args.wandb_project:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, name="e5_cka", config=vars(args))
            for name, curve in results["per_layer_cka_vs_reference"].items():
                run.log({f"cka_final/{name}": curve[-1], f"cka_min/{name}": float(min(curve))})
            run.finish()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"W&B failed: {exc}")
    logger.info(f"\nResults → {out_path}")


def _plot(results: Dict, plot_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir.mkdir(parents=True, exist_ok=True)

    curves = results["per_layer_cka_vs_reference"]
    if curves:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for name, curve in curves.items():
            ax.plot(range(len(curve)), curve, marker="o", label=f"CKA({results['reference']}, {name})")
        ax.set_xlabel("transformer layer (0 = embeddings)")
        ax.set_ylabel("linear CKA")
        ax.set_title("E5 — per-layer representational drift")
        ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout()
        out = plot_dir / "e5_per_layer_cka.png"
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  [plot] {out}")

    names = results["output_cka_matrix"]["names"]
    M = np.array(results["output_cka_matrix"]["matrix"])
    fig, ax = plt.subplots(figsize=(1.3 * len(names) + 2, 1.3 * len(names) + 1))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.6 else "black", fontsize=7)
    ax.set_title("E5 — output-space CKA across encoders")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out2 = plot_dir / "e5_output_cka_matrix.png"
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [plot] {out2}")


if __name__ == "__main__":
    main()
