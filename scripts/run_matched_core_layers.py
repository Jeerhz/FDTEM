#!/usr/bin/env python3
"""
run_matched_core_layers.py — Q3: where in the layer stack is the core lost?

Same matched-core inputs as run_matched_core.py, but embeddings are taken at
EVERY hidden layer of an HF encoder (masked mean pooling per layer). For each
(layer, L, φ) we report:

  * detection — P[cos(q_l, x_clean_l) > cos(q_l, x_pert_l)]  (signal survival)
  * core retention — cos(x_clean_l, core_l)                  (geometric drift)

Reading: if detection is high in middle layers but collapses in the last
layers as L grows, the information is present but destroyed by the final
representation (pooling-adjacent); if it dies early, the encoder never
separates the core from the filler. The inert vs natural contrast separates
mechanical length from semantic drift, layer by layer.

Reduced grid by default (this is a probe, not the headline experiment):
langs de fr · φ inert+natural · π = 0.5 · 80 cores per language.

Usage
-----
  python scripts/run_matched_core_layers.py \
      --model xlm-roberta-large --output results/matched_core/layer_probe.json
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from block_xsim_common import CATEGORIES, Perturber
from matched_core_common import (
    FillerBank, TokenCounter, build_input, pick_items,
)
from representation_analysis_common import load_flores_source, pick_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


@torch.no_grad()
def embed_layers(texts: List[str], model, tok, device: str,
                 batch_size: int) -> np.ndarray:
    """→ (n_layers, n_texts, dim) float16, L2-normalised per layer."""
    chunks: List[np.ndarray] = []
    for s in range(0, len(texts), batch_size):
        enc = tok(texts[s:s + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(device)
        out = model(**enc, output_hidden_states=True)
        m = enc["attention_mask"].unsqueeze(-1).to(out.hidden_states[0].dtype)
        layer_embs = []
        for h in out.hidden_states:                      # embeddings + 24 layers
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            layer_embs.append(pooled.cpu().to(torch.float16).numpy())
        chunks.append(np.stack(layer_embs, 0))
        if (s // batch_size) % 20 == 0:
            logger.info(f"    {s + len(layer_embs[0])}/{len(texts)} texts")
    return np.concatenate(chunks, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="xlm-roberta-large")
    ap.add_argument("--langs", nargs="+", default=["de", "fr"])
    ap.add_argument("--pivot", default="en")
    ap.add_argument("--L_grid", nargs="+", type=int, default=[60, 120, 240, 480])
    ap.add_argument("--phis", nargs="+", default=["inert", "natural"])
    ap.add_argument("--pi", type=float, default=0.5)
    ap.add_argument("--n_items", type=int, default=80)
    ap.add_argument("--core_min_tok", type=int, default=20)
    ap.add_argument("--core_max_tok", type=int, default=48)
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    ap.add_argument("--flores_source", choices=["plus", "raw"], default="plus")
    ap.add_argument("--splits", nargs="+", default=["dev", "devtest"])
    ap.add_argument("--max_sents", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/matched_core/layer_probe.json")
    args = ap.parse_args()

    device = pick_device(args.device)
    langs = [args.pivot] + [l for l in args.langs if l != args.pivot]
    loaded = [(sp, load_flores_source(args.flores_source, langs, sp, args.max_sents))
              for sp in args.splits]
    tc = TokenCounter()

    from transformers import AutoModel, AutoTokenizer
    logger.info(f"loading {args.model} on {device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    # stats[(phi, L)][layer] = [0/1 ...] ; ret[(phi, L)][layer] = [cos ...]
    det: Dict = {}
    ret: Dict = {}
    n_layers = None
    for lang in args.langs:
        if lang == args.pivot:
            continue
        rows, sents, urls = [], [], []
        for _sp, d in loaded:
            base = len(sents)
            for i in range(d.n):
                rows.append((base + i, d.urls[i], d.sentences[args.pivot][i],
                             d.sentences[lang][i]))
                sents.append(d.sentences[lang][i])
                urls.append(d.urls[i])
        pert = Perturber.for_corpus(sents, lang, args.seed)
        items = pick_items(rows, pert, tc, args.n_items, args.core_min_tok,
                           args.core_max_tok, args.seed, args.categories)
        bank = FillerBank(sents, urls, tc, lang, args.seed)

        texts: List[str] = []
        key2i: Dict[tuple, int] = {}

        def add(key, text):
            if key not in key2i:
                key2i[key] = len(texts)
                texts.append(text)

        for ii, it in enumerate(items):
            add(("q", ii), it.src)
            add(("core", ii), it.core)
            for L in args.L_grid:
                for phi in args.phis:
                    xc = build_input(bank, it, it.core, L, phi, args.pi)
                    if xc is None:
                        continue
                    add(("x", ii, L, phi, "clean"), xc)
                    for cat, neg in it.negs.items():
                        add(("x", ii, L, phi, cat),
                            build_input(bank, it, neg, L, phi, args.pi))
        logger.info(f"  [{lang}] {len(items)} cores, {len(texts)} texts")
        E = embed_layers(texts, model, tok, device, args.batch_size)
        n_layers = E.shape[0]

        for L in args.L_grid:
            for phi in args.phis:
                for ii, it in enumerate(items):
                    kc = ("x", ii, L, phi, "clean")
                    if kc not in key2i:
                        continue
                    q = E[:, key2i[("q", ii)], :].astype(np.float32)
                    xc = E[:, key2i[kc], :].astype(np.float32)
                    core = E[:, key2i[("core", ii)], :].astype(np.float32)
                    sim_clean = (q * xc).sum(-1)                  # (n_layers,)
                    ret.setdefault((phi, L), [[] for _ in range(n_layers)])
                    det.setdefault((phi, L), [[] for _ in range(n_layers)])
                    rc = (xc * core).sum(-1)
                    for l in range(n_layers):
                        ret[(phi, L)][l].append(float(rc[l]))
                    for cat in it.negs:
                        xn = E[:, key2i[("x", ii, L, phi, cat)], :].astype(np.float32)
                        sim_neg = (q * xn).sum(-1)
                        wins = sim_clean > sim_neg
                        for l in range(n_layers):
                            det[(phi, L)][l].append(float(wins[l]))
        del E

    results = {"experiment": "matched_core_layer_probe",
               "timestamp": datetime.now(timezone.utc).isoformat(),
               "config": vars(args), "n_layers": n_layers,
               "detection": {}, "core_retention": {}}
    for (phi, L), per_layer in det.items():
        results["detection"].setdefault(phi, {})[str(L)] = [
            float(np.mean(v)) if v else None for v in per_layer]
    for (phi, L), per_layer in ret.items():
        results["core_retention"].setdefault(phi, {})[str(L)] = [
            float(np.mean(v)) if v else None for v in per_layer]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    # ── plots ─────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = out_path.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric, data, ylab, chance in (
            ("layer_detection", results["detection"],
             "detection  P[clean closer]", 0.5),
            ("layer_retention", results["core_retention"],
             "cos( x(L), bare core )", None)):
        fig, axes = plt.subplots(1, len(args.phis),
                                 figsize=(5.2 * len(args.phis), 4.2),
                                 squeeze=False, sharey=True)
        for ax, phi in zip(axes[0], args.phis):
            for L in args.L_grid:
                y = data[phi][str(L)]
                ax.plot(range(len(y)), y, marker=".", label=f"L={L}")
            if chance is not None:
                ax.axhline(chance, color="grey", ls=":", lw=1)
            ax.set_xlabel("layer (0 = embeddings)")
            ax.set_title(f"φ = {phi}", fontsize=10)
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel(ylab)
        axes[0][-1].legend(fontsize=8)
        p = plot_dir / f"{metric}.png"
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
        logger.info(f"  [plot] {p}")
    logger.info(f"Results → {out_path}")


if __name__ == "__main__":
    main()
