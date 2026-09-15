#!/usr/bin/env python3
"""
run_xsim.py — xSIM++ on concatenated blocks: can COMET's encoder still
spot a single perturbed sentence once it is buried in a k-sentence block?

Setup (see blocks.py for the details)
-----------------------------------------------
  * FLORES+ articles → non-overlapping blocks of k consecutive sentences.
  * Query   = the source block (k source sentences, concatenated in order).
  * Pool    = every true target block  +  hard negatives obtained by applying
              ONE xSIM++ perturbation (causality / entity / number) to ONE
              sentence of a block, leaving the other k−1 untouched.
  * Correct = retrieving the concatenation of all k translations, in order,
              with no perturbation.

Every run scores four nested pools off the same similarity matrix (`pool_ablation`
in the JSON, plots/pool_ablation.png), so the contribution of the classic xsim
distractors — the *other* blocks' true targets — is read off directly:

    true_only           all true blocks                      classic xsim
    true+perturbed      all true blocks + all negatives      classic xsim++
    gold+all_perturbed  own gold + all negatives             distractors dropped
    gold+own_perturbed  own gold + own negatives only        pure dilution

Sweeping k ∈ {2,3,4,5} turns "how sensitive is the encoder to one error?" into
"how fast does that sensitivity dilute with block length?" — reported as xsim vs
xsim++ error, an error typology (misaligned vs which perturbation category won),
the push-apart detection rate, and the gold-minus-best-negative margin.

Usage
-----
  python experiments/length_isolation/run_xsim.py \
      --encoders comet:Unbabel/wmt22-comet-da \
                 "comet:$HOME/scratch/checkpoints/bio_mqm/.../last.ckpt" \
                 hf-mean:xlm-roberta-large labse e5 \
      --langs de es fr ru zh --k_list 2 3 4 5 \
      --splits dev devtest \
      --output results/block_xsim/block_xsim.json \
      --wandb_project comet-block-xsim

  # inspect the generated hard negatives without touching a GPU
  python experiments/length_isolation/run_xsim.py --dry_run --langs de --k_list 3
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from blocks import (
    CATEGORIES, block_text, build_blocks, build_pool, evaluate_blocks,
)
from nlp_perturb import build_perturber
from fdtem.encoders import build_embedder, cached_embed, enc_tag as _enc_tag, pick_device
from fdtem.flores import load_flores_source

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _load_splits(source: str, langs: List[str], splits: List[str], max_sents):
    out = []
    for sp in splits:
        data = load_flores_source(source, langs, sp, max_sents)
        out.append((sp, data))
    return out


def _dry_run(args, loaded) -> None:
    """Print block/variant coverage and a few example hard negatives."""
    for lang in args.langs:
        if lang == args.pivot:
            continue
        pool_lang = lang if args.direction == "en2xx" else args.pivot
        corpus = [s for _sp, d in loaded for s in d.sentences[pool_lang]]
        pert = build_perturber(corpus, pool_lang, args.seed, args.perturb_backend,
                               args.wordnet_langs)
        logger.info(f"\n══ {pool_lang}  (entity bank: {len(pert.bank)} surfaces) ══")
        for k in args.k_list:
            blocks = [(sp, d, build_blocks(d, k, split=sp)) for sp, d in loaded]
            n = sum(len(b) for _, _, b in blocks)
            _cands, _true, stats = build_pool(blocks, pool_lang, args.categories,
                                              args.variants_per_position, pert)
            per_blk = {c: round(stats[c] / max(1, n), 2) for c in args.categories}
            logger.info(f"  k={k}: {n} blocks, pool={n + sum(stats[c] for c in args.categories)}"
                        f"  variants/block={per_blk}")
        # examples
        sample = next(s for s in corpus if len(s) > 40)
        logger.info(f"  example source: {sample}")
        for c in args.categories:
            for v in pert.variants(sample, c, 2):
                logger.info(f"    [{c}] {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+",
                    default=["comet:Unbabel/wmt22-comet-da"])
    ap.add_argument("--langs", nargs="+", default=["de", "es", "fr", "ru"],
                    help="Non-pivot languages (the pivot is added automatically). "
                         "The heuristic backend needs letter case for the entity "
                         "perturbation, so zh/ja/th are excluded by default; "
                         "--perturb_backend spacy lifts that (real NER).")
    ap.add_argument("--pivot", default="en")
    ap.add_argument("--direction", choices=["en2xx", "xx2en"], default="en2xx",
                    help="en2xx: query=source block, pool=translation blocks "
                         "(perturbations applied to the translations). "
                         "xx2en: the original xSIM++ direction (pool=English).")
    ap.add_argument("--k_list", nargs="+", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES),
                    choices=list(CATEGORIES))
    ap.add_argument("--variants_per_position", type=int, default=2,
                    help="Hard negatives per (block, sentence position, category).")
    ap.add_argument("--perturb_backend", choices=["heuristic", "spacy", "auto"],
                    default="spacy",
                    help="spacy (default): real NER + morphology + parse-anchored "
                         "negation; needs `python -m spacy download "
                         "<lang>_core_*_sm`, and is the only backend that produces "
                         "entity negatives for uncased scripts (zh/ja). heuristic: "
                         "the self-contained regex/casing perturber. auto: spacy, "
                         "falling back to heuristic. Note the two backends generate "
                         "different negatives — do not compare runs across them.")
    ap.add_argument("--wordnet_langs", nargs="*", default=["en"],
                    help="Languages whose antonyms may come from WordNet, on top "
                         "of the curated lexicon (spacy backend only). Clean for "
                         "en; Open Multilingual WordNet has no de/ru and is "
                         "sense-ambiguous elsewhere — see nlp_perturb.py.")
    ap.add_argument("--flores_source", choices=["plus", "raw"], default="plus")
    ap.add_argument("--splits", nargs="+", default=["dev", "devtest"],
                    help="FLORES+ splits to pool (more splits = more blocks; "
                         "k=5 blocks are scarce in a single split).")
    ap.add_argument("--max_sents", type=int, default=None)
    ap.add_argument("--max_blocks", type=int, default=None,
                    help="Truncate the block list per k (smoke tests).")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", default="results/block_xsim/emb_cache")
    ap.add_argument("--output", default="results/block_xsim/block_xsim.json")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--dry_run", action="store_true",
                    help="Build blocks + perturbations, print stats and examples, exit.")
    args = ap.parse_args()

    langs = [args.pivot] + [l for l in args.langs if l != args.pivot]
    loaded = _load_splits(args.flores_source, langs, args.splits, args.max_sents)

    if args.dry_run:
        _dry_run(args, loaded)
        return

    device = pick_device(args.device)
    logger.info(f"Device: {device}")

    run_name = args.run_name or f"block_xsim_{args.direction}"
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project, name=run_name, job_type="block_xsim",
                tags=[f"dir:{args.direction}", f"flores:{args.flores_source}"]
                     + [_enc_tag(s) for s in args.encoders],
                config=vars(args))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"W&B init failed: {exc}")

    results: Dict = {"experiment": "block_xsim++",
                     "timestamp": datetime.now(timezone.utc).isoformat(),
                     "config": vars(args), "encoders": {}}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── pre-build every (lang, k) pool once; encoders then just embed ──────────
    pools: Dict[str, Dict[int, dict]] = {}
    for lang in langs:
        if lang == args.pivot:
            continue
        query_lang = args.pivot if args.direction == "en2xx" else lang
        pool_lang = lang if args.direction == "en2xx" else args.pivot
        corpus = [s for _sp, d in loaded for s in d.sentences[pool_lang]]
        pert = build_perturber(corpus, pool_lang, args.seed, args.perturb_backend,
                               args.wordnet_langs)
        if (args.perturb_backend == "heuristic" and pool_lang in ("zh", "ja", "th")
                and "entity" in args.categories):  # spacy has NER for these
            logger.warning(f"  [{pool_lang}] entity perturbation needs letter case — "
                           "no entity negatives will be generated for this language.")
        pools[lang] = {}
        for k in args.k_list:
            per_split = [(sp, d, build_blocks(d, k, split=sp)) for sp, d in loaded]
            if args.max_blocks:
                per_split = [(sp, d, b[:args.max_blocks]) for sp, d, b in per_split]
            n_blocks = sum(len(b) for _, _, b in per_split)
            if n_blocks < 2:
                logger.warning(f"  [{lang}] k={k}: {n_blocks} blocks — skipping")
                continue
            queries = [block_text(d, blk, query_lang)
                       for _sp, d, bs in per_split for blk in bs]
            cands, true_idx, stats = build_pool(per_split, pool_lang, args.categories,
                                                args.variants_per_position, pert)
            pools[lang][k] = {"queries": queries, "cands": cands,
                              "true_idx": true_idx, "stats": stats,
                              "query_lang": query_lang, "pool_lang": pool_lang}
            logger.info(f"  [{lang}] k={k}: {n_blocks} blocks, pool={len(cands)}, "
                        f"variants={{{', '.join(f'{c}:{stats[c]}' for c in args.categories)}}}")

    for spec in args.encoders:
        logger.info(f"\n══ {spec} ══")
        emb = build_embedder(spec, device)
        per_lang: Dict[str, Dict[str, dict]] = {}
        for lang, by_k in pools.items():
            per_lang[lang] = {}
            for k, P in by_k.items():
                tag = f"blk_{args.direction}_{lang}_k{k}_v{args.variants_per_position}"
                q = cached_embed(emb, P["queries"], f"{tag}_q", args.cache_dir,
                                 args.batch_size)
                c = cached_embed(emb, [x.text for x in P["cands"]], f"{tag}_c",
                                 args.cache_dir, args.batch_size)
                m = evaluate_blocks(q, c, P["cands"], P["true_idx"], args.categories)
                m["variant_stats"] = P["stats"]
                per_lang[lang][str(k)] = m
                pa = m["pool_ablation"]
                line = (f"  {lang} k={k}: own={_fmt(pa['gold+own_perturbed'])}"
                        f"/chance={_fmt(m['own_pool_chance_err'])} "
                        f"hard={_fmt(pa['gold+all_perturbed'])}"
                        f"  [classic: xsim={m['xsim_err']:.4f} "
                        f"xsim++={m['xsimpp_err']:.4f}]")
                if m["detection_rate"] is not None:
                    line += (f" detect={m['detection_rate']:.4f}"
                             f" margin_own={m['margin_vs_best_own_perturbation']:.4f}")
                logger.info(line)
                if wandb_run is not None:
                    pref = f"{emb.name}/{lang}/k{k}"
                    log = {f"{pref}/xsim_err": m["xsim_err"],
                           f"{pref}/xsimpp_err": m["xsimpp_err"],
                           f"{pref}/margin": m["margin_vs_best_negative"]}
                    for pool_key, wb_key in (("gold+all_perturbed", "err_hard_pool"),
                                             ("gold+own_perturbed", "err_own_pool")):
                        if pa[pool_key] is not None:
                            log[f"{pref}/{wb_key}"] = pa[pool_key]
                    if m["own_pool_chance_err"] is not None:
                        log[f"{pref}/err_own_pool_chance"] = m["own_pool_chance_err"]
                    for cat, v in m["own_pool_err_by_category"].items():
                        if v is not None:
                            log[f"{pref}/err_own_{cat}"] = v
                    if m["detection_rate"] is not None:
                        log[f"{pref}/detection_rate"] = m["detection_rate"]
                    if m["margin_vs_best_own_perturbation"] is not None:
                        log[f"{pref}/margin_own"] = m["margin_vs_best_own_perturbation"]
                    for cat, v in m["per_category_err"].items():
                        log[f"{pref}/err_{cat}"] = v
                    wandb_run.log(log)

        # mean over languages, per k
        mean_by_k: Dict[str, dict] = {}
        for k in args.k_list:
            cells = [per_lang[l][str(k)] for l in per_lang if str(k) in per_lang[l]]
            if not cells:
                continue
            mean_by_k[str(k)] = {
                "xsim_err": float(np.mean([c["xsim_err"] for c in cells])),
                "xsimpp_err": float(np.mean([c["xsimpp_err"] for c in cells])),
                "pool_ablation": {pool: _nanmean([c["pool_ablation"][pool] for c in cells])
                                  for pool in cells[0]["pool_ablation"]},
                "own_pool_err_by_category": {
                    cat: _nanmean([c["own_pool_err_by_category"][cat] for c in cells])
                    for cat in args.categories},
                "own_pool_n_blocks": int(sum(c["own_pool_n_blocks"] for c in cells)),
                "own_pool_chance_err": _nanmean([c["own_pool_chance_err"] for c in cells]),
                "own_pool_negatives_per_block": float(np.mean(
                    [c["own_pool_negatives_per_block"] for c in cells])),
                "margin_vs_best_negative": float(np.mean([c["margin_vs_best_negative"] for c in cells])),
                "detection_rate": _nanmean([c["detection_rate"] for c in cells]),
                "margin_vs_best_own_perturbation": _nanmean(
                    [c["margin_vs_best_own_perturbation"] for c in cells]),
                "per_category_err": {cat: float(np.mean([c["per_category_err"][cat] for c in cells]))
                                     for cat in args.categories},
                "error_breakdown": {key: float(np.mean([c["error_breakdown"][key] for c in cells]))
                                    for key in cells[0]["error_breakdown"]},
                "detection_by_position": _mean_dicts([c["detection_by_position"] for c in cells]),
            }
        per_lang["_mean_by_k"] = mean_by_k
        results["encoders"][emb.name] = per_lang

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        del emb
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plots = _plots(results, out_path.parent / "plots", args.categories)
    if wandb_run is not None:
        import wandb
        for p in plots:
            wandb_run.log({f"plots/{p.stem}": wandb.Image(str(p))})
        wandb_run.finish()
    logger.info(f"\nResults → {out_path}")


def _fmt(v) -> str:
    return "  n/a " if v is None else f"{v:.4f}"


def _nanmean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def _mean_dicts(dicts):
    keys = sorted({k for d in dicts for k in d}, key=int)
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts if d.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def _plots(results: Dict, plot_dir: Path, categories) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    encoders = list(results["encoders"])

    def ks(enc):
        return sorted(results["encoders"][enc]["_mean_by_k"], key=int)

    # 1. headline — the hard-negative pools: no other block's true target is a
    #    candidate, so only the injected edit can separate gold from distractor
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, enc in enumerate(encoders):
        mk = results["encoders"][enc]["_mean_by_k"]
        x = [int(k) for k in ks(enc)]
        color = f"C{i}"
        y_all = [mk[k]["pool_ablation"]["gold+all_perturbed"] for k in ks(enc)]
        y_own = [mk[k]["pool_ablation"]["gold+own_perturbed"] for k in ks(enc)]
        if any(v is not None for v in y_all):
            ax.plot(x, y_all, marker="^", ls="--", color=color, alpha=0.5,
                    label=f"{enc} — gold + all negatives")
        if any(v is not None for v in y_own):
            ax.plot(x, y_own, marker="d", ls="-", color=color,
                    label=f"{enc} — gold + own negatives")
    # chance: the gold ranked at random among its own m negatives, m grows with k
    mk0 = results["encoders"][encoders[0]]["_mean_by_k"]
    kk = ks(encoders[0])
    chance = [mk0[k].get("own_pool_chance_err") for k in kk]
    if any(v is not None for v in chance):
        ax.plot([int(k) for k in kk], chance, color="grey", ls=":", lw=1.2,
                label="chance (gold ranked at random in its own pool)")
    ax.set_xticks(sorted({int(k) for enc in encoders for k in ks(enc)}))
    ax.set_xlabel("block length k (sentences)")
    ax.set_ylabel("retrieval error rate")
    ax.set_title("Hard-negative retrieval error vs block length "
                 "(mean over languages)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    p = plot_dir / "hard_negative_error.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # 2. dilution — can the encoder still push the perturbed block away?
    fig, ax = plt.subplots(figsize=(8, 5))
    for enc in encoders:
        mk = results["encoders"][enc]["_mean_by_k"]
        x = [int(k) for k in ks(enc)]
        y = [mk[k]["detection_rate"] for k in ks(enc)]
        if any(v is not None for v in y):
            ax.plot(x, y, marker="o", label=enc)
    ax.axhline(0.5, color="grey", ls=":", lw=1, label="chance")
    ax.set_xlabel("block length k (sentences)")
    ax.set_ylabel("P[cos(query, gold) > cos(query, perturbed)]")
    ax.set_title("Single-sentence error detection vs block length")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    p = plot_dir / "detection_vs_length.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # 3. per-category error, own pool (gold + this block's own negatives of one
    #    category) — the classic distractors are out of this one too
    fig, axes = plt.subplots(1, len(encoders), figsize=(4.2 * len(encoders), 4),
                             squeeze=False, sharey=True)
    for ax, enc in zip(axes[0], encoders):
        mk = results["encoders"][enc]["_mean_by_k"]
        x = np.arange(len(ks(enc)))
        w = 0.8 / max(1, len(categories))
        for j, cat in enumerate(categories):
            ax.bar(x + j * w,
                   [mk[k]["own_pool_err_by_category"].get(cat) or 0.0
                    for k in ks(enc)], width=w, label=cat)
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(ks(enc))
        ax.set_xlabel("k"); ax.set_title(enc, fontsize=9)
    axes[0][0].set_ylabel("error (pool = gold + own negatives, one category)")
    axes[0][-1].legend(fontsize=8)
    p = plot_dir / "error_by_category.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # 4. is the encoder blind to errors late in the block?
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, enc in enumerate(encoders):
        mk = results["encoders"][enc]["_mean_by_k"]
        for k in ks(enc):
            d = mk[k]["detection_by_position"]
            pos = sorted((p for p, v in d.items() if v is not None), key=int)
            if pos:
                ax.plot([int(p) for p in pos], [d[p] for p in pos], marker="o",
                        color=f"C{i}", alpha=0.4 + 0.5 * int(k) / max(int(x) for x in ks(enc)),
                        label=f"{enc} k={k}")
    ax.set_xlabel("position of the perturbed sentence in the block")
    ax.set_ylabel("detection rate")
    ax.set_title("Position of the injected error vs detectability")
    ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=2)
    p = plot_dir / "detection_by_position.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); paths.append(p)

    # 5. pool ablation — what do the classic xsim distractors actually buy?
    pools = [("true_only", ":", "o", "true blocks only (xsim)"),
             ("true+perturbed", "-", "s", "true blocks + negatives (xsim++)"),
             ("gold+all_perturbed", "--", "^", "gold + all negatives"),
             ("gold+own_perturbed", "-.", "d", "gold + own negatives")]
    fig, axes = plt.subplots(1, len(encoders), figsize=(4.2 * len(encoders), 4),
                             squeeze=False, sharey=True)
    for ax, enc in zip(axes[0], encoders):
        mk = results["encoders"][enc]["_mean_by_k"]
        x = [int(k) for k in ks(enc)]
        for pool, ls, mk_, lab in pools:
            y = [mk[k].get("pool_ablation", {}).get(pool) for k in ks(enc)]
            if any(v is not None for v in y):
                xs = [xi for xi, yi in zip(x, y) if yi is not None]
                ax.plot(xs, [v for v in y if v is not None], ls=ls, marker=mk_,
                        label=lab)
        ax.set_xticks(x)
        ax.set_xlabel("block length k"); ax.set_title(enc, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("retrieval error rate")
    axes[0][-1].legend(fontsize=7)
    p5 = plot_dir / "pool_ablation.png"
    fig.tight_layout(); fig.savefig(p5, dpi=130); plt.close(fig); paths.append(p5)

    for pth in paths:
        logger.info(f"  [plot] {pth}")
    return paths


if __name__ == "__main__":
    main()
