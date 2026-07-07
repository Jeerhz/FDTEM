#!/usr/bin/env python3
"""
features.py — frozen representation extraction + cache.

Both encoders are loaded once, strictly frozen, and run with *identical* code:
forward in eval() under torch.no_grad(), fp16 on GPU, output_hidden_states=True,
then pooled. The only thing that differs between them is the weights — same
backbone, same tokenizer, same pooling — so any accuracy-vs-length gap is
attributable to COMET's fine-tuning.

Encoders
--------
  raw   (kind: hf)    -> transformers.AutoModel.from_pretrained(ref)       (e.g. xlm-roberta-large)
  comet (kind: comet) -> the underlying transformer extracted from a COMET checkpoint:
                           m = load_from_checkpoint(download_model(ref))
                           hf_model = m.encoder.model      # XLMRobertaModel
                           tokenizer = m.encoder.tokenizer # XLMRobertaTokenizerFast

Pooling (config features.pooling)
  mean             masked mean over the last hidden layer (primary; excludes padding)
  cls              the [CLS]/<s> vector of the last layer
  all_layers_mean  mean over all hidden layers, then masked mean over tokens

Length buckets: a document is tokenised with max_length=L, truncation=True, so the
feature at bucket L sees exactly the first L tokens (incl. special tokens). Because
both encoders share the XLM-R tokenizer, the token ids at each L are identical.

Features are cached to .npy keyed by (encoder, pooling, normalize, L, hash(doc_ids)),
so they are extracted once and reused across probes/seeds.

backend: mock  -> a torch-free deterministic stand-in used ONLY by the local logic
smoke test. It fabricates 1024-d features from hashed whitespace tokens; it is NOT a
scientific encoder. A loud warning is printed.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from common import (
    active_encoders, content_hash, ensure_dirs, encoder_tag, load_config,
    pick_device, banner,
)

logger = logging.getLogger(__name__)
HIDDEN_DEFAULT = 1024


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


# ════════════════════════════════════════════════════════════════════════════
# Real (torch) encoders
# ════════════════════════════════════════════════════════════════════════════
class TorchEncoder:
    """Wraps a frozen HF transformer + tokenizer with identical pooling for both
    the raw baseline and the COMET-extracted transformer."""

    def __init__(self, tag: str, model, tokenizer, device: str, fp16: bool):
        import torch
        self.tag = tag
        self.tok = tokenizer
        self.device = device
        self.fp16 = fp16 and device == "cuda"
        model = model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        if self.fp16:
            model = model.half()
        self.model = model
        self.hidden_size = int(getattr(model.config, "hidden_size", HIDDEN_DEFAULT))

    @classmethod
    def from_hf(cls, role: str, ref: str, device: str, fp16: bool) -> "TorchEncoder":
        from transformers import AutoModel, AutoTokenizer
        logger.info("[hf] loading %s", ref)
        tok = AutoTokenizer.from_pretrained(ref)
        model = AutoModel.from_pretrained(ref)
        return cls(encoder_tag(role, ref), model, tok, device, fp16)

    @classmethod
    def from_comet(cls, role: str, ref: str, device: str, fp16: bool) -> "TorchEncoder":
        import os
        from comet import download_model, load_from_checkpoint
        ckpt = ref if os.path.isfile(ref) else download_model(ref)
        logger.info("[comet] loading %s → extracting encoder.model / encoder.tokenizer", ckpt)
        m = load_from_checkpoint(ckpt)
        hf_model = m.encoder.model          # XLMRobertaModel (the underlying transformer)
        tokenizer = m.encoder.tokenizer     # XLMRobertaTokenizerFast
        return cls(encoder_tag(role, ref), hf_model, tokenizer, device, fp16)

    def _pool(self, hidden_states, attention_mask, pooling: str):
        import torch
        am = attention_mask.unsqueeze(-1).to(torch.float32)
        if pooling == "cls":
            return hidden_states[-1][:, 0, :].float()
        if pooling == "all_layers_mean":
            stack = torch.stack(hidden_states, dim=0).float().mean(0)  # mean over layers -> [B,T,H]
            return (stack * am).sum(1) / am.sum(1).clamp(min=1e-9)
        # mean (primary): masked mean over the last hidden layer
        last = hidden_states[-1].float()
        return (last * am).sum(1) / am.sum(1).clamp(min=1e-9)

    def embed(self, texts: Sequence[str], L: int, pooling: str, normalize: bool,
              batch_size: int) -> np.ndarray:
        import torch
        out: List[np.ndarray] = []
        for s in range(0, len(texts), batch_size):
            batch = list(texts[s:s + batch_size])
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=L)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                res = self.model(input_ids=enc["input_ids"],
                                 attention_mask=enc["attention_mask"],
                                 output_hidden_states=True, return_dict=True)
                pooled = self._pool(res.hidden_states, enc["attention_mask"], pooling)
            out.append(pooled.cpu().float().numpy())
        arr = np.concatenate(out, 0)
        return _l2norm(arr) if normalize else arr


# ════════════════════════════════════════════════════════════════════════════
# Mock encoder (torch-free; LOGIC SMOKE TEST ONLY — not a scientific encoder)
# ════════════════════════════════════════════════════════════════════════════
class MockEncoder:
    def __init__(self, tag: str, role: str, hidden: int = HIDDEN_DEFAULT):
        self.tag = tag
        self.hidden_size = hidden
        # per-role fixed random projection so the two "encoders" differ (not engineered
        # to favour either: both masked-mean hashed tokens, only the projection differs)
        seed = 1 if role == "comet" else 2
        self._proj = np.random.RandomState(seed).randn(hidden, hidden).astype(np.float32) / np.sqrt(hidden)
        self._cache: Dict[str, np.ndarray] = {}

    def _tok_vec(self, token: str) -> np.ndarray:
        v = self._cache.get(token)
        if v is None:
            h = abs(hash(("mock", token))) % (2**31)
            v = np.random.RandomState(h).randn(self.hidden_size).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            self._cache[token] = v
        return v

    def embed(self, texts: Sequence[str], L: int, pooling: str, normalize: bool,
              batch_size: int) -> np.ndarray:
        rows = []
        for t in texts:
            toks = str(t).split()[:max(1, L - 2)]  # ~L tokens minus specials
            if not toks:
                toks = ["<empty>"]
            mat = np.stack([self._tok_vec(tok) for tok in toks], 0)
            pooled = mat[0] if pooling == "cls" else mat.mean(0)
            rows.append(pooled)
        arr = np.stack(rows, 0) @ self._proj
        return _l2norm(arr) if normalize else arr.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# Factory + cache
# ════════════════════════════════════════════════════════════════════════════
def build_encoder(role: str, spec: dict, cfg: dict, device: str):
    backend = cfg["features"]["backend"]
    if backend == "mock":
        logger.warning("  *** MOCK ENCODER (%s) — logic smoke test only, NOT a real encoder ***", role)
        return MockEncoder(encoder_tag(role, spec["ref"]) + "-MOCK", role)
    fp16 = bool(cfg["features"].get("fp16", True))
    if spec["kind"] == "comet":
        return TorchEncoder.from_comet(role, spec["ref"], device, fp16)
    if spec["kind"] == "hf":
        return TorchEncoder.from_hf(role, spec["ref"], device, fp16)
    raise ValueError(f"Unknown encoder kind {spec['kind']!r}")


def cached_embed(encoder, doc_ids: Sequence[str], texts: Sequence[str], L: int,
                 cfg: dict, cache_dir: str) -> np.ndarray:
    pooling = cfg["features"]["pooling"]
    normalize = bool(cfg["features"]["normalize_l2"])
    key = content_hash(doc_ids, L, pooling, normalize)
    tag = f"{encoder.tag}__L{L}__{pooling}__norm{int(normalize)}__{key}.npy"
    path = Path(cache_dir) / tag
    if path.exists():
        arr = np.load(path)
    else:
        arr = encoder.embed(texts, L, pooling, normalize,
                            int(cfg["features"]["batch_size"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
    if not np.isfinite(arr).all():
        raise FloatingPointError(f"NaN/inf in features {tag}")
    return arr


def assert_same_tokenizer(enc_a, enc_b) -> None:
    """Verify the default pair shares the tokenizer so token-length buckets compare."""
    if not (hasattr(enc_a, "tok") and hasattr(enc_b, "tok")):
        return
    probe = "Le déficit budgétaire et la santé publique 12345"
    ia = enc_a.tok(probe)["input_ids"]
    ib = enc_b.tok(probe)["input_ids"]
    same_vocab = enc_a.tok.get_vocab() == enc_b.tok.get_vocab()
    print(f"  tokenizer check: vocab_size {enc_a.tok.vocab_size} vs {enc_b.tok.vocab_size} | "
          f"same vocab={same_vocab} | same ids on probe={ia == ib}")
    if not (same_vocab and ia == ib):
        logger.warning("  *** tokenizers differ — token-length buckets are NOT directly comparable ***")


# ════════════════════════════════════════════════════════════════════════════
# CLI: extract + cache for both encoders across all buckets + sanity
# ════════════════════════════════════════════════════════════════════════════
def extract_all(cfg: dict, df, buckets: List[int]) -> Dict[str, Dict[int, np.ndarray]]:
    ensure_dirs(cfg)
    device = pick_device(cfg["features"].get("device"))
    cache_dir = cfg["paths"]["feature_cache"]
    doc_ids = df["doc_id"].tolist()
    texts = df["text"].tolist()
    pair = active_encoders(cfg)
    encs = {role: build_encoder(role, spec, cfg, device) for role, spec in pair.items()}
    if "comet" in encs and "raw" in encs:
        assert_same_tokenizer(encs["comet"], encs["raw"])

    feats: Dict[str, Dict[int, np.ndarray]] = {}
    counts = {}
    for role, enc in encs.items():
        feats[enc.tag] = {}
        for L in buckets:
            arr = cached_embed(enc, doc_ids, texts, L, cfg, cache_dir)
            feats[enc.tag][L] = arr
            counts.setdefault(L, {})[enc.tag] = arr.shape
    banner("Feature extraction sanity")
    print(f"  device={device}  pooling={cfg['features']['pooling']}  "
          f"normalize_l2={cfg['features']['normalize_l2']}  fp16={cfg['features']['fp16']}")
    for L in buckets:
        shapes = counts[L]
        dims = {t: s[1] for t, s in shapes.items()}
        nitems = {t: s[0] for t, s in shapes.items()}
        same_n = len(set(nitems.values())) == 1
        print(f"  L={L:<4} dims={dims}  n_items={nitems}  same_n_per_bucket={same_n}")
        assert same_n, f"encoders disagree on #items at L={L}"
    return feats


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset", default=None, help="cap_dataset.csv (else build from data.py)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dataset:
        df = pd.read_csv(args.dataset)
    else:
        from data import build_dataset
        df = build_dataset(cfg, save=True)
    extract_all(cfg, df, cfg["length_buckets"])


if __name__ == "__main__":
    main()
