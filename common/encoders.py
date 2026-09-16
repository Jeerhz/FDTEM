"""Encoder zoo — one uniform `Embedder` interface over four model families.

    comet:<id-or-ckpt>   COMET pipeline (layerwise attention + avg pool)
    hf-mean:<hf_id>      raw HF encoder, masked-mean pooled (untrained control)
    labse                LaBSE (CLS + dense)
    e5[:<hf_id>]         multilingual-E5 (mean pool, "query:" prefix)

Use `build_embedder(ref)` to get one, `cached_embed(...)` to embed with an
on-disk cache keyed by (encoder, text set).
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# Device helper
# ════════════════════════════════════════════════════════════════════════════
def pick_device(device_arg: Optional[str] = None) -> str:
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


# ════════════════════════════════════════════════════════════════════════════
# Embedder zoo
# ════════════════════════════════════════════════════════════════════════════
class Embedder:
    """Unified embedding interface. Sub-classes implement ``_embed_raw``
    (sentence embeddings, un-normalised); ``embed`` adds optional L2-norm."""

    name: str = "embedder"
    hidden_size: int = 0

    def embed(self, texts: Sequence[str], batch_size: int = 64,
              normalize: bool = True) -> np.ndarray:
        raw = self._embed_raw(list(texts), batch_size)
        return _l2norm(raw) if normalize else raw

    # -- to be overridden --
    def _embed_raw(self, texts: List[str], batch_size: int) -> np.ndarray:
        raise NotImplementedError


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked mean pooling over sequence dimension."""
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


class CometEmbedder(Embedder):
    """COMET checkpoint → sentence embedding via the *real* COMET pipeline
    (layerwise attention over all layers + average pooling), identical to what
    COMET uses when scoring. Loaded from a HF id or a local .ckpt."""

    def __init__(self, ref: str, device: str):
        from comet import download_model, load_from_checkpoint
        ckpt = ref if os.path.isfile(ref) else download_model(ref)
        logger.info(f"[comet] loading {ckpt}")
        self.model = load_from_checkpoint(ckpt).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.caching = False  # disable LRU cache (we feed fresh batches)
        self.device = device
        self.hidden_size = self.model.encoder.model.config.hidden_size
        self.is_xlmr_large = "xlm-roberta-large" in str(
            getattr(self.model.hparams, "pretrained_model", "")).lower()
        # informative label (keeps the run-id dir for checkpoints, e.g.
        # 'comet:4yeqp7cn-last') — used for metric keys, JSON, and the embed cache.
        self.name = enc_tag("comet:" + ref)

    @torch.no_grad()
    def _embed_raw(self, texts, batch_size):
        out = []
        for s in tqdm(range(0, len(texts), batch_size), desc=f"  {self.name}", leave=False):
            enc = self.model.encoder.prepare_sample(texts[s:s + batch_size])
            ids = enc["input_ids"].to(self.device)
            am = enc["attention_mask"].to(self.device)
            tt = enc.get("token_type_ids")
            tt = tt.to(self.device) if tt is not None else None
            emb = self.model.get_sentence_embedding(ids, am, tt)
            out.append(emb.cpu().float().numpy())
        return np.concatenate(out, 0)


class HFMeanEmbedder(Embedder):
    """Raw HuggingFace encoder, masked-mean pooled. The untrained control."""

    def __init__(self, hf_id: str, device: str):
        from transformers import AutoModel, AutoTokenizer
        logger.info(f"[hf-mean] loading {hf_id}")
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModel.from_pretrained(hf_id).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.hidden_size = self.model.config.hidden_size
        self.name = "xlmr:" + hf_id.split("/")[-1]

    def _encode(self, batch):
        return self.tok(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512)

    @torch.no_grad()
    def _embed_raw(self, texts, batch_size):
        out = []
        for s in tqdm(range(0, len(texts), batch_size), desc=f"  {self.name}", leave=False):
            enc = self._encode(texts[s:s + batch_size]).to(self.device)
            h = self.model(**enc).last_hidden_state
            out.append(_masked_mean(h, enc["attention_mask"]).cpu().float().numpy())
        return np.concatenate(out, 0)


class LaBSEEmbedder(Embedder):
    """LaBSE — the canonical contrastive parallel-sentence aligner. The sentence
    embedding is the pooler output (dense+tanh over CLS), L2-normalised."""

    def __init__(self, device: str, hf_id: str = "setu4993/LaBSE"):
        from transformers import AutoModel, AutoTokenizer
        logger.info(f"[labse] loading {hf_id}")
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModel.from_pretrained(hf_id).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.hidden_size = self.model.config.hidden_size
        self.name = "labse"

    @torch.no_grad()
    def _embed_raw(self, texts, batch_size):
        out = []
        for s in tqdm(range(0, len(texts), batch_size), desc="  labse", leave=False):
            enc = self.tok(texts[s:s + batch_size], return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(self.device)
            out.append(self.model(**enc).pooler_output.cpu().float().numpy())
        return np.concatenate(out, 0)


class E5Embedder(Embedder):
    """multilingual-E5 — modern contrastive text embedder. Mean pool + the
    mandatory "query: " prefix, L2-normalised."""

    def __init__(self, device: str, hf_id: str = "intfloat/multilingual-e5-base"):
        from transformers import AutoModel, AutoTokenizer
        logger.info(f"[e5] loading {hf_id}")
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModel.from_pretrained(hf_id).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.hidden_size = self.model.config.hidden_size
        self.name = "e5:" + hf_id.split("/")[-1]

    def _prep(self, batch):
        return ["query: " + t for t in batch]

    @torch.no_grad()
    def _embed_raw(self, texts, batch_size):
        out = []
        for s in tqdm(range(0, len(texts), batch_size), desc="  e5", leave=False):
            enc = self.tok(self._prep(texts[s:s + batch_size]), return_tensors="pt",
                           padding=True, truncation=True, max_length=512).to(self.device)
            h = self.model(**enc).last_hidden_state
            out.append(_masked_mean(h, enc["attention_mask"]).cpu().float().numpy())
        return np.concatenate(out, 0)


def enc_tag(spec: str) -> str:
    """A short, human-readable label for an encoder spec — for W&B run tags and
    log lines, computed *without* loading the model. Mirrors the Embedder naming,
    but for COMET checkpoint paths it keeps the run-id folder so bio checkpoints
    are identifiable, e.g. '.../4yeqp7cn/checkpoints/last.ckpt' → 'comet:4yeqp7cn-last'.
    """
    kind, _, ref = spec.partition(":")
    if kind == "comet":
        if "/" in ref or os.path.isfile(ref):
            parts = [p for p in ref.split("/") if p]
            leaf = parts[-1].replace(".ckpt", "")
            if "checkpoints" in parts:
                # use the LAST 'checkpoints' (paths can nest e.g. scratch/checkpoints/.../<run>/checkpoints/last.ckpt)
                ci = len(parts) - 1 - parts[::-1].index("checkpoints")
                if ci > 0:
                    return f"comet:{parts[ci - 1]}-{leaf}"
            return f"comet:{leaf}"
        return f"comet:{ref.split('/')[-1]}"
    if kind in ("hf-mean", "xlmr"):
        return f"xlmr:{ref.split('/')[-1]}"
    if kind == "labse":
        return "labse"
    if kind == "e5":
        return "e5:" + (ref.split("/")[-1] if ref else "multilingual-e5-base")
    return spec


def build_embedder(spec: str, device: str) -> Embedder:
    """spec → Embedder. See module docstring for the spec grammar."""
    kind, _, ref = spec.partition(":")
    if kind == "comet":
        return CometEmbedder(ref, device)
    if kind == "hf-mean" or kind == "xlmr":
        return HFMeanEmbedder(ref, device)
    if kind == "labse":
        return LaBSEEmbedder(device, ref) if ref else LaBSEEmbedder(device)
    if kind == "e5":
        return E5Embedder(device, ref) if ref else E5Embedder(device)
    raise ValueError(f"Unknown embedder spec {spec!r}. "
                     "Use comet:<id>, hf-mean:<id>, labse, or e5[:<id>].")


# ════════════════════════════════════════════════════════════════════════════
# Embedding cache  (experiments share embeddings — embed once, reuse everywhere)
# ════════════════════════════════════════════════════════════════════════════
def cached_embed(emb: Embedder, texts: Sequence[str], cache_key: str,
                 cache_dir: Optional[str], batch_size: int,
                 normalize: bool = True) -> np.ndarray:
    if cache_dir is None:
        return emb.embed(texts, batch_size, normalize)
    h = hashlib.md5(("\n".join(texts)).encode("utf-8")).hexdigest()[:10]
    tag = f"{emb.name}__{cache_key}__n{len(texts)}__{h}__norm{int(normalize)}"
    tag = tag.replace("/", "_").replace(":", "-")
    path = Path(cache_dir) / f"{tag}.npy"
    if path.exists():
        return np.load(path)
    arr = emb.embed(texts, batch_size, normalize)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


