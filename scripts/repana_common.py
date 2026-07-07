#!/usr/bin/env python3
"""
repana_common.py — shared building blocks for the *representation-analysis* suite.

Motivation
----------
COMET's encoder (XLM-R-large) is trained to *predict translation quality*. That
objective has two faces that pull in opposite directions:

  (a) it must place genuinely parallel (src, mt) pairs close together, and
  (b) it must place *near*-parallel pairs that contain errors far enough apart
      that the regression head can tell good from bad.

Pure cross-lingual aligners (LaBSE, LASER, E5, SONAR) only ever optimise (a):
pull parallel sentences together with a contrastive / distance loss. The whole
point of this suite is to characterise how COMET's extra "push apart the
near-misses" pressure reshapes the multilingual representation space — and what
happens once the text is longer than the short segments COMET was trained on.

Everything downstream (E1–E5) imports from here so that **every encoder goes
through one identical embedding interface**. The only thing that varies between
runs is the encoder weights — which is exactly what makes "what did COMET change"
a fair question.

Encoder zoo (all pure `transformers`, no sentence-transformers needed)
----------------------------------------------------------------------
  comet:<id-or-ckpt>     COMET full pipeline (layerwise attention + avg pool).
                         e.g. comet:Unbabel/wmt22-comet-da   comet:/path/model.ckpt
  hf-mean:<hf_id>        Raw HF encoder, masked-mean pooled. The *untrained*
                         control — use hf-mean:xlm-roberta-large for COMET's own
                         backbone.
  labse                  LaBSE (contrastive parallel-sentence aligner; CLS+dense).
  e5[:<hf_id>]           multilingual-E5 (contrastive text embedder; mean pool,
                         "query:" prefix). Default intfloat/multilingual-e5-base.

All embeddings are L2-normalised by default (cosine geometry for retrieval).
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Languages
# ════════════════════════════════════════════════════════════════════════════
# Map ISO-639-1 → FLORES-200 code. The core set mirrors the Bio-MQM training
# pairs (en/de/es/fr/ru/zh) so the analysis lines up with the fine-tuning; the
# extended set adds typological + script diversity (incl. low-resource swh).
FLORES_CODE = {
    "en": "eng_Latn", "de": "deu_Latn", "es": "spa_Latn", "fr": "fra_Latn",
    "ru": "rus_Cyrl", "zh": "zho_Hans", "ar": "arb_Arab", "hi": "hin_Deva",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "tr": "tur_Latn", "vi": "vie_Latn",
    "sw": "swh_Latn", "el": "ell_Grek", "th": "tha_Thai", "bg": "bul_Cyrl",
    "ur": "urd_Arab",
}
CORE_LANGS = ["en", "de", "es", "fr", "ru", "zh"]
EXT_LANGS = CORE_LANGS + ["ar", "hi", "ja", "ko", "tr", "vi", "sw"]

# Languages with no whitespace word segmentation — perturbations operate on
# characters instead of space-delimited tokens for these.
NO_SPACE_LANGS = {"zh", "ja", "th"}


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
    """Unified embedding interface.

    Sub-classes implement ``_embed_raw`` (sentence embeddings, un-normalised) and,
    where meaningful, ``layer_reps`` (per-layer masked-mean reps for CKA) and
    ``layer_weights`` (COMET's learned layer mixture).
    """

    name: str = "embedder"
    hidden_size: int = 0
    is_xlmr_large: bool = False  # True for the 3 XLM-R-large variants (CKA group)

    def embed(self, texts: Sequence[str], batch_size: int = 64,
              normalize: bool = True) -> np.ndarray:
        raw = self._embed_raw(list(texts), batch_size)
        return _l2norm(raw) if normalize else raw

    # -- to be overridden --
    def _embed_raw(self, texts: List[str], batch_size: int) -> np.ndarray:
        raise NotImplementedError

    def layer_reps(self, texts: Sequence[str], batch_size: int = 64
                   ) -> np.ndarray:
        """Return (n_layers, N, D) masked-mean-pooled hidden states per layer."""
        raise NotImplementedError(f"{self.name} has no per-layer reps")

    def layer_weights(self) -> Optional[np.ndarray]:
        """COMET's learned (sparse)softmax layer-mixture weights, else None."""
        return None


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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

    @torch.no_grad()
    def layer_reps(self, texts, batch_size=64):
        per_layer: Optional[List[List[np.ndarray]]] = None
        for s in tqdm(range(0, len(texts), batch_size), desc=f"  {self.name}[layers]", leave=False):
            enc = self.model.encoder.prepare_sample(texts[s:s + batch_size])
            ids = enc["input_ids"].to(self.device)
            am = enc["attention_mask"].to(self.device)
            hs = self.model.encoder.model(
                input_ids=ids, attention_mask=am,
                output_hidden_states=True, return_dict=True).hidden_states
            if per_layer is None:
                per_layer = [[] for _ in hs]
            for li, h in enumerate(hs):
                per_layer[li].append(_masked_mean(h, am).cpu().float().numpy())
        return np.stack([np.concatenate(c, 0) for c in per_layer], 0)

    def layer_weights(self):
        la = getattr(self.model, "layerwise_attention", None)
        if la is None:
            return None
        w = torch.cat([p for p in la.scalar_parameters])
        return la.transform_fn(w, dim=0).detach().cpu().float().numpy()


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
        self.is_xlmr_large = (self.model.config.hidden_size == 1024
                              and "roberta" in self.model.config.model_type.lower())
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

    @torch.no_grad()
    def layer_reps(self, texts, batch_size=64):
        per_layer = None
        for s in tqdm(range(0, len(texts), batch_size), desc=f"  {self.name}[layers]", leave=False):
            enc = self._encode(texts[s:s + batch_size]).to(self.device)
            hs = self.model(**enc, output_hidden_states=True).hidden_states
            if per_layer is None:
                per_layer = [[] for _ in hs]
            for li, h in enumerate(hs):
                per_layer[li].append(_masked_mean(h, enc["attention_mask"]).cpu().float().numpy())
        return np.stack([np.concatenate(c, 0) for c in per_layer], 0)


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

    @torch.no_grad()
    def layer_reps(self, texts, batch_size=64):
        per_layer = None
        for s in tqdm(range(0, len(texts), batch_size), desc="  labse[layers]", leave=False):
            enc = self.tok(texts[s:s + batch_size], return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(self.device)
            hs = self.model(**enc, output_hidden_states=True).hidden_states
            if per_layer is None:
                per_layer = [[] for _ in hs]
            for li, h in enumerate(hs):
                per_layer[li].append(_masked_mean(h, enc["attention_mask"]).cpu().float().numpy())
        return np.stack([np.concatenate(c, 0) for c in per_layer], 0)


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

    @torch.no_grad()
    def layer_reps(self, texts, batch_size=64):
        per_layer = None
        for s in tqdm(range(0, len(texts), batch_size), desc="  e5[layers]", leave=False):
            enc = self.tok(self._prep(texts[s:s + batch_size]), return_tensors="pt",
                           padding=True, truncation=True, max_length=512).to(self.device)
            hs = self.model(**enc, output_hidden_states=True).hidden_states
            if per_layer is None:
                per_layer = [[] for _ in hs]
            for li, h in enumerate(hs):
                per_layer[li].append(_masked_mean(h, enc["attention_mask"]).cpu().float().numpy())
        return np.stack([np.concatenate(c, 0) for c in per_layer], 0)


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


# ════════════════════════════════════════════════════════════════════════════
# Data — FLORES-200 (multi-parallel) + pseudo-paragraph builder
# ════════════════════════════════════════════════════════════════════════════
# We read the *raw* FLORES-200 distribution (plain aligned text files), not the
# HF dataset scripts — those hardcode a `/datasets` path that is not writable on
# the cluster. The raw layout is:
#     <flores_dir>/{dev,devtest}/<code>.<split>     one sentence per line
#     <flores_dir>/metadata_<split>.tsv             URL/domain/... per line
# Line i is parallel across every language. To download once:
#     wget https://tinyurl.com/flores200dataset -O flores200_dataset.tar.gz
#     tar -xzf flores200_dataset.tar.gz             # -> flores200_dataset/
# Point at it with $FLORES_DIR; otherwise we auto-discover the copy HF already
# extracted into the datasets cache.


def _find_flores_dir() -> Optional[Path]:
    """Locate an extracted flores200_dataset directory."""
    import glob
    env = os.environ.get("FLORES_DIR")
    cands: List[str] = [env] if env else []
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/scratch/hf_cache"))
    cands += [
        os.path.expanduser("~/scratch/flores200_dataset"),
        "data/flores200_dataset",
    ]
    cands += glob.glob(os.path.join(hf_home, "datasets", "downloads", "extracted",
                                    "*", "flores200_dataset"))
    cands += glob.glob(os.path.expanduser(
        "~/.cache/huggingface/datasets/downloads/extracted/*/flores200_dataset"))
    for c in cands:
        if c and (Path(c) / "devtest").is_dir():
            return Path(c)
    return None


@dataclass
class FloresData:
    langs: List[str]                  # ISO codes actually loaded
    sentences: Dict[str, List[str]]   # lang -> N aligned sentences
    urls: List[str]                   # per-row article URL (for grouping)

    @property
    def n(self) -> int:
        return len(next(iter(self.sentences.values())))


def load_flores(langs: Sequence[str], split: str = "devtest",
                max_sents: Optional[int] = None,
                flores_dir: Optional[str] = None) -> FloresData:
    """Load FLORES-200 for several languages from the raw text files, row-aligned."""
    root = Path(flores_dir) if flores_dir else _find_flores_dir()
    if root is None or not (root / split).is_dir():
        raise FileNotFoundError(
            "Could not find an extracted flores200_dataset directory. Set "
            "$FLORES_DIR to it (see repana_common.py header for the download "
            "command).")
    logger.info(f"  FLORES root: {root}")

    # article URLs for pseudo-paragraph grouping (one per sentence row)
    urls: List[str] = []
    meta = root / f"metadata_{split}.tsv"
    if meta.is_file():
        with open(meta, encoding="utf-8") as fh:
            rows = fh.read().splitlines()
        urls = [r.split("\t")[0] for r in rows[1:]]  # skip header; col0 = URL

    sentences: Dict[str, List[str]] = {}
    kept: List[str] = []
    ref_n = None
    for lang in langs:
        code = FLORES_CODE.get(lang)
        if code is None:
            logger.warning(f"  no FLORES code for {lang!r}, skipping")
            continue
        fp = root / split / f"{code}.{split}"
        if not fp.is_file():
            logger.warning(f"  missing FLORES file {fp}, skipping {lang}")
            continue
        with open(fp, encoding="utf-8") as fh:
            sents = fh.read().splitlines()
        if max_sents:
            sents = sents[:max_sents]
        if ref_n is None:
            ref_n = len(sents)
        elif len(sents) != ref_n:
            raise RuntimeError(f"FLORES length mismatch for {lang}: "
                               f"{len(sents)} vs {ref_n}")
        sentences[lang] = sents
        kept.append(lang)
    if not kept:
        raise RuntimeError("No FLORES languages loaded.")
    if not urls or len(urls) < ref_n:
        urls = [str(i) for i in range(ref_n)]      # fallback: every row its own "article"
    urls = urls[:ref_n]
    logger.info(f"  FLORES {split}: {len(kept)} langs × {ref_n} sentences")
    return FloresData(langs=kept, sentences=sentences, urls=urls)


# ────────────────────────────────────────────────────────────────────────────
# Official FLORES+ (OLDI) loader — the access path documented on the dataset card
# ────────────────────────────────────────────────────────────────────────────
# Unlike load_flores() above (which reads the legacy flores200_dataset text dump),
# this follows the *official* FLORES+ guidelines verbatim:
#
#     from datasets import load_dataset
#     ds = load_dataset("openlanguagedata/flores_plus", "fra_Latn", split="devtest")
#
# FLORES+ is gated: you must (1) `pip install datasets`, (2) log in to the HF Hub
# (`hf auth login` or huggingface_hub.login()), and (3) accept the terms of use at
# https://huggingface.co/datasets/openlanguagedata/flores_plus once.
#
# Each languoid config is "<iso_639_3>_<iso_15924>" and rows carry a shared `id`
# field; rows with the same id+split are mutual translations, so we align across
# languages by id to recover the multi-parallel matrix. NB the codes differ from
# the legacy FLORES-200 ones in places — most notably Mandarin is cmn_Hans (not
# zho_Hans). Names/codes follow the dataset card's language-coverage table.
FLORES_PLUS_REPO = "openlanguagedata/flores_plus"
FLORES_PLUS_CODE = {
    "en": "eng_Latn", "de": "deu_Latn", "es": "spa_Latn", "fr": "fra_Latn",
    "ru": "rus_Cyrl", "zh": "cmn_Hans", "ar": "arb_Arab", "hi": "hin_Deva",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "tr": "tur_Latn", "vi": "vie_Latn",
    "sw": "swh_Latn", "el": "ell_Grek", "th": "tha_Thai", "bg": "bul_Cyrl",
    "ur": "urd_Arab",
}


def load_flores_plus(langs: Sequence[str], split: str = "devtest",
                     max_sents: Optional[int] = None,
                     repo: str = FLORES_PLUS_REPO) -> FloresData:
    """Load the official FLORES+ benchmark (OLDI), row-aligned across languages.

    Follows the dataset card's recommended access pattern
    ``load_dataset(repo, "<iso_639_3>_<iso_15924>", split=split)`` per languoid,
    then intersects the shared ``id`` field so every loaded language is parallel.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError("FLORES+ needs the `datasets` package (pip install datasets).") from exc

    # Resolve the stored HF token explicitly: FLORES+ is gated, and the datasets /
    # hf:// download path only authenticates when HF_TOKEN is in the environment.
    # NOTE huggingface_hub.get_token() reads $HF_HOME/token, so when HF_HOME is
    # redirected (e.g. to scratch) it misses the default ~/.cache/huggingface/token
    # — hence the explicit default-file fallback below.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:  # noqa: BLE001
            token = None
    if not token:
        default_token = Path.home() / ".cache" / "huggingface" / "token"
        if default_token.is_file():
            token = default_token.read_text(encoding="utf-8").strip() or None
    if token:
        # make the resolved token visible to the fsspec/datasets download path
        os.environ.setdefault("HF_TOKEN", token)

    per_lang_rows: Dict[str, Dict[str, str]] = {}   # lang -> {id: text}
    meta_url: Dict[str, str] = {}                    # id -> article url (for E3 grouping)
    kept: List[str] = []
    for lang in langs:
        code = FLORES_PLUS_CODE.get(lang)
        if code is None:
            logger.warning(f"  no FLORES+ config for {lang!r}, skipping")
            continue
        try:
            ds = load_dataset(repo, code, split=split, token=token)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load FLORES+ config {code!r} (split={split!r}) from {repo}. "
                "Make sure you are logged in to the HF Hub (`hf auth login` or "
                "huggingface_hub.login()) and have accepted the dataset terms at "
                f"https://huggingface.co/datasets/{repo}.") from exc
        rows: Dict[str, str] = {}
        for r in ds:
            rid = str(r["id"])
            rows[rid] = r["text"]
            meta_url.setdefault(rid, r.get("url", "") or "")
        per_lang_rows[lang] = rows
        kept.append(lang)
        logger.info(f"  FLORES+ {code} {split}: {len(rows)} rows")
    if not kept:
        raise RuntimeError("No FLORES+ languages loaded.")

    # keep only ids present in every language → guaranteed multi-parallel; sort numerically
    common = set.intersection(*(set(per_lang_rows[l]) for l in kept))

    def _key(i: str):
        try:
            return (0, int(i))
        except ValueError:
            return (1, i)

    ids = sorted(common, key=_key)
    if max_sents:
        ids = ids[:max_sents]
    if not ids:
        raise RuntimeError("FLORES+: no ids shared across the requested languages.")

    sentences = {l: [per_lang_rows[l][i] for i in ids] for l in kept}
    urls = [meta_url.get(i, "") or i for i in ids]
    logger.info(f"  FLORES+ {split}: {len(kept)} langs × {len(ids)} aligned sentences")
    return FloresData(langs=kept, sentences=sentences, urls=urls)


def load_flores_source(source: str, langs: Sequence[str], split: str = "devtest",
                       max_sents: Optional[int] = None) -> FloresData:
    """Dispatch FLORES loading by source.

    ``source="plus"`` → official OLDI FLORES+ via the HF Hub (gated; recommended).
    ``source="raw"``  → legacy flores200_dataset local text files.
    """
    if source == "plus":
        return load_flores_plus(langs, split, max_sents)
    if source == "raw":
        return load_flores(langs, split, max_sents)
    raise ValueError(f"Unknown flores source {source!r} (use 'plus' or 'raw').")


def build_pseudo_docs(data: FloresData, k: int) -> Dict[str, List[str]]:
    """Concatenate k consecutive same-article sentences into pseudo-paragraphs.

    FLORES rows are in document order and the URL column marks article
    boundaries, so windows stay semantically coherent — and stay parallel across
    languages because we apply identical row windows to every language.
    k=1 returns the original sentences.
    """
    if k <= 1:
        return {l: list(s) for l, s in data.sentences.items()}
    # contiguous windows that never cross an article boundary
    windows: List[List[int]] = []
    i, n = 0, data.n
    while i < n:
        j = i
        while j < n and j - i < k and data.urls[j] == data.urls[i]:
            j += 1
        if j - i == k:               # only keep full-length windows
            windows.append(list(range(i, j)))
        i = j if j > i else i + 1
    joiner = {l: ("" if l in NO_SPACE_LANGS else " ") for l in data.langs}
    out: Dict[str, List[str]] = {}
    for lang, sents in data.sentences.items():
        out[lang] = [joiner[lang].join(sents[w] for w in win) for win in windows]
    return out


# ════════════════════════════════════════════════════════════════════════════
# Hard-negative generation (xsim++-style perturbations = "translation errors")
# ════════════════════════════════════════════════════════════════════════════
import random as _random

_DIGITS = "0123456789"


def _tokens(text: str, lang: str) -> Tuple[List[str], bool]:
    """Return (tokens, is_char_level). Char-level for no-space languages."""
    if lang in NO_SPACE_LANGS:
        return list(text), True
    return text.split(" "), False


def _detok(tokens: List[str], char_level: bool) -> str:
    return ("" if char_level else " ").join(tokens)


def perturb(text: str, lang: str, kind: str, rng: _random.Random) -> Optional[str]:
    """Apply one error-like perturbation. Returns None if not applicable.

    kinds
    -----
      number   replace one digit with a different digit  (number error)
      delete   drop one content token                    (omission)
      swap     swap two adjacent content tokens          (reordering)
      replace  overwrite one token with another from the same text (mistranslation)
    """
    toks, char_level = _tokens(text, lang)
    if len(toks) < 3:
        return None
    idxs = list(range(len(toks)))

    if kind == "number":
        digit_pos = [(i, j) for i in idxs for j, c in enumerate(toks[i]) if c in _DIGITS]
        if not digit_pos:
            return None
        i, j = rng.choice(digit_pos)
        old = toks[i][j]
        new = rng.choice([d for d in _DIGITS if d != old])
        toks[i] = toks[i][:j] + new + toks[i][j + 1:]
        return _detok(toks, char_level)

    if kind == "delete":
        i = rng.randrange(len(toks))
        del toks[i]
        return _detok(toks, char_level)

    if kind == "swap":
        if len(toks) < 4:
            return None
        i = rng.randrange(len(toks) - 1)
        toks[i], toks[i + 1] = toks[i + 1], toks[i]
        out = _detok(toks, char_level)
        return out if out != text else None

    if kind == "replace":
        if len(set(toks)) < 2:
            return None
        i = rng.randrange(len(toks))
        choices = [t for t in toks if t != toks[i]]
        if not choices:
            return None
        toks[i] = rng.choice(choices)
        return _detok(toks, char_level)

    raise ValueError(f"unknown perturbation kind {kind!r}")


PERTURBATIONS = ["number", "delete", "swap", "replace"]


# ════════════════════════════════════════════════════════════════════════════
# Metrics — cross-lingual retrieval (xsim) and CKA
# ════════════════════════════════════════════════════════════════════════════
def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A,B assumed L2-normalised → cosine = dot product."""
    return A @ B.T


def xsim_error(src: np.ndarray, tgt: np.ndarray, margin: str = "ratio",
               k: int = 4) -> Tuple[float, np.ndarray]:
    """LASER-style xsim error rate for aligned src/tgt (row i ↔ row i).

    margin="cosine"  : nearest neighbour by plain cosine.
    margin="ratio"   : margin-based score cos(x,y) / mean_k of the two sides'
                       top-k neighbour cosines (robust to hubness; LASER default).
    Returns (error_rate, predicted_indices).
    """
    sim = cosine_matrix(src, tgt)  # (N, N)
    if margin == "cosine":
        pred = sim.argmax(1)
    elif margin == "ratio":
        n = sim.shape[0]
        kk = min(k, n - 1) if n > 1 else 1
        # average top-k similarity from each side (exclude self via partition)
        fwd = np.sort(sim, axis=1)[:, -kk:].mean(1, keepdims=True)        # (N,1)
        bwd = np.sort(sim, axis=0)[-kk:, :].mean(0, keepdims=True)        # (1,N)
        denom = (fwd + bwd) / 2.0
        pred = (sim / np.clip(denom, 1e-9, None)).argmax(1)
    else:
        raise ValueError(margin)
    err = float((pred != np.arange(len(pred))).mean())
    return err, pred


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representation matrices (N×D1, N×D2).

    Permutation/-isotropy-robust similarity of two representation spaces; 1 =
    identical up to rotation/scale, 0 = unrelated. Uses the HSIC formulation,
    centring features over the N examples.
    """
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xtx = X.T @ X
    yty = Y.T @ Y
    xty = X.T @ Y
    hsic = float((xty ** 2).sum())
    denom = float(np.sqrt((xtx ** 2).sum()) * np.sqrt((yty ** 2).sum()))
    return hsic / denom if denom > 0 else 0.0
