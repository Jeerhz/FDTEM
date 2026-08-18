#!/usr/bin/env python3
"""
xglue_common.py

Shared building blocks for the two XGlue evaluation procedures:

  * probe_xglue.py     — frozen encoder + linear probe
  * finetune_xglue.py  — full end-to-end fine-tuning

Both procedures import from here so that a plain HuggingFace encoder and the
encoder extracted from a COMET checkpoint are loaded, tokenized, pooled and
scored through *exactly* the same code path. The only variable across runs is
the encoder's initial weights — which is what makes "how do COMET's
representations differ from XLM-R" a fair question to ask.

Nothing here is task- or procedure-specific beyond what is strictly shared.
"""
from __future__ import annotations

import contextlib
import io
import logging
import os
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── language lists (English is always the training language) ───────────────────

XNLI_LANGS = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
PAWS_LANGS  = ["de", "en", "es", "fr", "ja", "ko", "zh"]
NC_LANGS    = ["de", "en", "es", "fr", "ru"]

# Per-task constants. `max_len`/`epochs` are recipe defaults the entry points may
# override; `num_labels` and `langs` are intrinsic to the task.
TASK_INFO = {
    "xnli":   {"num_labels": 3,  "max_len": 128, "epochs": 3, "langs": XNLI_LANGS,
               "label_names": ["entailment", "neutral", "contradiction"]},
    "paws-x": {"num_labels": 2,  "max_len": 128, "epochs": 5, "langs": PAWS_LANGS,
               "label_names": ["not_paraphrase", "paraphrase"]},
    "nc":     {"num_labels": 10, "max_len": 256, "epochs": 3, "langs": NC_LANGS,
               "label_names": None},  # discovered from the data
}


# ── encoder loading (unified for HF and COMET) ─────────────────────────────────

class EncoderAdapter(nn.Module):
    """Normalises any transformer encoder so ``forward`` returns the last hidden
    state of shape (batch, seq_len, hidden).

    Works for both a HuggingFace ``AutoModel`` and the ``XLMRobertaModel`` that
    lives inside a COMET checkpoint (``comet_model.encoder.model``); both return
    an object exposing ``.last_hidden_state``.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state


def load_encoder(spec: str) -> Tuple[EncoderAdapter, "transformers.PreTrainedTokenizerBase", int, str]:
    """Load an encoder from a spec string.

    Spec format
    -----------
      hf:<model_id>          e.g.  hf:xlm-roberta-base   hf:xlm-roberta-large
      comet:<id-or-ckpt>     e.g.  comet:Unbabel/wmt22-comet-da
                                   comet:/path/to/model.ckpt

    Returns
    -------
      (adapter, tokenizer, hidden_size, tag)
    """
    kind, _, ref = spec.partition(":")
    if not ref:
        raise ValueError(f"Encoder spec must be 'hf:<id>' or 'comet:<id-or-ckpt>', got {spec!r}")

    if kind == "hf":
        from transformers import AutoModel, AutoTokenizer
        logger.info(f"Loading HF encoder: {ref}")
        tokenizer = AutoTokenizer.from_pretrained(ref)
        base = AutoModel.from_pretrained(ref)
        hidden = base.config.hidden_size
        tag = ref.split("/")[-1]

    elif kind == "comet":
        from comet import download_model, load_from_checkpoint
        ckpt = ref if os.path.isfile(ref) else download_model(ref)
        logger.info(f"Loading COMET encoder from: {ckpt}")
        cmodel = load_from_checkpoint(ckpt)
        enc = cmodel.encoder        # XLMREncoder / BERTEncoder
        base = enc.model            # underlying HF transformer (e.g. XLMRobertaModel)
        tokenizer = enc.tokenizer
        hidden = base.config.hidden_size
        tag = ref.split("/")[-1].replace(".ckpt", "")

    else:
        raise ValueError(f"Unknown encoder kind {kind!r}. Use 'hf' or 'comet'.")

    adapter = EncoderAdapter(base)
    n_params = sum(p.numel() for p in adapter.parameters()) / 1e6
    logger.info(f"Encoder loaded  ({n_params:.1f}M params, hidden={hidden})")
    return adapter, tokenizer, hidden, tag


# ── pooling (identical readout for every encoder) ──────────────────────────────

def pool(hidden: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Reduce token embeddings (B, L, D) to a sentence embedding (B, D).

    mode="cls"  : first token (<s>/[CLS]).
    mode="mean" : attention-masked mean over tokens — the robust default for a
                  *frozen* encoder, where the first token carries little signal.

    Using the same pooling for both encoders is essential: it isolates the
    representation difference from any difference in readout.
    """
    if mode == "cls":
        return hidden[:, 0, :]
    if mode == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    raise ValueError(f"Unknown pooling mode {mode!r}. Use 'cls' or 'mean'.")


# ── data loading ───────────────────────────────────────────────────────────────
# Every example is a dict {"a": str, "b": Optional[str], "label": int}.
# Training data is always English; test data spans the requested languages.

def _parse_nc_rows(fileobj):
    # Read the whole member as bytes — a streaming tar's ExFileObject isn't
    # seekable, so io.TextIOWrapper can't wrap it.
    rows = []
    for line in fileobj.read().decode("utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        title, body, category = parts[0], parts[1], parts[2]
        rows.append((title + " " + body, category))
    return rows


def build_data(
    task: str,
    languages: List[str],
    max_train: Optional[int],
    xglue_tar: Optional[str],
) -> Tuple[List[dict], Dict[str, List[dict]], List[str]]:
    """Return (train_examples, test_examples_by_lang, label_names)."""
    if task == "xnli":
        from datasets import load_dataset
        train = load_dataset("xnli", "en", split="train")
        if max_train:
            train = train.select(range(min(max_train, len(train))))
        train_ex = [{"a": p, "b": h, "label": y}
                    for p, h, y in zip(train["premise"], train["hypothesis"], train["label"])]
        test_by_lang = {}
        for lang in languages:
            try:
                ds = load_dataset("xnli", lang, split="test")
            except Exception as exc:
                logger.warning(f"  Skipping {lang}: {exc}")
                continue
            test_by_lang[lang] = [{"a": p, "b": h, "label": y}
                                  for p, h, y in zip(ds["premise"], ds["hypothesis"], ds["label"])]
        return train_ex, test_by_lang, TASK_INFO["xnli"]["label_names"]

    if task == "paws-x":
        from datasets import load_dataset
        train = load_dataset("google-research-datasets/paws-x", "en", split="train")
        if max_train:
            train = train.select(range(min(max_train, len(train))))
        train_ex = [{"a": s1, "b": s2, "label": y}
                    for s1, s2, y in zip(train["sentence1"], train["sentence2"], train["label"])]
        test_by_lang = {}
        for lang in languages:
            try:
                ds = load_dataset("google-research-datasets/paws-x", lang, split="test")
            except Exception as exc:
                logger.warning(f"  Skipping {lang}: {exc}")
                continue
            test_by_lang[lang] = [{"a": s1, "b": s2, "label": y}
                                  for s1, s2, y in zip(ds["sentence1"], ds["sentence2"], ds["label"])]
        return train_ex, test_by_lang, TASK_INFO["paws-x"]["label_names"]

    if task == "nc":
        if not xglue_tar or not os.path.isfile(xglue_tar):
            raise FileNotFoundError("NC requires --xglue_tar path/to/xglue_full_dataset.tar.gz")
        # Stream the archive (r|gz, never seeks) and pull the NC splits in a single
        # forward pass, stopping as soon as we have them. The NC directory sits near
        # the front of the tar, so this stays robust even if the archive is truncated
        # — a random-access "r:gz" open would scan to the end and choke on a bad tail.
        train_member = "xglue_full_dataset/NC/xglue.nc.en.train"
        test_members = {f"xglue_full_dataset/NC/xglue.nc.{lang}.test": lang for lang in languages}
        train_rows = None
        test_raw: Dict[str, list] = {}
        try:
            with tarfile.open(xglue_tar, "r|gz") as tar:
                for member in tar:
                    if member.name == train_member:
                        f = tar.extractfile(member)
                        if f is not None:
                            train_rows = _parse_nc_rows(f)
                    elif member.name in test_members:
                        f = tar.extractfile(member)
                        if f is not None:
                            test_raw[test_members[member.name]] = _parse_nc_rows(f)
                    if train_rows is not None and len(test_raw) == len(test_members):
                        break
        except (tarfile.ReadError, EOFError) as exc:
            logger.warning(f"  NC tar ended early ({exc}); using the splits read so far.")
        if train_rows is None:
            raise RuntimeError("Could not read English NC training data from the tar.")
        label_names = sorted({c for _, c in train_rows})
        label2id = {c: i for i, c in enumerate(label_names)}
        if max_train:
            train_rows = train_rows[:max_train]
        train_ex = [{"a": a, "b": None, "label": label2id[c]} for a, c in train_rows]
        test_by_lang = {}
        for lang in languages:
            rows = test_raw.get(lang)
            if rows is None:
                logger.warning(f"  Skipping {lang}: no test file")
                continue
            test_by_lang[lang] = [{"a": a, "b": None, "label": label2id[c]}
                                  for a, c in rows if c in label2id]
        return train_ex, test_by_lang, label_names

    raise ValueError(f"Unknown task {task!r}")


def make_collate(tokenizer, max_len: int):
    """Collate fn that tokenizes a batch of examples (joint pair encoding when
    a second segment is present) into (input_ids, attention_mask, labels)."""
    def collate(batch: List[dict]):
        a = [ex["a"] for ex in batch]
        has_pair = batch[0]["b"] is not None
        b = [ex["b"] for ex in batch] if has_pair else None
        enc = tokenizer(
            a, b,
            truncation=True, max_length=max_len, padding=True, return_tensors="pt",
        )
        labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
        return enc["input_ids"], enc["attention_mask"], labels
    return collate


# ── precision helper ───────────────────────────────────────────────────────────

def resolve_device_and_dtype(device_arg: Optional[str], precision: str):
    """Pick a device and an autocast dtype (autocast only on CUDA)."""
    if device_arg:
        device = device_arg
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    autocast_dtype = None
    if device.startswith("cuda"):
        if precision == "bf16" and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        elif precision in ("fp16", "bf16"):
            autocast_dtype = torch.float16
    return device, autocast_dtype


def amp_context(device: str, dtype):
    """autocast context when a dtype is set, else a no-op."""
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.split(":")[0], dtype=dtype)


# ── metrics ─────────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }


def average_over_langs(per_lang: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    langs = [l for l in per_lang if l != "avg"]
    return {k: float(np.mean([per_lang[l][k] for l in langs]))
            for k in ("accuracy", "f1_macro")}


# ── plotting ─────────────────────────────────────────────────────────────────────

def plot_task(task: str, per_lang: Dict[str, Dict[str, float]],
              out_dir: Path, run_name: str, subtitle: str = "") -> None:
    langs = [l for l in per_lang if l != "avg"]
    if not langs:
        return
    accs = [per_lang[l]["accuracy"] for l in langs]
    f1s  = [per_lang[l]["f1_macro"] for l in langs]

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(langs) * 0.55 + 1.5)))
    title = f"{run_name}  —  {task.upper()}"
    if subtitle:
        title += f"  ({subtitle})"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, values, label, color in zip(
        axes, [accs, f1s], ["Accuracy", "F1-macro"], ["#4C72B0", "#DD8452"]
    ):
        bars = ax.barh(langs, values, color=color, alpha=0.85)
        if "avg" in per_lang:
            key = "accuracy" if label == "Accuracy" else "f1_macro"
            ax.axvline(per_lang["avg"][key], color="black", linestyle="--", linewidth=1.2,
                       label=f"avg={per_lang['avg'][key]:.3f}")
            ax.legend(fontsize=8)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_xlabel(label); ax.set_xlim(0, 1.05); ax.set_title(label); ax.invert_yaxis()

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}_{task}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  [plot] {out_path}")
