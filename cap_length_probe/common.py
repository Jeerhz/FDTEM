#!/usr/bin/env python3
"""
common.py — shared building blocks for cap_length_probe.

Config loading (yaml + ${ENV} expansion), the CAP major-topic codebook and label
maps, the BTL language-name → ISO map, deterministic hashing for the feature
cache, and small helpers. No heavy deps here (no torch / transformers), so this
module imports cleanly in the logic-only smoke environment.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

# ════════════════════════════════════════════════════════════════════════════
# CAP major-topic codebook (the ~21 target classes)
# ════════════════════════════════════════════════════════════════════════════
# CAP major-topic codes in canonical order. These are the 21 target classes; the
# non-topic "No Policy Content" (999) is excluded by default (data.drop_no_policy_content).
CAP_CODES: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23]

CAP_LABEL_NAMES: Dict[int, str] = {
    1: "Macroeconomics", 2: "Civil Rights", 3: "Health", 4: "Agriculture",
    5: "Labor", 6: "Education", 7: "Environment", 8: "Energy", 9: "Immigration",
    10: "Transportation", 12: "Law and Crime", 13: "Social Welfare", 14: "Housing",
    15: "Domestic Commerce", 16: "Defense", 17: "Technology", 18: "Foreign Trade",
    19: "International Affairs", 20: "Government Operations", 21: "Public Lands",
    23: "Culture", 999: "No Policy Content",
}

# The "Beyond Token Limits" pipeline (00_data_clean.ipynb) relabels CAP codes to a
# 0-indexed space with this mapping: code -> contiguous id. 999 (NPC) -> 21.
CAP_CODE_TO_BTL_IDX: Dict[int, int] = {c: i for i, c in enumerate(CAP_CODES)}
CAP_CODE_TO_BTL_IDX[999] = 21
BTL_IDX_TO_CAP_CODE: Dict[int, int] = {i: c for c, i in CAP_CODE_TO_BTL_IDX.items()}

# Contiguous classifier label space over the 21 *target* topics (NPC excluded).
CAP_CODE_TO_LABEL: Dict[int, int] = {c: i for i, c in enumerate(CAP_CODES)}
LABEL_TO_CAP_CODE: Dict[int, int] = {i: c for c, i in CAP_CODE_TO_LABEL.items()}
N_CLASSES = len(CAP_CODES)

# BTL stores language as a lowercase full name; map to ISO-639-1.
LANG_NAME_TO_ISO: Dict[str, str] = {
    "english": "en", "hungarian": "hu", "dutch": "nl", "french": "fr",
    "italian": "it", "german": "de", "spanish": "es", "portuguese": "pt",
}
ISO_TO_LANG_NAME = {v: k for k, v in LANG_NAME_TO_ISO.items()}


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} and ${VAR:-default} in strings, and ~ in paths."""
    if isinstance(value, str):
        def repl(m: "re.Match") -> str:
            var, default = m.group(1), m.group(2)
            return os.environ.get(var, default if default is not None else "")
        out = _ENV_RE.sub(repl, value)
        return out
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load config.yaml, expand env vars, return a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return _expand_env(cfg)


def expanduser_path(p: str | None) -> str | None:
    return os.path.expanduser(p) if p else p


def active_encoders(cfg: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Return {'comet': {...}, 'raw': {...}} for the active pair."""
    pair = cfg["encoders"]["active_pair"]
    return cfg["encoders"]["pairs"][pair]


# ════════════════════════════════════════════════════════════════════════════
# Hashing / naming
# ════════════════════════════════════════════════════════════════════════════
def md5_short(text: str, n: int = 12) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def doc_id(text: str) -> str:
    """Stable content hash for a document (used as the dataset key)."""
    return md5_short(text, 16)


def sanitize(name: str) -> str:
    return name.replace("/", "_").replace(":", "-").replace(" ", "_")


def encoder_tag(role: str, ref: str) -> str:
    """Short, filesystem-safe encoder name. role in {comet, raw}."""
    base = ref.rstrip("/").split("/")[-1].replace(".ckpt", "")
    prefix = "comet" if role == "comet" else "xlmr"
    return sanitize(f"{prefix}-{base}")


def content_hash(doc_ids: Sequence[str], L: int, pooling: str, normalize: bool) -> str:
    """Cache key over a set of documents at a given length / pooling."""
    h = hashlib.md5()
    h.update(("|".join(doc_ids)).encode("utf-8"))
    h.update(f"|L{L}|{pooling}|norm{int(normalize)}".encode("utf-8"))
    return h.hexdigest()[:12]


# ════════════════════════════════════════════════════════════════════════════
# Misc
# ════════════════════════════════════════════════════════════════════════════
def pick_device(device_arg: str | None = None) -> str:
    if device_arg and device_arg != "auto":
        return device_arg
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for key in ("results_dir", "data_dir", "feature_cache", "analysis_dir", "figures_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def banner(msg: str) -> None:
    line = "═" * max(8, len(msg) + 2)
    print(f"\n{line}\n {msg}\n{line}")
