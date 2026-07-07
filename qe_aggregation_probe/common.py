#!/usr/bin/env python3
"""
common.py — shared building blocks for qe_aggregation_probe (Exp 0).

Config loading (yaml + ${ENV} expansion), deterministic hashing, condition
constants, and small helpers. No heavy deps here (no torch / transformers /
datasets), so this module imports cleanly in the logic-only smoke environment.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

# ════════════════════════════════════════════════════════════════════════════
# Experimental conditions
# ════════════════════════════════════════════════════════════════════════════
SEVERITIES = ["minor", "major"]
FAMILIES = ["accuracy", "fluency"]

# condition id "none" is reserved for the unperturbed (m=0) paragraph
COND_NONE = "none"


def cond_id(severity: str, family: str) -> str:
    return f"{severity}_{family}"


def all_conditions(cfg: Dict[str, Any]) -> List[str]:
    return [cond_id(s, f) for s in cfg["design"]["severities"]
            for f in cfg["design"]["families"]]


# Languages written without spaces — unit joining and rule-based perturbations
# operate on characters instead of whitespace tokens for these.
NO_SPACE_LANGS = {"zh", "ja", "th"}


def iso_of_lp(lp: str) -> str:
    """'en-de_DE' → 'de';  'en-zh_CN' → 'zh';  plain 'de' passes through."""
    if "-" in lp:
        tgt = lp.split("-", 1)[1]
        return tgt.split("_", 1)[0]
    return lp


def joiner_for(lang_iso: str) -> str:
    return "" if lang_iso in NO_SPACE_LANGS else " "


# ════════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════════
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} and ${VAR:-default} in strings."""
    if isinstance(value, str):
        def repl(m: "re.Match") -> str:
            var, default = m.group(1), m.group(2)
            return os.environ.get(var, default if default is not None else "")
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return _expand_env(cfg)


# ════════════════════════════════════════════════════════════════════════════
# Hashing / naming / misc
# ════════════════════════════════════════════════════════════════════════════
def md5_short(text: str, n: int = 16) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def sanitize(name: str) -> str:
    return name.replace("/", "_").replace(":", "-").replace(" ", "_")


def pick_device(device_arg: str | None = None) -> str:
    if device_arg and device_arg != "auto":
        return device_arg
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for key in ("results_dir", "data_dir", "perturb_cache", "score_cache",
                "analysis_dir", "figures_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def banner(msg: str) -> None:
    line = "═" * max(8, len(msg) + 2)
    print(f"\n{line}\n {msg}\n{line}")
