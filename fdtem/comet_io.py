"""Loading, discovering and scoring with COMET checkpoints.

Centralises three things that were previously copy-pasted across eval scripts:
finding the right checkpoint inside a training run, making a Lightning run
directory loadable by `comet.load_from_checkpoint`, and caching predictions.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TAU = re.compile(r"val_kendall=([0-9.]+)\.ckpt$")


def write_hparams(run_dir: Path) -> Optional[Path]:
    """Write `<run_dir>/hparams.yaml` from a checkpoint if Lightning did not.

    `comet.load_from_checkpoint` requires it, but the WandbLogger layout never
    creates one — without this every trained arm is unloadable.
    """
    import torch
    import yaml

    run_dir = Path(run_dir)
    out = run_dir / "hparams.yaml"
    if out.exists():
        return out
    ckpts = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        return None
    ck = torch.load(ckpts[0], map_location="cpu", weights_only=False, mmap=True)
    hp = {}
    for k, v in dict(ck["hyper_parameters"]).items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        try:
            yaml.safe_dump({k: v})
            hp[k] = v
        except Exception:  # noqa: BLE001 — keep unserialisable values as text
            hp[k] = str(v)
    hp.setdefault("class_identifier",
                  "unified_metric" if "input_segments" in hp else "regression_metric")
    out.write_text(yaml.safe_dump(hp, sort_keys=False))
    return out


def best_checkpoint(arm_dir: Path, prefer: str = "best") -> Optional[Path]:
    """Best (highest val_kendall) or last checkpoint of a training arm.

    `arm_dir` is the directory given to the trainer; the Lightning/W&B layout
    nests as <arm_dir>/<project>/<run-id>/checkpoints/.
    """
    arm_dir = Path(arm_dir)
    ckpts = list(arm_dir.glob("*/*/checkpoints/*.ckpt"))
    if not ckpts:
        return None
    if prefer == "last":
        last = [c for c in ckpts if c.name == "last.ckpt"]
        chosen = max(last, key=lambda p: p.stat().st_mtime) if last else None
        if chosen:
            write_hparams(chosen.parents[1])
            return chosen
    scored = [(float(m.group(1)), c) for c in ckpts if (m := _TAU.search(c.name))]
    if not scored:
        return best_checkpoint(arm_dir, prefer="last") if prefer != "last" else None
    chosen = max(scored)[1]
    write_hparams(chosen.parents[1])
    return chosen


def discover_arms(root: Path, prefix: str = "", label_fn=None) -> Dict[str, Path]:
    """Map arm name -> best checkpoint for every training arm under `root`."""
    out: Dict[str, Path] = {}
    for d in sorted(Path(root).glob(f"{prefix}*")):
        if not d.is_dir():
            continue
        ck = best_checkpoint(d)
        if ck:
            out[label_fn(d.name) if label_fn else d.name] = ck
    return out


def cache_path(label: str, df: pd.DataFrame, cache_dir: Path,
               columns: Optional[List[str]] = None) -> Path:
    """Where predictions for (model, exact input rows) live. One definition, so
    scoring and later analysis always agree on the key."""
    cols = [c for c in (columns or ["src", "mt", "ref"]) if c in df.columns]
    joined = df[cols[0]].astype(str)
    for c in cols[1:]:
        joined = joined + "|" + df[c].astype(str)
    h = hashlib.md5("\n".join(joined).encode("utf-8")).hexdigest()[:10]
    return Path(cache_dir) / f"{label}__{h}__n{len(df)}.npy"


def model_fingerprint(ref: str) -> str:
    """Short digest of what a model reference currently resolves to.

    Caches are keyed by label, and labels are stable across retrainings — so
    without this, re-evaluating `da-frac040` after retraining that arm silently
    returns the previous run's scores. For a local checkpoint the digest covers
    path + size + mtime, so it changes whenever the file does; Hub weights are
    immutable, so their id is enough.
    """
    path = Path(os.path.expanduser(str(ref)))
    if path.is_file():
        st = path.stat()
        key = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    else:
        key = f"hub:{ref}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:8]


def load_comet(ref: str):
    """Load a COMET model from a local checkpoint path or a Hub id."""
    from comet import download_model, load_from_checkpoint

    path = os.path.expanduser(str(ref))
    ckpt = path if os.path.isfile(path) else download_model(ref)
    model = load_from_checkpoint(os.path.expanduser(ckpt))
    # remembered so `score` can tell a stale cache entry from a fresh one
    model._fdtem_source = str(ref)
    model._fdtem_fingerprint = model_fingerprint(ref)
    return model


def uses_reference(model) -> bool:
    """Does this model actually read the `ref` field?

    CometKiwi is a UnifiedMetric with input_segments = [mt, src], so a
    class-name test misses it; the hyper-parameters are authoritative.
    """
    segments = getattr(model.hparams, "input_segments", None)
    if segments is not None:
        return "ref" in segments
    return "referenceless" not in type(model).__name__.lower()


def score(model, label: str, df: pd.DataFrame, cache_dir: Path,
          batch_size: int = 32, gpus: int = 1,
          columns: Optional[List[str]] = None) -> np.ndarray:
    """Score a dataframe, caching to disk keyed by (label, exact input texts).

    Reference-free models simply ignore a `ref` column they were not trained on.

    A cache entry is only reused when the sidecar `.fp` file records the same
    checkpoint the model was loaded from. Labels outlive the checkpoints behind
    them — retrain `da-frac040` and the label is unchanged — so reusing on the
    label alone would report the old run's scores for the new model. Entries
    with no sidecar are of unknown provenance and are rescored.
    """
    cols = [c for c in (columns or ["src", "mt", "ref"]) if c in df.columns]
    cache = cache_path(label, df, cache_dir, cols)
    fp = getattr(model, "_fdtem_fingerprint", None)
    fp_file = cache.with_suffix(cache.suffix + ".fp")
    if cache.exists():
        if fp is None:
            return np.load(cache)
        seen = fp_file.read_text().strip() if fp_file.exists() else None
        if seen == fp:
            return np.load(cache)
        logger.warning(
            "%s: cached predictions are from a different checkpoint (%s != %s)"
            " — rescoring", label, seen or "unrecorded", fp)
    data = df[cols].astype(str).to_dict("records")
    out = model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=True)
    scores = np.asarray(out["scores"], dtype=float)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, scores)
    if fp is not None:
        fp_file.write_text(fp)
    return scores
