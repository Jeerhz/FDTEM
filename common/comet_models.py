"""Loading, finding and scoring COMET checkpoints (Hub ids or local .ckpt files).

Three things every evaluation needs: the right checkpoint inside a training run,
a run directory that `comet.load_from_checkpoint` accepts, and predictions cached
per (model, exact input rows) so re-running an evaluation only scores what changed.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TAU = re.compile(r"val_kendall=([0-9.]+)\.ckpt$")


def resolve_checkpoint(ref: str) -> str:
    """A local .ckpt path as is, or the checkpoint of a Hub model id (downloaded once)."""
    path = os.path.expanduser(str(ref))
    if os.path.isfile(path):
        return path
    from comet import download_model
    return os.path.expanduser(download_model(ref))


def write_hparams(run_dir: Path) -> Path | None:
    """Write `<run_dir>/hparams.yaml` from a checkpoint when Lightning did not.

    `comet.load_from_checkpoint` requires it, but the WandbLogger layout never
    creates one, so a trained arm would otherwise be unloadable.
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


def write_missing_hparams(ckpt_root: Path) -> list[Path]:
    """Repair every run under `<ckpt_root>/<project>/<run-id>/checkpoints/`."""
    written = []
    for ck in sorted(Path(ckpt_root).glob("*/*/checkpoints/last.ckpt")):
        out = write_hparams(ck.parents[1])
        if out:
            written.append(out)
    return written


def best_checkpoint(arm_dir: Path, prefer: str = "best") -> Path | None:
    """Best (highest val_kendall) or last checkpoint of a training arm.

    `arm_dir` is the directory given to the trainer; the Lightning/W&B layout
    nests as <arm_dir>/<project>/<run-id>/checkpoints/.
    """
    ckpts = list(Path(arm_dir).glob("*/*/checkpoints/*.ckpt"))
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


def model_fingerprint(ref: str) -> str:
    """Short digest of what a model reference resolves to right now.

    Caches are keyed by label and labels outlive the checkpoints behind them, so
    without this a retrained arm would silently replay the previous run's scores.
    """
    path = Path(os.path.expanduser(str(ref)))
    if path.is_file():
        st = path.stat()
        key = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    else:
        key = f"hub:{ref}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:8]


def load_comet(ref: str):
    """Load a COMET model from a local checkpoint or a Hub id, repairing hparams.yaml if needed."""
    from comet import load_from_checkpoint

    ckpt = resolve_checkpoint(ref)
    run_dir = Path(ckpt).parent.parent
    if (run_dir / "checkpoints").is_dir() and not (run_dir / "hparams.yaml").exists():
        if write_hparams(run_dir):
            logger.info("wrote missing %s", run_dir / "hparams.yaml")
    model = load_from_checkpoint(ckpt)
    model._fdtem_source = str(ref)
    model._fdtem_fingerprint = model_fingerprint(ref)
    return model


def uses_reference(model) -> bool:
    """Does this model read the `ref` field? (CometKiwi is a UnifiedMetric with [mt, src].)"""
    segments = getattr(model.hparams, "input_segments", None)
    if segments is not None:
        return "ref" in segments
    return "referenceless" not in type(model).__name__.lower()


def cache_path(label: str, df: pd.DataFrame, cache_dir: Path,
               columns: list[str] | None = None) -> Path:
    """Where predictions for (model label, exact input rows) live."""
    cols = [c for c in (columns or ["src", "mt", "ref"]) if c in df.columns]
    joined = df[cols[0]].astype(str)
    for c in cols[1:]:
        joined = joined + "|" + df[c].astype(str)
    h = hashlib.md5("\n".join(joined).encode("utf-8")).hexdigest()[:10]
    return Path(cache_dir) / f"{label}__{h}__n{len(df)}.npy"


def score(model, label: str, df: pd.DataFrame, cache_dir: Path,
          batch_size: int = 32, gpus: int = 1, columns: list[str] | None = None) -> np.ndarray:
    """Score a dataframe, caching to disk keyed by (label, exact input rows).

    A cache entry is reused only when its `.fp` sidecar records the checkpoint the
    model was loaded from; entries of unknown provenance are rescored.
    """
    cols = [c for c in (columns or ["src", "mt", "ref"]) if c in df.columns]
    cache = cache_path(label, df, cache_dir, cols)
    fp = getattr(model, "_fdtem_fingerprint", None)
    fp_file = cache.with_suffix(cache.suffix + ".fp")
    if cache.exists():
        seen = fp_file.read_text().strip() if fp_file.exists() else None
        if fp is None or seen == fp:
            return np.load(cache)
        logger.warning("%s: cached predictions are from another checkpoint (%s != %s), rescoring",
                       label, seen or "unrecorded", fp)
    data = df[cols].astype(str).to_dict("records")
    out = model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=True)
    scores = np.asarray(out["scores"], dtype=float)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, scores)
    if fp is not None:
        fp_file.write_text(fp)
    return scores
