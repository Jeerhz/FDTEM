"""Hugging Face and Weights & Biases credentials, resolved in one place.

Both services are logged into once per machine (`hf auth login`, `wandb login`).
Cluster jobs redirect HF_HOME to scratch, which hides the default token file, and
may run where no W&B login exists: this module bridges both cases.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from common.paths import SCRATCH

logger = logging.getLogger(__name__)


def configure_hf_cache() -> None:
    """Point the HF caches at scratch unless the caller already chose."""
    os.environ.setdefault("HF_HOME", str(SCRATCH / "hf_cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))


def hf_token() -> str | None:
    """Resolve the HF token and export it as HF_TOKEN so `datasets` sees it."""
    configure_hf_cache()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:  # noqa: BLE001
            token = None
    if not token:
        default = Path.home() / ".cache" / "huggingface" / "token"
        if default.is_file():
            token = default.read_text(encoding="utf-8").strip() or None
    if token:
        os.environ.setdefault("HF_TOKEN", token)
    return token


def wandb_logged_in() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path.home() / ".netrc"
    return netrc.is_file() and "api.wandb.ai" in netrc.read_text(encoding="utf-8")


def init_wandb(project: str | None, name: str, job_type: str,
               tags: list[str] | None = None, config: dict | None = None):
    """`wandb.init`, or None when no project is given or W&B is unreachable."""
    if not project:
        return None
    if not wandb_logged_in():
        os.environ.setdefault("WANDB_MODE", "offline")
    try:
        import wandb
        return wandb.init(project=project, name=name, job_type=job_type, tags=tags, config=config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("W&B init failed: %s", exc)
        return None


def wandb_logger(project: str, name: str, save_dir: Path, tags: list[str] | None = None,
                 run_id: str | None = None):
    """Lightning WandbLogger; `run_id` continues an existing run (chained jobs)."""
    from pytorch_lightning.loggers import WandbLogger

    if not wandb_logged_in():
        os.environ.setdefault("WANDB_MODE", "offline")
    return WandbLogger(project=project, name=name, save_dir=str(save_dir), id=run_id,
                       resume="allow" if run_id else None, log_model=False, tags=tags or None)
