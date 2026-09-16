"""Repository and cluster paths shared by every part."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get("FDTEM_SCRATCH", "~/scratch")).expanduser()
CHECKPOINTS = SCRATCH / "checkpoints"
