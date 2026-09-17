"""Clone (depth 1) or pull the MetaDocEval contrastive test set (Dahan, Bawden & Yvon, EAMT 2026).

  python -m part2_length_training.load_metadoceval [--data_dir ~/scratch/metadoceval-testset]
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common.paths import SCRATCH

REPO = "https://github.com/nicolasdahan/metadoceval-testset.git"
DEFAULT_DIR = SCRATCH / "metadoceval-testset"


def ensure_testset(data_dir: Path = DEFAULT_DIR) -> Path:
    data_dir = Path(data_dir).expanduser()
    if (data_dir / "data" / "en_fr_aya.json").exists():
        subprocess.run(["git", "-C", str(data_dir), "pull", "--ff-only"], check=False)
    else:
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(data_dir)], check=True)
    return data_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(DEFAULT_DIR))
    args = ap.parse_args()
    print(f"test set at {ensure_testset(Path(args.data_dir))}")


if __name__ == "__main__":
    main()
