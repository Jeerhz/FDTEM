"""Part 1: FLORES+ blocks, xSIM++ hard negatives, encoder cosine vs COMET score."""
from pathlib import Path

PART_DIR = Path(__file__).resolve().parent
DATA_DIR = PART_DIR / "data"          # gitignored: corpora, blocks, pools, emb_cache
RESULTS_DIR = PART_DIR / "results"
FIGURES_DIR = PART_DIR / "figures"
