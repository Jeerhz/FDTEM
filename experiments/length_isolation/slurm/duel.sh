#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# slurm_block_duel.sh — the fixed-pool duel variant of block-level xSIM++.
#
# Every retrieval is a duel with a FIXED number of candidates at every k: the
# gold block + D single-error hard negatives of that same block. D = 6 is the
# k=2 budget (2 positions × 3 categories, one variant per pair); D = 5 and 4
# trade pool size for k=2 coverage (requiring all 6 edits keeps only blocks
# whose both sentences host all three error types — ~9% of de blocks at k=2).
# The duel is averaged over ALL C(m,D) subsets of the block's m available
# negatives — computed in closed form, C(w,D)/C(m,D). Blocks with m < D are
# skipped, so pool size is constant and length is the only variable. All duel
# sizes are scored from the same embeddings in one pass.
#
# Produces, in results/block_duel/:
#   plots/duel_err_vs_length.png   duel error vs k, one panel per D
#   plots/duel_coverage.png        fraction of blocks with ≥ D negatives
#   plots/beat_by_category.png     P[gold beats the negative] per category
#   block_duel.json                all numbers
# and logs to W&B (project $WANDB_PROJECT).
#
# Usage:
#   sbatch experiments/length_isolation/slurm/duel.sh
#   LANGS="de es fr ru" K_LIST="2 3 4 5" sbatch experiments/length_isolation/slurm/duel.sh
#
# Tunables (env at submit time):
#   CONDA_ENV      conda env                (default comet-bio)
#   BIO_CKPT       Bio-MQM COMET .ckpt      (default: epoch-12 4yeqp7cn/last.ckpt)
#   LANGS          non-pivot ISO langs      (default: de es fr ru; cased only)
#   K_LIST         block lengths            (default: 2 3 4 5)
#   SPLITS         FLORES+ splits to pool   (default: dev devtest)
#   DIRECTION      en2xx | xx2en            (default: en2xx)
#   DUEL_SIZES     negatives per duel       (default "6 5 4")
#   WANDB_PROJECT  W&B project              (default comet-block-duel)
#
# FLORES+ is gated: be logged in to the HF Hub and have accepted the terms at
# https://huggingface.co/datasets/openlanguagedata/flores_plus.
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=block-duel
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# ── environment ───────────────────────────────────────────────────────────────
CONDA_BASE="$HOME/miniconda3"
if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
else
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-comet-bio}"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-comet-block-duel}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME"
# gated FLORES+ auth under redirected HF_HOME (token lives in the default location)
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

LANGS="${LANGS:-de es fr ru}"
K_LIST="${K_LIST:-2 3 4 5}"
SPLITS="${SPLITS:-dev devtest}"
DIRECTION="${DIRECTION:-en2xx}"
DUEL_SIZES="${DUEL_SIZES:-6 5 4}"
OUT=results/block_duel
mkdir -p "$OUT"

# ── encoder zoo ────────────────────────────────────────────────────────────────
DEFAULT_BIO="$HOME/scratch/checkpoints/bio_mqm/comet-bio-mqm/4yeqp7cn/checkpoints/last.ckpt"
BIO_CKPT="${BIO_CKPT:-$DEFAULT_BIO}"
ENCODERS=( "comet:Unbabel/wmt22-comet-da" "hf-mean:xlm-roberta-large" "labse" "e5" )
[[ -f "$BIO_CKPT" ]] && ENCODERS+=( "comet:$BIO_CKPT" )

echo "═══════════════════════════════════════════════════════════"
echo " Node      : $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo " Task      : block duel  sizes=[$DUEL_SIZES]  direction=$DIRECTION"
echo " FLORES+   : splits=[$SPLITS]  langs=[$LANGS]  k=[$K_LIST]"
echo " Encoders  : ${ENCODERS[*]}"
echo " W&B       : project=$WANDB_PROJECT mode=${WANDB_MODE:-online}"
echo "═══════════════════════════════════════════════════════════"
nvidia-smi || true

# ── sanity check — gated FLORES+ access ─────────────────────────────────────────
echo; echo "### Sanity check — FLORES+ access ###"
srun python - <<'PY'
from datasets import load_dataset
ds = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
print(f"FLORES+ OK: eng_Latn/devtest = {len(ds)} rows")
PY

# ── run ────────────────────────────────────────────────────────────────────────
echo; echo "### Block duel (sizes: $DUEL_SIZES) ###"
srun python experiments/length_isolation/run_duel.py \
  --encoders "${ENCODERS[@]}" \
  --langs $LANGS --k_list $K_LIST --splits $SPLITS \
  --direction "$DIRECTION" --duel_sizes $DUEL_SIZES \
  --flores_source plus \
  --output "$OUT/block_duel.json" --wandb_project "$WANDB_PROJECT"

echo; echo "Done → $OUT/  (headline plot: $OUT/plots/duel_err_vs_length.png)"
