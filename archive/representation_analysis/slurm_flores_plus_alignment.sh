#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# slurm_flores_plus_alignment.sh — ALIGNMENT-ONLY study on official FLORES+.
#
# Runs *only* the cross-lingual alignment task (xSIM retrieval, E1/E3) — no
# xSIM++ error-sensitivity — comparing the COMET encoder with its raw XLM-R
# backbone and dedicated aligners (LaBSE, multilingual-E5).
#
# Produces, in results/flores_plus/:
#   plots/e1_alignment_barplot.png   one bar per model = xSIM accuracy (k=1, en↔xx)
#   plots/e1_e3_length_scaling.png   alignment vs pseudo-doc length (k=1,2,4)
#   e1_e3_retrieval.json             all numbers
# and logs metrics to W&B (project $WANDB_PROJECT, run 'plus_devtest_xsim_alignment').
#
# Usage:
#   sbatch scripts/slurm_flores_plus_alignment.sh
#   LANGS="en de fr ru zh ja ar" sbatch scripts/slurm_flores_plus_alignment.sh
#
# FLORES+ is gated: be logged in to the HF Hub and have accepted the terms at
# https://huggingface.co/datasets/openlanguagedata/flores_plus.
#
# Tunables (env at submit time):
#   CONDA_ENV      conda env                (default comet-bio)
#   BIO_CKPT       Bio-MQM COMET .ckpt      (default: epoch-12 4yeqp7cn/last.ckpt)
#   LANGS          space-separated ISO langs(default: en de es fr ru zh)
#   SPLIT          FLORES+ split            (default: devtest)
#   WANDB_PROJECT  W&B project              (default: comet-flores-plus)
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=flores-align
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
mkdir -p logs

# ── environment ───────────────────────────────────────────────────────────────
CONDA_BASE="$HOME/miniconda3"
if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
else
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-comet-bio}"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-comet-flores-plus}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME"

# FLORES+ is gated; get_token() reads $HF_HOME/token, so when HF_HOME is redirected
# the default ~/.cache/huggingface/token is missed → 401. Export HF_TOKEN explicitly.
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

SPLIT="${SPLIT:-devtest}"
LANGS="${LANGS:-en de es fr ru zh}"
OUT=results/flores_plus
mkdir -p "$OUT"

# ── Bio-MQM checkpoint (≥10 epochs): default to the epoch-12 last.ckpt ──────────
DEFAULT_BIO="$HOME/scratch/checkpoints/bio_mqm/comet-bio-mqm/4yeqp7cn/checkpoints/last.ckpt"
BIO_CKPT="${BIO_CKPT:-$DEFAULT_BIO}"

ENCODERS=( "comet:Unbabel/wmt22-comet-da" "hf-mean:xlm-roberta-large" "labse" "e5" )
if [[ -f "$BIO_CKPT" ]]; then
  ENCODERS=( "comet:Unbabel/wmt22-comet-da" "comet:$BIO_CKPT" "hf-mean:xlm-roberta-large" "labse" "e5" )
else
  echo "WARNING: Bio-MQM checkpoint not found at $BIO_CKPT — running without COMET-bio."
fi

echo "═══════════════════════════════════════════════════════════"
echo " Node    : $(hostname)   GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo " Task    : ALIGNMENT only (xSIM, E1/E3)"
echo " FLORES+ : split=$SPLIT  langs=[$LANGS]"
echo " Bio ckpt: ${BIO_CKPT}$( [[ -f "$BIO_CKPT" ]] && echo '' || echo '  (MISSING)')"
echo " WANDB   : project=$WANDB_PROJECT mode=${WANDB_MODE:-online}"
echo "═══════════════════════════════════════════════════════════"
nvidia-smi || true

# ── sanity check — gated FLORES+ access ─────────────────────────────────────────
echo; echo "### Sanity check — official FLORES+ access ###"
srun python - "$SPLIT" <<'PY'
import sys
from datasets import load_dataset
split = sys.argv[1]
ds = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split=split)
print(f"FLORES+ OK: eng_Latn/{split} = {len(ds)} rows")
PY

# ── E1 + E3 — cross-lingual alignment (xSIM) ───────────────────────────────────
echo; echo "### E1 + E3 — xSIM alignment on FLORES+ (bar plot + length scaling) ###"
srun python scripts/run_retrieval.py \
  --flores_source plus --split "$SPLIT" \
  --encoders "${ENCODERS[@]}" --langs $LANGS --lengths 1 2 4 \
  --output "$OUT/e1_e3_retrieval.json" --wandb_project "$WANDB_PROJECT"

echo; echo "Alignment study done → $OUT/  (bar plot: $OUT/plots/e1_alignment_barplot.png)"
