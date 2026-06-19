#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# slurm_repana.sh — run the representation-analysis suite (E1–E5) on Cleps.
#
# Two modes:
#   (1) Full suite, no args — runs E1/E3, E2, E4, E5 over the default encoder zoo
#       (COMET, COMET-bio, XLM-R-large, LaBSE, E5) and core languages:
#
#         sbatch scripts/slurm_repana.sh
#
#   (2) Pass-through — give it a full python command and it just runs that
#       (same convention as slurm_xglue.sh), e.g. to run a single experiment:
#
#         sbatch scripts/slurm_repana.sh python scripts/run_e2_error_sensitivity.py \
#             --encoders comet:Unbabel/wmt22-comet-da labse --langs en de fr
#
# Tunables (env at submit time):
#   CONDA_ENV   conda env (default comet-bio)
#   BIO_CKPT    path to the bio-finetuned COMET .ckpt (default: auto-discover)
#   LANGS       space-separated ISO langs (default: en de es fr ru zh)
#   WANDB_PROJECT  (default comet-repana; set WANDB_MODE=offline if no network)
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=repana
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

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

export WANDB_PROJECT="${WANDB_PROJECT:-comet-repana}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME"

# FLORES-200 is read from raw text files (the HF dataset scripts are broken on
# Cleps). Auto-discovered from the HF cache; override with FLORES_DIR=/path.
export FLORES_DIR="${FLORES_DIR:-}"

echo "═══════════════════════════════════════════════════════════"
echo " Node   : $(hostname)   GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo " Job    : ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?})"
echo " WANDB  : project=$WANDB_PROJECT mode=${WANDB_MODE:-online}"
echo "═══════════════════════════════════════════════════════════"
nvidia-smi || true

# ── pass-through mode ─────────────────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
  echo " Command: $*"
  srun "$@"
  exit $?
fi

# ── full-suite mode ───────────────────────────────────────────────────────────
# Auto-discover the bio-finetuned checkpoint unless BIO_CKPT is set.
if [[ -z "${BIO_CKPT:-}" ]]; then
  BIO_CKPT=$(find "$HOME/scratch/checkpoints/bio_mqm" -name '*val_kendall*.ckpt' 2>/dev/null | sort | tail -1 || true)
fi

ENCODERS=( "comet:Unbabel/wmt22-comet-da" "hf-mean:xlm-roberta-large" "labse" "e5" )
if [[ -n "${BIO_CKPT:-}" && -f "$BIO_CKPT" ]]; then
  ENCODERS=( "comet:Unbabel/wmt22-comet-da" "comet:$BIO_CKPT" "hf-mean:xlm-roberta-large" "labse" "e5" )
  echo " Bio ckpt: $BIO_CKPT"
else
  echo " Bio ckpt: NOT FOUND — running without COMET-bio (set BIO_CKPT=... to include it)"
fi

LANGS="${LANGS:-en de es fr ru zh}"
OUT=results/repana
mkdir -p "$OUT"

echo; echo "### E1 + E3 — alignment & length scaling ###"
srun python scripts/run_e1_e3_retrieval.py \
  --encoders "${ENCODERS[@]}" --langs $LANGS --lengths 1 2 4 8 \
  --output "$OUT/e1_e3_retrieval.json" --wandb_project "$WANDB_PROJECT"

echo; echo "### E2 — error-sensitivity trade-off ###"
srun python scripts/run_e2_error_sensitivity.py \
  --encoders "${ENCODERS[@]}" --langs $LANGS \
  --output "$OUT/e2_error_sensitivity.json" --wandb_project "$WANDB_PROJECT"

echo; echo "### E4 — geometry ###"
srun python scripts/run_e4_geometry.py \
  --encoders "${ENCODERS[@]}" --langs $LANGS \
  --output "$OUT/e4_geometry.json" --wandb_project "$WANDB_PROJECT"

echo; echo "### E5 — layer-wise CKA ###"
srun python scripts/run_e5_cka.py \
  --encoders "${ENCODERS[@]}" --reference hf-mean:xlm-roberta-large \
  --langs en de zh --n_per_lang 400 \
  --output "$OUT/e5_cka.json" --wandb_project "$WANDB_PROJECT"

echo; echo "All experiments done → $OUT/"
