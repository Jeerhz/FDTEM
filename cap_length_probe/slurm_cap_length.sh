#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# slurm_cap_length.sh — run the CAP length-probe on Cleps (GPU).
#
# Submit from the repo root:
#     sbatch cap_length_probe/slurm_cap_length.sh                 # full run, default pair
#     sbatch cap_length_probe/slurm_cap_length.sh --pair alt      # InfoXLM pair
#     sbatch cap_length_probe/slurm_cap_length.sh --smoke         # 1 lang, ~200 docs, 2 buckets
#
# Tunables (env at submit time):
#   CONDA_ENV   conda env with torch+transformers+unbabel-comet+sklearn (default comet-bio)
#   VENV_PATH   alternative: a venv to `source $VENV_PATH/bin/activate`
#   BTL_DIR     dir with the Beyond Token Limits CSVs (train/ test/ or pooled csv)
#   CONFIG      config file (default config.yaml)
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=cap_len
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00

set -euo pipefail

PROJ="${SLURM_SUBMIT_DIR:-$PWD}/cap_length_probe"
[[ -f "$PROJ/run.py" ]] || PROJ="${SLURM_SUBMIT_DIR:-$PWD}"   # also works if submitted from inside
cd "$PROJ"
mkdir -p logs

# ── environment ───────────────────────────────────────────────────────────────
if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
else
  source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-comet-bio}"
fi

export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export BTL_DIR="${BTL_DIR:-$HOME/scratch/cap_btl}"
mkdir -p "$HF_HOME"

CONFIG="${CONFIG:-config.yaml}"
# all remaining args (e.g. --pair alt, --smoke) forward straight to run.py
EXTRA=("$@")

echo "═══════════════════════════════════════════════════════════"
echo " Node    : $(hostname)   GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo " Job     : ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?})"
echo " Proj    : $PROJ"
echo " BTL_DIR : $BTL_DIR  $( [[ -d "$BTL_DIR" ]] && echo '(found)' || echo '(MISSING → data.py will fall back to synthetic!)')"
echo " Config  : $CONFIG   Extra: ${EXTRA[*]:-none}"
echo "═══════════════════════════════════════════════════════════"
nvidia-smi || true

srun python run.py --config "$CONFIG" ${EXTRA[@]+"${EXTRA[@]}"}
echo "Done → $PROJ/results/"
