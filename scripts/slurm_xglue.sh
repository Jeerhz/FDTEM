#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# slurm_xglue.sh — generic Slurm wrapper for the XGlue eval scripts on Cleps.
#
# It does NOT hard-code which script to run: you pass the full python command as
# arguments, so the same wrapper submits probe_xglue.py and finetune_xglue.py.
#
#   sbatch scripts/slurm_xglue.sh python scripts/finetune_xglue.py --encoder hf:xlm-roberta-large --task xnli ...
#
# Tunables (override at submit time, they win over the #SBATCH defaults):
#   sbatch --partition=gpu --time=24:00:00 -J ft_xnli scripts/slurm_xglue.sh <cmd>
#   PART=gpu_p1 ACCT=xxx sbatch scripts/slurm_xglue.sh <cmd>      # via env (see below)
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=xglue
#SBATCH --output=logs/%x-%j.out          # logs/<jobname>-<jobid>.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu                   # <-- CHECK with `sinfo -s` on Cleps
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
# #SBATCH --account=CHANGE_ME             # <-- uncomment & set if your alloc needs it

set -euo pipefail

# ── project root (this script lives in scripts/) ──────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs

# ── environment ───────────────────────────────────────────────────────────────
# Adjust to however your env is set up on Cleps (conda or venv).
# Example for conda:
#   source "$HOME/miniconda3/etc/profile.d/conda.sh"
#   conda activate fdtem
# Example for a module + venv:
#   module load cuda/12.1
#   source "$HOME/envs/fdtem/bin/activate"
if [[ -n "${CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi

# ── W&B ───────────────────────────────────────────────────────────────────────
export WANDB_PROJECT="${WANDB_PROJECT:-comet-xglue}"
# If W&B can't reach the network from compute nodes, set: export WANDB_MODE=offline

# ── caches on scratch (HF datasets / models can be large) ─────────────────────
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME"

echo "═══════════════════════════════════════════════════════════"
echo " Node     : $(hostname)"
echo " Job      : ${SLURM_JOB_NAME:-?}  (${SLURM_JOB_ID:-?})"
echo " GPU      : ${CUDA_VISIBLE_DEVICES:-none}"
echo " WANDB    : project=$WANDB_PROJECT mode=${WANDB_MODE:-online}"
echo " Command  : $*"
echo "═══════════════════════════════════════════════════════════"
nvidia-smi || true

# ── run whatever command was passed ───────────────────────────────────────────
srun "$@"
