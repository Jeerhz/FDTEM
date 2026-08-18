#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# slurm_agg_probe.sh — run Exp 0 (aggregation-function probe) on the cluster.
#
# Submit from the repo root:
#     sbatch qe_aggregation_probe/slurm_agg_probe.sh                    # full run
#     sbatch qe_aggregation_probe/slurm_agg_probe.sh --metrics cometkiwi
#     sbatch qe_aggregation_probe/slurm_agg_probe.sh --perturb_backend rule
#
# Tunables (env at submit time):
#   CONDA_ENV         conda env with torch+transformers+unbabel-comet (default comet-bio)
#   VENV_PATH         alternative: a venv to source
#   OPENAI_API_KEY    required for perturb.backend: llm
#   PERTURB_BASE_URL  optional OpenAI-compatible endpoint (vLLM, ...)
#   PERTURB_MODEL     perturbation model (default gpt-4o-mini)
#   CONFIG            config file (default config.yaml)
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=agg_probe
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

PROJ="${SLURM_SUBMIT_DIR:-$PWD}/qe_aggregation_probe"
[[ -f "$PROJ/run.py" ]] || PROJ="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJ"
mkdir -p logs

if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
else
  source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-comet-bio}"
fi

export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
mkdir -p "$HF_HOME"

CONFIG="${CONFIG:-config.yaml}"
EXTRA=("$@")

echo "═══════════════════════════════════════════════════════════"
echo " Node   : $(hostname)   GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo " Job    : ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?})"
echo " Config : $CONFIG   Extra: ${EXTRA[*]:-none}"
echo " LLM    : ${PERTURB_MODEL:-gpt-4o-mini} @ ${PERTURB_BASE_URL:-api.openai.com}"
echo "═══════════════════════════════════════════════════════════"
nvidia-smi || true

srun python run.py --config "$CONFIG" ${EXTRA[@]+"${EXTRA[@]}"}
echo "Done → $PROJ/results/"
