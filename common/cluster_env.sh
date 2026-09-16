# Sourced by every slurm script right after its #SBATCH header. Sets the
# environment only: repo root as cwd, python env, HF caches + token, W&B mode.
#
#   source common/cluster_env.sh
#
# Tunables: VENV_PATH (else conda), CONDA_BASE (~/miniconda3), CONDA_ENV (comet-bio),
#           FDTEM_SCRATCH (~/scratch), HF_HOME, HF_DATASETS_CACHE, HF_TOKEN.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
else
  source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-comet-bio}"
fi
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export FDTEM_SCRATCH="${FDTEM_SCRATCH:-$HOME/scratch}"
export HF_HOME="${HF_HOME:-$FDTEM_SCRATCH/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME"
# HF_HOME is redirected, so the token written by `hf auth login` must be re-exported.
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

if ! wandb status &>/dev/null; then
  echo "W&B not configured — WANDB_MODE=offline"
  export WANDB_MODE=offline
fi
