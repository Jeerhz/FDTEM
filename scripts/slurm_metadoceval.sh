#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# slurm_metadoceval.sh — document-level meta-evaluation of COMET checkpoints on
# the MetaDocEval contrastive test set (Dahan, Bawden & Yvon, EAMT 2026).
#
# Clones the test set if missing, then runs scripts/eval_metadoceval.py over a
# set of models. Scoring is deduplicated and cached, so re-running with an extra
# checkpoint only scores the new one.
#
# Usage:
#   sbatch scripts/slurm_metadoceval.sh                       # baselines only
#   MODELS="wmt22=Unbabel/wmt22-comet-da longonly=$HOME/scratch/checkpoints/retrain/mix-frac000/<run>/checkpoints/last.ckpt" \
#       sbatch scripts/slurm_metadoceval.sh
#   # sweep every sentence-fraction arm that has a checkpoint:
#   SWEEP=1 sbatch scripts/slurm_metadoceval.sh
#
# Tunables (env at submit time):
#   MODELS      space-separated label=<hf-id-or-ckpt> entries (default: baselines)
#   SWEEP       1 = auto-discover mix-frac*/ checkpoints and append them
#   DATA_DIR    test-set clone            (default ~/scratch/metadoceval-testset)
#   WINDOWS     window sizes              (default "1 3 6 9")
#   CONDA_ENV   conda env                 (default comet-bio)
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=metadoceval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

CONDA_BASE="$HOME/miniconda3"
if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
else
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-comet-bio}"
fi

export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
mkdir -p "$HF_HOME"
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

DATA_DIR="${DATA_DIR:-$HOME/scratch/metadoceval-testset}"
WINDOWS="${WINDOWS:-1 3 6 9}"

# ── test set ──────────────────────────────────────────────────────────────────
if [[ ! -f "$DATA_DIR/data/en_fr_aya.json" ]]; then
  echo "### Cloning MetaDocEval test set → $DATA_DIR ###"
  git clone --depth 1 https://github.com/nicolasdahan/metadoceval-testset.git "$DATA_DIR"
else
  echo "Test set present: $DATA_DIR"
fi

# ── models ────────────────────────────────────────────────────────────────────
# cometkiwi is reference-free; eval_metadoceval.py detects that and drops `ref`
DEFAULT_MODELS="wmt22=Unbabel/wmt22-comet-da kiwi=Unbabel/wmt22-cometkiwi-da"
MODEL_ARGS="${MODELS:-$DEFAULT_MODELS}"

if [[ "${SWEEP:-0}" == "1" ]]; then
  SWEEP_ROOT="${SWEEP_ROOT:-$HOME/scratch/checkpoints/retrain}"
  for d in "$SWEEP_ROOT"/mix-*/ "$SWEEP_ROOT"/kiwi-mix-*/; do
    [[ -d "$d" ]] || continue
    arm=$(basename "$d")
    # Lightning nests as $CKPT_DIR/<wandb-project>/<run-id>/checkpoints/last.ckpt,
    # so search rather than glob a fixed depth; newest wins if a run was resumed
    ckpt=$(find "$d" -path '*/checkpoints/last.ckpt' -printf '%T@ %p\n' 2>/dev/null \
             | sort -rn | head -1 | cut -d' ' -f2- || true)
    if [[ -n "$ckpt" ]]; then
      MODEL_ARGS="$MODEL_ARGS ${arm}=${ckpt}"
      echo "  + $arm -> $ckpt"
    else
      echo "  ! $arm has no last.ckpt yet — skipping"
    fi
  done
fi

echo "═══════════════════════════════════════════════════════════"
echo " Node    : $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo " Data    : $DATA_DIR"
echo " Windows : $WINDOWS"
echo " Models  : $MODEL_ARGS"
echo "═══════════════════════════════════════════════════════════"
nvidia-smi || true

srun python scripts/eval_metadoceval.py \
    --data_dir "$DATA_DIR" \
    --models $MODEL_ARGS \
    --windows $WINDOWS \
    --output results/metadoceval/accuracy.json \
    --wandb_project "${WANDB_PROJECT:-comet-retrain}"

echo; echo "Done. Results in results/metadoceval/ (accuracy.json + plots/)."
