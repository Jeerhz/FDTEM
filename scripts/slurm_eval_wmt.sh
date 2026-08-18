#!/usr/bin/env bash
#SBATCH --job-name=eval-wmt
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
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate comet-bio
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
[[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]] && export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
MODELS="$(cat "$MODEL_FILE")"
echo "=== $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
srun python scripts/eval_length_correlation.py \
    --models da-base=Unbabel/wmt22-comet-da qe-base=Unbabel/wmt22-cometkiwi-da $MODELS \
    --data_dir "$HOME/scratch/wmt_eval_portion" \
    --batch_size 32 \
    --cache_dir results/retrain_wmt/pred_cache \
    --output results/retrain_wmt/length_correlation_portion.json \
    --run_name length_correlation_wmt
echo DONE
