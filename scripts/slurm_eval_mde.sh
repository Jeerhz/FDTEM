#!/usr/bin/env bash
#SBATCH --job-name=eval-mde
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
echo "=== $(hostname) ==="
srun python scripts/eval_metadoceval.py \
    --data_dir "$HOME/scratch/metadoceval-testset" \
    --models da-base=Unbabel/wmt22-comet-da qe-base=Unbabel/wmt22-cometkiwi-da $MODELS \
    --windows 1 6 --batch_size 32 \
    --cache_dir results/retrain_wmt/mde_cache \
    --output results/retrain_wmt/metadoceval_wmt.json \
    --run_name metadoceval_wmt
echo DONE
