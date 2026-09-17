#!/usr/bin/env bash
# Alignment by COMET score vs by encoder cosine (evaluate_comet_score.py).
# Needs the pools from build_pools.sh (k=1 included). A dry-run pass first prints
# how many COMET forward passes the job will cost. Writes results/comet_score.json + plot.
#
#   sbatch part1_block_alignment/slurm/evaluate_comet_score.sh
#   QE_CKPT=<kiwi arm ckpt> DA_CKPT=<da arm ckpt> sbatch part1_block_alignment/slurm/evaluate_comet_score.sh
#
# Tunables: LANGS, K_LIST, BACKEND (pools to read), NEGATIVES (per block),
#           SHORTLIST (candidates per decision), QE_CKPT (added on both rules),
#           DA_CKPT (encoder-cos only: no reference-free mode), WANDB_PROJECT, BATCH_SIZE.
#SBATCH --job-name=comet-align
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
source common/cluster_env.sh

LANGS="${LANGS:-de es fr ru}"
K_LIST="${K_LIST:-1 2 3 4 5}"
BACKEND="${BACKEND:-spacy}"
NEGATIVES="${NEGATIVES:-6}"
SHORTLIST="${SHORTLIST:-3}"
WANDB_PROJECT="${WANDB_PROJECT:-comet-align}"

# The published pair on both rules is the minimum comparison: CometKiwi can do
# both; COMET-DA (reference-based) only the cosine.
SCORERS=( "comet-score:Unbabel/wmt22-cometkiwi-da"
          "encoder-cos:Unbabel/wmt22-cometkiwi-da"
          "encoder-cos:Unbabel/wmt22-comet-da" )
[[ -n "${QE_CKPT:-}" && -f "$QE_CKPT" ]] && SCORERS+=( "comet-score:$QE_CKPT" "encoder-cos:$QE_CKPT" )
[[ -n "${DA_CKPT:-}" && -f "$DA_CKPT" ]] && SCORERS+=( "encoder-cos:$DA_CKPT" )
echo "scorers: ${SCORERS[*]}"

srun python -m part1_block_alignment.evaluate_comet_score --dry_run \
  --langs $LANGS --k_list $K_LIST --backend "$BACKEND" \
  --negatives_per_block "$NEGATIVES" --shortlist_size "$SHORTLIST"

srun python -m part1_block_alignment.evaluate_comet_score \
  --scorers "${SCORERS[@]}" --langs $LANGS --k_list $K_LIST --backend "$BACKEND" \
  --negatives_per_block "$NEGATIVES" --shortlist_size "$SHORTLIST" \
  --batch_size "${BATCH_SIZE:-32}" --wandb_project "$WANDB_PROJECT"
