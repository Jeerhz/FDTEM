#!/usr/bin/env bash
# Fixed-size duel, gold vs D own negatives (evaluate_duel.py), full encoder zoo.
# Needs the pools from build_pools.sh. Writes results/duel.json + plots.
#
#   sbatch part1_block_alignment/slurm/evaluate_duel.sh
#   DUEL_SIZES="6 5 4" K_LIST="2 3 4 5" sbatch part1_block_alignment/slurm/evaluate_duel.sh
#
# Tunables: LANGS, K_LIST, BACKEND (pools to read), DUEL_SIZES, BIO_CKPT, WANDB_PROJECT, BATCH_SIZE.
#SBATCH --job-name=block-duel
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
source common/cluster_env.sh

LANGS="${LANGS:-de es fr ru}"
K_LIST="${K_LIST:-2 3 4 5}"
BACKEND="${BACKEND:-spacy}"
DUEL_SIZES="${DUEL_SIZES:-6 5 4}"
WANDB_PROJECT="${WANDB_PROJECT:-comet-block-duel}"
BIO_CKPT="${BIO_CKPT:-$FDTEM_SCRATCH/checkpoints/bio_mqm/comet-bio-mqm/4yeqp7cn/checkpoints/last.ckpt}"

ENCODERS=( "comet:Unbabel/wmt22-comet-da" "hf-mean:xlm-roberta-large" "labse" "e5" )
[[ -f "$BIO_CKPT" ]] && ENCODERS+=( "comet:$BIO_CKPT" )
echo "encoders: ${ENCODERS[*]}"

srun python -m part1_block_alignment.evaluate_duel \
  --encoders "${ENCODERS[@]}" --langs $LANGS --k_list $K_LIST --backend "$BACKEND" \
  --duel_sizes $DUEL_SIZES --batch_size "${BATCH_SIZE:-32}" --wandb_project "$WANDB_PROJECT"
