#!/usr/bin/env bash
# Encoder cosine on the candidate pools (evaluate_encoders.py), full encoder zoo.
# Needs the pools from build_pools.sh. Writes results/encoder_cosine.json + plots.
#
#   sbatch part1_block_alignment/slurm/evaluate_encoders.sh
#   LANGS="de es fr ru zh" K_LIST="2 3 4 5" sbatch part1_block_alignment/slurm/evaluate_encoders.sh
#
# Tunables: LANGS, K_LIST, BACKEND (pools to read), BIO_CKPT (Bio-MQM COMET .ckpt),
#           RETRAIN_CKPT (a length-trained arm to add), WANDB_PROJECT, BATCH_SIZE.
#SBATCH --job-name=block-xsim
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
source common/cluster_env.sh

LANGS="${LANGS:-de es fr ru}"
K_LIST="${K_LIST:-2 3 4 5}"
BACKEND="${BACKEND:-spacy}"
WANDB_PROJECT="${WANDB_PROJECT:-comet-block-xsim}"
BIO_CKPT="${BIO_CKPT:-$FDTEM_SCRATCH/checkpoints/bio_mqm/comet-bio-mqm/4yeqp7cn/checkpoints/last.ckpt}"

ENCODERS=( "comet:Unbabel/wmt22-comet-da" "hf-mean:xlm-roberta-large" "labse" "e5" )
[[ -f "$BIO_CKPT" ]] && ENCODERS+=( "comet:$BIO_CKPT" )
[[ -n "${RETRAIN_CKPT:-}" && -f "$RETRAIN_CKPT" ]] && ENCODERS+=( "comet:$RETRAIN_CKPT" )
echo "encoders: ${ENCODERS[*]}"

srun python -m part1_block_alignment.evaluate_encoders \
  --encoders "${ENCODERS[@]}" --langs $LANGS --k_list $K_LIST --backend "$BACKEND" \
  --batch_size "${BATCH_SIZE:-32}" --wandb_project "$WANDB_PROJECT"
