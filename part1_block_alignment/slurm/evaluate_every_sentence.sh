#!/usr/bin/env bash
# Every sentence perturbed vs first sentence only, D = 1..6 (evaluate_every_sentence.py).
# Needs the dataset from build_every_sentence.sh. Writes results/every_sentence.json + plot.
#
#   sbatch part1_block_alignment/slurm/evaluate_every_sentence.sh
#   QE_CKPT=<kiwi arm ckpt> DA_CKPT=<da arm ckpt> sbatch part1_block_alignment/slurm/evaluate_every_sentence.sh
#
# Tunables: LANGS, K_LIST, D_LIST, BACKEND (dataset to read), BIO_CKPT (Bio-MQM COMET, cosine),
#           QE_CKPT (a reference-free arm, score), DA_CKPT (cosine), WANDB_PROJECT, BATCH_SIZE.
#SBATCH --job-name=every-sentence
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
K_LIST="${K_LIST:-1 2 3 4 5}"
D_LIST="${D_LIST:-1 2 3 4 5 6}"
BACKEND="${BACKEND:-spacy}"
WANDB_PROJECT="${WANDB_PROJECT:-comet-every-sentence}"
BIO_CKPT="${BIO_CKPT:-$FDTEM_SCRATCH/checkpoints/bio_mqm/comet-bio-mqm/4yeqp7cn/checkpoints/last.ckpt}"

# The metric (CometKiwi score) first, then the encoders read by cosine.
SCORERS=( "comet-score:Unbabel/wmt22-cometkiwi-da" "comet:Unbabel/wmt22-comet-da"
          "hf-mean:xlm-roberta-large" "labse" "e5" )
[[ -f "$BIO_CKPT" ]] && SCORERS+=( "comet:$BIO_CKPT" )
[[ -n "${QE_CKPT:-}" && -f "$QE_CKPT" ]] && SCORERS+=( "comet-score:$QE_CKPT" )
[[ -n "${DA_CKPT:-}" && -f "$DA_CKPT" ]] && SCORERS+=( "comet:$DA_CKPT" )
echo "scorers: ${SCORERS[*]}"

srun python -m part1_block_alignment.evaluate_every_sentence \
  --scorers "${SCORERS[@]}" --langs $LANGS --k_list $K_LIST --D_list $D_LIST \
  --backend "$BACKEND" --batch_size "${BATCH_SIZE:-32}" --wandb_project "$WANDB_PROJECT"
