#!/usr/bin/env bash
#SBATCH --job-name=comet-bio-mqm-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Base vs fine-tuned COMET on the Bio-MQM validation split (predictions cached).
#   sbatch part3_biomed_finetune/slurm/evaluate.sh
# Tunables: BIO_CKPT (auto = best checkpoint under CKPT_DIR | <ckpt>)  CKPT_DIR  DATA_DIR
set -euo pipefail
source common/cluster_env.sh

BIO_CKPT="${BIO_CKPT:-auto}"
CKPT_DIR="${CKPT_DIR:-$FDTEM_SCRATCH/checkpoints/bio_mqm}"
DATA_DIR="${DATA_DIR:-$FDTEM_SCRATCH/bio_mqm}"

srun python -m part3_biomed_finetune.evaluate --bio "$BIO_CKPT" --ckpt_dir "$CKPT_DIR" --data_dir "$DATA_DIR"
