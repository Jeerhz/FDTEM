#!/usr/bin/env bash
#SBATCH --job-name=comet-bio-mqm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Bio-MQM data -> fine-tune wmt22-comet-da -> evaluate base vs fine-tuned.
#   sbatch part3_biomed_finetune/slurm/finetune.sh
# Tunables: DATA_DIR CKPT_DIR RUN_NAME WANDB_PROJECT MAX_EPOCHS PATIENCE SEED
#           RESUME (empty | auto | <ckpt>)  LANG_PAIRS (empty = all ten pairs)
set -euo pipefail
source common/cluster_env.sh

DATA_DIR="${DATA_DIR:-$FDTEM_SCRATCH/bio_mqm}"
CKPT_DIR="${CKPT_DIR:-$FDTEM_SCRATCH/checkpoints/bio_mqm}"
RUN_NAME="${RUN_NAME:-comet-bio-mqm-$(date +%Y%m%d-%H%M)}"
WANDB_PROJECT="${WANDB_PROJECT:-comet-bio-mqm}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
SEED="${SEED:-42}"
RESUME="${RESUME:-}"
LANG_PAIRS="${LANG_PAIRS:-}"

if [[ ! -f "$DATA_DIR/all_train.csv" ]]; then
  # shellcheck disable=SC2086
  srun python -m part3_biomed_finetune.load_bio_mqm --output_dir "$DATA_DIR" \
    ${LANG_PAIRS:+--lang_pairs $LANG_PAIRS}
fi

srun python -m common.train_comet \
  --base_cfg part3_biomed_finetune/configs/finetune.yaml \
  --train_files "$DATA_DIR/all_train.csv" --val_files "$DATA_DIR/all_val.csv" \
  --base_model Unbabel/wmt22-comet-da \
  --ckpt_dir "$CKPT_DIR" --run_name "$RUN_NAME" --wandb_project "$WANDB_PROJECT" \
  --tags bio-mqm,comet,finetuning \
  --max_epochs "$MAX_EPOCHS" --patience "$PATIENCE" --save_top_k 1 --seed "$SEED" \
  ${RESUME:+--resume "$RESUME"}

srun python -m part3_biomed_finetune.evaluate --bio auto --data_dir "$DATA_DIR" --ckpt_dir "$CKPT_DIR"

echo "To publish the checkpoint:"
echo "  python -m part3_biomed_finetune.upload_to_huggingface --checkpoint auto --ckpt_dir $CKPT_DIR --repo_id <user>/comet-bio-mqm --run_name $RUN_NAME"
