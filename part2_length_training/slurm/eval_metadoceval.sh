#!/usr/bin/env bash
# MetaDocEval lens (contrastive accuracy per perturbation and window) for every trained arm
# plus the two published baselines. Clones or pulls the test set first.
#   sbatch part2_length_training/slurm/eval_metadoceval.sh
#   ARMS="frac000 frac100" WINDOWS="1 6" sbatch part2_length_training/slurm/eval_metadoceval.sh
# Tunables: MODELS (explicit label=ckpt list) ARMS SELECT (best|last) NEWER_THAN WINDOWS ("1 3 6 9")
#           CKPT_ROOT (~/scratch/checkpoints/retrain-wmt-v3) DATA_DIR (~/scratch/metadoceval-testset)
#           OUT_DIR (part2_length_training/results) BATCH_SIZE (32) WANDB_PROJECT RUN_NAME
#SBATCH --job-name=eval-mde
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00

set -euo pipefail
source common/cluster_env.sh

CKPT_ROOT="${CKPT_ROOT:-$FDTEM_SCRATCH/checkpoints/retrain-wmt-v3}"
DATA_DIR="${DATA_DIR:-$FDTEM_SCRATCH/metadoceval-testset}"
OUT_DIR="${OUT_DIR:-part2_length_training/results}"
WINDOWS="${WINDOWS:-1 3 6 9}"
BASELINES="da-base=Unbabel/wmt22-comet-da qe-base=Unbabel/wmt22-cometkiwi-da"

if [[ -n "${MODELS:-}" ]]; then
  ARM_MODELS="$MODELS"
else
  DISCOVER=(--root "$CKPT_ROOT" --select "${SELECT:-best}")
  if [[ -n "${ARMS:-}" ]]; then DISCOVER+=(--arms $ARMS); fi
  if [[ -n "${NEWER_THAN:-}" ]]; then DISCOVER+=(--newer_than "$NEWER_THAN"); fi
  ARM_MODELS="$(python -m part2_length_training.list_arms "${DISCOVER[@]}" | tr '\n' ' ')"
fi
echo "node $(hostname)  windows=$WINDOWS  $(wc -w <<<"$ARM_MODELS") checkpoints + 2 baselines"

srun python -m part2_length_training.load_metadoceval --data_dir "$DATA_DIR"
srun python -m part2_length_training.eval_metadoceval --data_dir "$DATA_DIR" \
    --models $BASELINES $ARM_MODELS --windows $WINDOWS --batch_size "${BATCH_SIZE:-32}" \
    --cache_dir "$OUT_DIR/cache/metadoceval" --output "$OUT_DIR/metadoceval.json" \
    --run_name "${RUN_NAME:-metadoceval}" ${WANDB_PROJECT:+--wandb_project "$WANDB_PROJECT"}
echo "done -> $OUT_DIR/metadoceval.json (+ plots/metadoceval_accuracy.png)"
