#!/usr/bin/env bash
# One arm of the length-composition sweep (part 2). Run from the repo root.
#   MODEL=da MIX=frac040 sbatch part2_length_training/slurm/train.sh
#   MODEL=qe MIX=frac000 FROZEN=1 PRESET=wave1 sbatch part2_length_training/slurm/train.sh
#   MODEL=da MIX=frac040 RESUME=auto sbatch ...      # continue this arm's own last.ckpt (chained jobs)
# Tunables: MODEL (da|qe) MIX (required) FROZEN (0) PRESET (default|wave1|uncontrolled)
#           RESUME ('' | auto | <ckpt>) DATA_DIR MIX_DIR CKPT_ROOT SEED WANDB_PROJECT RUN_NAME
#SBATCH --job-name=comet-retrain
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail
source common/cluster_env.sh

MODEL="${MODEL:-da}"
FROZEN="${FROZEN:-0}"
PRESET="${PRESET:-default}"
DATA_DIR="${DATA_DIR:-$FDTEM_SCRATCH/wmt_length_data_v2}"
MIX_DIR="${MIX_DIR:-$DATA_DIR/mixes_pure}"
CKPT_ROOT="${CKPT_ROOT:-retrain-wmt-v3}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-comet-retrain-wmt}"
[[ -n "${MIX:-}" ]] || { echo "MIX=<mix name> is required"; exit 1; }

ARGS=(--family "$MODEL" --mix "$MIX" --preset "$PRESET" --data_dir "$DATA_DIR" --mix_dir "$MIX_DIR"
      --ckpt_root "$CKPT_ROOT" --seed "$SEED" --wandb_project "$WANDB_PROJECT")
if [[ "$FROZEN" == "1" ]]; then ARGS+=(--frozen); fi
if [[ -n "${RESUME:-}" ]]; then ARGS+=(--resume "$RESUME"); fi
if [[ -n "${RUN_NAME:-}" ]]; then ARGS+=(--run_name "$RUN_NAME"); fi

echo "node $(hostname)  model=$MODEL mix=$MIX frozen=$FROZEN preset=$PRESET resume=${RESUME:-none}"
nvidia-smi || true
srun python -m part2_length_training.train "${ARGS[@]}"
