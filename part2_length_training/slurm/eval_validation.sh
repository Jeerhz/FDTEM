#!/usr/bin/env bash
# Correlation lens (val + heldout) and, with PROFILE=1, the length profile on the same
# prediction cache, for every trained arm plus the two published baselines.
#   sbatch part2_length_training/slurm/eval_validation.sh
#   LENS=heldout ARMS="frac000 frac100" sbatch part2_length_training/slurm/eval_validation.sh
#   MODELS="da-frac040=/path/to.ckpt" sbatch part2_length_training/slurm/eval_validation.sh
# Tunables: LENS (both|val|heldout) MODELS (explicit label=ckpt list) ARMS SELECT (best|last) NEWER_THAN
#           CKPT_ROOT (~/scratch/checkpoints/retrain-wmt-v3) OUT_DIR (part2_length_training/results)
#           VAL_DATA_DIR HELDOUT_DATA_DIR PROFILE (1) BATCH_SIZE (32)
#SBATCH --job-name=eval-corr
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

LENS="${LENS:-both}"
CKPT_ROOT="${CKPT_ROOT:-$FDTEM_SCRATCH/checkpoints/retrain-wmt-v3}"
OUT_DIR="${OUT_DIR:-part2_length_training/results}"
VAL_DATA_DIR="${VAL_DATA_DIR:-$FDTEM_SCRATCH/wmt_length_data_v2}"
HELDOUT_DATA_DIR="${HELDOUT_DATA_DIR:-$FDTEM_SCRATCH/wmt_eval_portion}"
BASELINES="da-base=Unbabel/wmt22-comet-da qe-base=Unbabel/wmt22-cometkiwi-da"

if [[ -n "${MODELS:-}" ]]; then
  ARM_MODELS="$MODELS"
else
  DISCOVER=(--root "$CKPT_ROOT" --select "${SELECT:-best}")
  if [[ -n "${ARMS:-}" ]]; then DISCOVER+=(--arms $ARMS); fi
  if [[ -n "${NEWER_THAN:-}" ]]; then DISCOVER+=(--newer_than "$NEWER_THAN"); fi
  ARM_MODELS="$(python -m part2_length_training.list_arms "${DISCOVER[@]}" | tr '\n' ' ')"
fi
echo "node $(hostname)  lens=$LENS  $(wc -w <<<"$ARM_MODELS") checkpoints + 2 baselines"

run_lens () {
  local lens="$1" data_dir="$2"
  srun python -m part2_length_training.eval_validation --lens "$lens" --data_dir "$data_dir" \
      --models $BASELINES $ARM_MODELS --batch_size "${BATCH_SIZE:-32}" \
      --cache_dir "$OUT_DIR/cache/pred_$lens" --output "$OUT_DIR/correlation_$lens.json"
  if [[ "${PROFILE:-1}" == "1" ]]; then
    srun python -m part2_length_training.eval_length_profile --lens "$lens" --data_dir "$data_dir" \
        --models $BASELINES $ARM_MODELS --batch_size "${BATCH_SIZE:-32}" \
        --cache_dir "$OUT_DIR/cache/pred_$lens" --output "$OUT_DIR/length_profile_$lens.json"
  fi
}

if [[ "$LENS" == "val" || "$LENS" == "both" ]]; then run_lens val "$VAL_DATA_DIR"; fi
if [[ "$LENS" == "heldout" || "$LENS" == "both" ]]; then run_lens heldout "$HELDOUT_DATA_DIR"; fi
echo "done -> $OUT_DIR/correlation_*.json  $OUT_DIR/length_profile_*.json"
echo "table: python -m part2_length_training.analyze --lens heldout"
