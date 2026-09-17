#!/usr/bin/env bash
# Login-node launcher: build the missing mixes, check that every arm has the same number
# of rows, then submit each (model, arm, frozen) as a chain of CHAIN jobs linked by
# --dependency=afterany, every link carrying RESUME=auto (one W&B run per arm; Lightning
# counts max_epochs globally, so the chain stops itself at the preset's budget).
#
#   part2_length_training/slurm/launch_arms.sh                 # dry run
#   RUN=1 part2_length_training/slurm/launch_arms.sh           # build missing mixes
#   RUN=1 SUBMIT=1 part2_length_training/slurm/launch_arms.sh  # ... and submit
#   SUBMIT=1 MODELS=da ARMS=uncontrolled PRESET=uncontrolled part2_length_training/slurm/launch_arms.sh
#   PRESET=wave1 ARMS="frac000 frac010 frac020 frac040 frac060 frac080 frac100" FROZEN_LEVELS="0 1" \
#       CHAIN=1 MIX_DIR=$HOME/scratch/wmt_length_data_v2/mixes SUBMIT=1 part2_length_training/slurm/launch_arms.sh
#
# Tunables: MODELS ("da qe") ARMS ("frac100 frac000agg frac000nat frac000") FROZEN_LEVELS ("0")
#           PRESET (default) CHAIN (4) DATA_DIR MIX_DIR ($DATA_DIR/mixes_pure) CKPT_ROOT (retrain-wmt-v3)
#           TOTAL_POLICY (pure) EVAL_DIRS (uncontrolled arm: ~/scratch/wmt_eval_portion)
#           SMALL_GPUS (excluded for unfrozen arms) RUN SUBMIT
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source common/cluster_env.sh

MODELS="${MODELS:-da qe}"
ARMS="${ARMS:-frac100 frac000agg frac000nat frac000}"
FROZEN_LEVELS="${FROZEN_LEVELS:-0}"
PRESET="${PRESET:-default}"
CHAIN="${CHAIN:-4}"
DATA_DIR="${DATA_DIR:-$FDTEM_SCRATCH/wmt_length_data_v2}"
MIX_DIR="${MIX_DIR:-$DATA_DIR/mixes_pure}"
CKPT_ROOT="${CKPT_ROOT:-retrain-wmt-v3}"
TOTAL_POLICY="${TOTAL_POLICY:-pure}"
EVAL_DIRS="${EVAL_DIRS:-$FDTEM_SCRATCH/wmt_eval_portion}"
SMALL_GPUS="${SMALL_GPUS:-gpu001,gpu004,gpu011,gpu014}"
SRUN_CPU=(srun --partition=cpu_devel --cpus-per-task=4 --mem=24G --time=01:00:00)

[[ -f "$DATA_DIR/all_train.csv" ]] || { echo "no pools in $DATA_DIR: python -m part2_length_training.load_wmt_pools --output_dir $DATA_DIR"; exit 1; }

# 1. mixes: build whatever is missing (all_train.csv is ~350 MB, so on a compute node)
CONTROLLED=""; MISSING_CONTROLLED=""; MISSING_UNCONTROLLED=0
for arm in $ARMS; do
  if [[ "$arm" == "uncontrolled" ]]; then
    [[ -f "$MIX_DIR/uncontrolled/all_train.csv" ]] || MISSING_UNCONTROLLED=1
  else
    CONTROLLED="$CONTROLLED $arm"
    [[ -f "$MIX_DIR/$arm/all_train.csv" ]] || MISSING_CONTROLLED="$MISSING_CONTROLLED $arm"
  fi
done
if [[ -n "$MISSING_CONTROLLED" || "$MISSING_UNCONTROLLED" == "1" ]]; then
  if [[ "${RUN:-0}" != "1" ]]; then
    echo "mixes missing under $MIX_DIR (controlled:${MISSING_CONTROLLED:- none}, uncontrolled: $MISSING_UNCONTROLLED) - set RUN=1 to build them"
    [[ "${SUBMIT:-0}" == "1" ]] && exit 1
  else
    if [[ -n "$CONTROLLED" ]]; then
      "${SRUN_CPU[@]}" python -m part2_length_training.make_mixtures --data_dir "$DATA_DIR" --out_dir "$MIX_DIR" \
          --total_policy "$TOTAL_POLICY" --arms $CONTROLLED
    fi
    if [[ "$MISSING_UNCONTROLLED" == "1" ]]; then
      "${SRUN_CPU[@]}" python -m part2_length_training.make_mixtures --data_dir "$DATA_DIR" --out_dir "$MIX_DIR" \
          --arms uncontrolled --eval_dirs $EVAL_DIRS
    fi
    "${SRUN_CPU[@]}" python -m part2_length_training.inspect_mixtures --data_dirs "$DATA_DIR" \
        --mix_subdirs "$(basename "$MIX_DIR")" || true
  fi
fi

# 2. design guard: every arm the same N, and the budget that implies
if [[ -z "$MISSING_CONTROLLED" && "$MISSING_UNCONTROLLED" == "0" ]]; then
  python -m part2_length_training.make_mixtures --data_dir "$DATA_DIR" --out_dir "$MIX_DIR" \
      --verify --arms $ARMS --preset "$PRESET"
fi

# 3. submit the chains
echo
echo "models=$MODELS  arms=$ARMS  frozen=$FROZEN_LEVELS  preset=$PRESET  chain=$CHAIN  ckpts=$FDTEM_SCRATCH/checkpoints/$CKPT_ROOT"
n=0
for model in $MODELS; do
  for arm in $ARMS; do
    for frozen in $FROZEN_LEVELS; do
      dep=""
      for link in $(seq 1 "$CHAIN"); do
        args=(--export="ALL,MODEL=$model,MIX=$arm,FROZEN=$frozen,PRESET=$PRESET,RESUME=auto,DATA_DIR=$DATA_DIR,MIX_DIR=$MIX_DIR,CKPT_ROOT=$CKPT_ROOT")
        [[ "$frozen" == "0" ]] && args+=(--exclude="$SMALL_GPUS")
        [[ -n "$dep" ]] && args+=(--dependency="afterany:$dep")
        n=$((n + 1))
        if [[ "${SUBMIT:-0}" == "1" ]]; then
          jid=$(sbatch --parsable "${args[@]}" part2_length_training/slurm/train.sh)
          echo "submitted $jid  model=$model arm=$arm frozen=$frozen link=$link/$CHAIN${dep:+ after $dep}"
          dep="$jid"
        else
          echo "would submit: model=$model arm=$arm frozen=$frozen link=$link/$CHAIN${dep:+ after previous link}"
          dep="PREV"
        fi
      done
    done
  done
done
echo
if [[ "${SUBMIT:-0}" == "1" ]]; then
  echo "$n job(s) submitted. Then: CKPT_ROOT=$FDTEM_SCRATCH/checkpoints/$CKPT_ROOT sbatch part2_length_training/slurm/eval_validation.sh"
else
  echo "$n job(s) - dry run, set SUBMIT=1 to send them."
fi
