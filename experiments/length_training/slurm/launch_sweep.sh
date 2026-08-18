#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# launch_sweep.sh — submit the length-composition grid: 2 base metrics × 9 mixes
# × 2 encoder regimes = 36 arms, all on the same budget.
#
# Run it from a login node (it calls sbatch; it is not itself a batch job).
#
#   experiments/length_training/slurm/launch_sweep.sh              # dry run
#   SUBMIT=1 experiments/length_training/slurm/launch_sweep.sh     # really submit
#   SUBMIT=1 MODELS=da MIXES="frac000 frac100" .../launch_sweep.sh # first wave
#
# Tunables: MODELS ("da qe") · MIXES · FROZEN_LEVELS ("0 1") · MAX_EPOCHS ·
# PATIENCE · SEED · plus anything train.sh reads.
#
# Wave discipline: send frac000/frac100 × frozen/unfrozen per family first and
# check one full validation cycle before committing the other 28 arms.
#
# Unfrozen arms need >16 GB: gpu001/004/011/014 OOM at unfreeze, so they are
# excluded for FROZEN=0.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

EXP=experiments/length_training
MODELS="${MODELS:-da qe}"
MIXES="${MIXES:-frac000 frac010 frac020 frac040 frac060 frac080 frac100 frac000nat frac000agg}"
FROZEN_LEVELS="${FROZEN_LEVELS:-0 1}"
SMALL_GPUS="${SMALL_GPUS:-gpu001,gpu004,gpu011,gpu014}"

n=0
for model in $MODELS; do
  for mix in $MIXES; do
    for frozen in $FROZEN_LEVELS; do
      args=(--export="ALL,MODEL=$model,MIX=$mix,FROZEN=$frozen")
      [[ "$frozen" == "0" ]] && args+=(--exclude="$SMALL_GPUS")
      n=$((n + 1))
      if [[ "${SUBMIT:-0}" == "1" ]]; then
        jid=$(sbatch --parsable "${args[@]}" "$EXP/slurm/train.sh")
        echo "submitted $jid  model=$model mix=$mix frozen=$frozen"
      else
        echo "would submit: MODEL=$model MIX=$mix FROZEN=$frozen sbatch ${args[*]} $EXP/slurm/train.sh"
      fi
    done
  done
done

echo
echo "$n arm(s)$( [[ "${SUBMIT:-0}" == "1" ]] && echo " submitted" || echo " — dry run, set SUBMIT=1 to send them" )."
if [[ "${SUBMIT:-0}" != "1" ]]; then
  echo "Budget per arm: MAX_EPOCHS=${MAX_EPOCHS:-6} passes over a 24,000-row mix"
  echo "               (~4,500 optimizer steps at batch 4 x accum 8), PATIENCE=${PATIENCE:-3}."
fi
