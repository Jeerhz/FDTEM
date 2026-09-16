#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# launch_uncontrolled.sh — the deliberately contaminated arm.
#
# Trains on every scored row available, evaluation sets included, with an
# aggressive schedule, for as long as the other arms get. The point is not a
# reportable metric — it is the far end of the axis: the published metric is
# "saw none of this", the four composition arms are a controlled step away, and
# this is as far as continued training can push. See make_uncontrolled_mix.py.
#
# What it is NOT: a correlation result. Its validation and held-out numbers
# measure memorisation. MetaDocEval is a different corpus and nothing from it
# enters the mix, so that lens stays honest for this arm — which is the whole
# reason to build it.
#
# Differences from a sweep arm, all deliberate:
#   * data     every pool split + every held-out evaluation portion
#   * encoder  unfrozen from step 0 (NR_FROZEN_EPOCHS=0), not after 0.3 epochs
#   * LR       10x the conservative continue-training rates
#   * val      a random 2 % slice of the contaminated pool, so the trainer has
#              something to checkpoint on. Not a measurement.
#
# Usage (login node):
#   experiments/length_training/slurm/launch_uncontrolled.sh              # dry run
#   RUN=1          .../launch_uncontrolled.sh                             # build the mix
#   RUN=1 SUBMIT=1 .../launch_uncontrolled.sh                             # ... + train
#
# Tunables: MODELS ("da qe") · MAX_EPOCHS (60) · CHAIN (4) · ENCODER_LR (5e-6) ·
#           HEAD_LR (1e-4) · DATA_DIR · MIX_DIR · EVAL_DIRS · CKPT_ROOT
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
EXP=experiments/length_training

MODELS="${MODELS:-da qe}"
CHAIN="${CHAIN:-4}"
EVAL_DIRS="${EVAL_DIRS:-$HOME/scratch/wmt_eval_portion}"

export DATA_DIR="${DATA_DIR:-$HOME/scratch/wmt_length_data_v2}"
export MIX_DIR="${MIX_DIR:-$DATA_DIR/mixes_pure}"
export CKPT_ROOT="${CKPT_ROOT:-retrain-wmt-v3}"
export MAX_EPOCHS="${MAX_EPOCHS:-60}"
export PATIENCE="${PATIENCE:-1000}"        # never stop early: the goal is drift
export ENCODER_LR="${ENCODER_LR:-5.0e-6}"
export HEAD_LR="${HEAD_LR:-1.0e-4}"
export NR_FROZEN_EPOCHS="${NR_FROZEN_EPOCHS:-0}"
SMALL_GPUS="${SMALL_GPUS:-gpu001,gpu004,gpu011,gpu014}"

MIX=uncontrolled
UMIX="$MIX_DIR/$MIX"

echo "plan:"
echo "  1. mix    → $UMIX   (pools train+val + $EVAL_DIRS)"
echo "  2. arms   → ~/scratch/checkpoints/$CKPT_ROOT/{,kiwi-}mix-$MIX"
echo "  3. budget → MAX_EPOCHS=$MAX_EPOCHS, encoder LR $ENCODER_LR, head LR $HEAD_LR,"
echo "              unfrozen from step 0, early stopping effectively off"
[[ "${RUN:-0}" == "1" ]] || { echo "dry run — set RUN=1 to execute."; exit 0; }

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-comet-bio}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"

if [[ ! -f "$UMIX/all_train.csv" ]]; then
  echo; echo "### Building the contaminated mix ###"
  python "$EXP/make_uncontrolled_mix.py" \
      --data_dir "$DATA_DIR" \
      --eval_dirs $EVAL_DIRS \
      --out_dir "$UMIX"
else
  echo "Mix present: $UMIX"
fi
[[ -f "$UMIX/CONTAMINATED" ]] || { echo "ERROR: $UMIX has no CONTAMINATED marker"; exit 1; }
cat "$UMIX/CONTAMINATED"

if [[ "${SUBMIT:-0}" != "1" ]]; then
  echo; echo "mix ready — submit with: RUN=1 SUBMIT=1 $0"
  exit 0
fi

# The trainer must monitor the contaminated mix's own slice, not the clean
# pools' all_val.csv: half of that file is now in this arm's training set.
export VAL_FILES="$UMIX/all_val.csv"

n=0
for model in $MODELS; do
  dep=""
  for link in $(seq 1 "$CHAIN"); do
    args=(--export="ALL,MODEL=$model,MIX=$MIX,FROZEN=0,RESUME=auto,\
MAX_EPOCHS=$MAX_EPOCHS,PATIENCE=$PATIENCE,DATA_DIR=$DATA_DIR,MIX_DIR=$MIX_DIR,\
CKPT_ROOT=$CKPT_ROOT,VAL_FILES=$VAL_FILES,ENCODER_LR=$ENCODER_LR,HEAD_LR=$HEAD_LR,\
NR_FROZEN_EPOCHS=$NR_FROZEN_EPOCHS")
    args+=(--exclude="$SMALL_GPUS")
    [[ -n "$dep" ]] && args+=(--dependency="afterany:$dep")
    jid=$(sbatch --parsable "${args[@]}" "$EXP/slurm/train.sh")
    echo "submitted $jid  model=$model arm=$MIX link=$link/$CHAIN${dep:+ after $dep}"
    dep="$jid"
    n=$((n + 1))
  done
done

echo
echo "$n job(s) submitted."
echo "Evaluate on MetaDocEval ONLY — the correlation lens is contaminated:"
echo "  sbatch --export=ALL,CKPT_ROOT=$HOME/scratch/checkpoints/$CKPT_ROOT,\
ARMS=uncontrolled $EXP/slurm/eval_metadoceval.sh"
