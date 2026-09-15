#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# deploy_four_arms.sh — build the mixes the four-arm comparison needs, then hand
# off to launch_long.sh.
#
# The four arms asked for are
#   frac100      100 % single sentences
#   frac000agg   100 % concatenated same-document windows
#   frac000nat   100 % natively long documents
#   frac000       50 / 50 concatenated + native
#
# and the first three take their whole N from ONE pool. The default mix policy
# sizes N as min(sent, 2*agg, 2*native) — fine for the fracNNN ladder, where no
# mix draws more than half its rows from a single long pool, but it leaves the
# pure arms infeasible, and make_mixtures.py used to drop them with a one-line
# log. That is why frac000nat and frac000agg were never trained.
#
# So this script rebuilds ONLY the mixes, with --total_policy pure
# (N = min(sent, agg, native)), into a separate mixes directory. The pools
# themselves are unchanged — the v2 pools already carry the 2026-08-24
# concatenation-aware token filter — so nothing has to be re-tokenised and the
# v1/v2 mixes and checkpoints are left exactly where they are.
#
# Usage (login node):
#   experiments/length_training/slurm/deploy_four_arms.sh              # dry run
#   RUN=1          .../deploy_four_arms.sh                             # build mixes
#   RUN=1 SUBMIT=1 .../deploy_four_arms.sh                             # ... + submit
#
# Tunables: DATA_DIR (pools, default ~/scratch/wmt_length_data_v2)
#           MIX_DIR  (default $DATA_DIR/mixes_pure)
#           CKPT_ROOT (default retrain-wmt-v3)
#           MODELS · ARMS · MAX_EPOCHS · PATIENCE · CHAIN → launch_long.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
EXP=experiments/length_training

export DATA_DIR="${DATA_DIR:-$HOME/scratch/wmt_length_data_v2}"
export MIX_DIR="${MIX_DIR:-$DATA_DIR/mixes_pure}"
export CKPT_ROOT="${CKPT_ROOT:-retrain-wmt-v3}"
ARMS="${ARMS:-frac100 frac000agg frac000nat frac000}"

echo "plan:"
echo "  0. pools   ← $DATA_DIR                (reused as-is, not rebuilt)"
echo "  1. mixes   → $MIX_DIR                 (--total_policy pure, arms: $ARMS)"
echo "  2. arms    → ~/scratch/checkpoints/$CKPT_ROOT, MAX_EPOCHS=${MAX_EPOCHS:-60}"
[[ "${RUN:-0}" == "1" ]] || { echo "dry run — set RUN=1 to execute."; exit 0; }

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-comet-bio}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"

if [[ ! -f "$DATA_DIR/all_train.csv" ]]; then
  echo "ERROR: no pools in $DATA_DIR."
  echo "Build them once (~30-60 min CPU, tokenisation-bound):"
  echo "  python $EXP/prepare_data.py --output_dir $DATA_DIR"
  exit 1
fi

if [[ ! -f "$MIX_DIR/manifest.json" ]]; then
  echo; echo "### Building the pure-policy mixes ###"
  # make_mixtures.py now REFUSES to silently skip an infeasible arm, so if the
  # native pool cannot fill an arm this exits loudly instead of quietly
  # producing a two-arm grid.
  # all_train.csv is ~350 MB and pandas holds it whole — run it on a compute
  # node rather than on the login node.
  srun --partition=cpu_devel --cpus-per-task=4 --mem=24G --time=01:00:00 \
    python "$EXP/make_mixtures.py" \
      --data_dir "$DATA_DIR" \
      --out_dir "$MIX_DIR" \
      --total_policy pure \
      --arms $ARMS
else
  echo "Mixes present: $MIX_DIR"
fi

echo; echo "### What is actually in each mix ###"
srun --partition=cpu_devel --cpus-per-task=4 --mem=24G --time=00:30:00 \
  python "$EXP/report_mix_composition.py" --data_dirs "$DATA_DIR" \
    --output "$MIX_DIR/composition.json" || true

if [[ "${SUBMIT:-0}" == "1" ]]; then
  SUBMIT=1 ARMS="$ARMS" "$EXP/slurm/launch_long.sh"
else
  echo
  echo "mixes ready — submit with:"
  echo "  SUBMIT=1 DATA_DIR=$DATA_DIR MIX_DIR=$MIX_DIR CKPT_ROOT=$CKPT_ROOT \\"
  echo "    $EXP/slurm/launch_long.sh"
fi
