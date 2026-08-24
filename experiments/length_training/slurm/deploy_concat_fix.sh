#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# deploy_concat_fix.sh — rebuild the WMT pools with the concatenation-aware
# token filter (QE truncation fix, 2026-08-24) and relaunch the sweep on them.
#
# Why: CometKiwi concatenates mt+src into ONE 512-token sequence, but the old
# filter capped each side at 480 tokens separately. 7.8% of native and 10.7% of
# k=6 training rows overflowed, silently cutting the source for the QE arms.
# The fix (prepare_data.py --max_concat_tokens, default 508) drops those rows
# from the SHARED pools, so DA and QE still train on byte-identical rows.
#
# v1 data (~/scratch/wmt_length_data) and checkpoints (retrain-wmt) are left
# untouched; everything new goes to *_v2 paths.
#
# Usage (login node):
#   experiments/length_training/slurm/deploy_concat_fix.sh              # dry run
#   RUN=1          experiments/length_training/slurm/deploy_concat_fix.sh  # pools + mixes (srun, ~1 h)
#   RUN=1 SUBMIT=1 experiments/length_training/slurm/deploy_concat_fix.sh  # ... then submit the arms
#
# Tunables: DATA_DIR (default ~/scratch/wmt_length_data_v2)
#           CKPT_ROOT (default retrain-wmt-v2)
#           MODELS · MIXES · FROZEN_LEVELS — forwarded to launch_sweep.sh
#           (default: first wave, frac000 + frac100, both families)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
EXP=experiments/length_training

export DATA_DIR="${DATA_DIR:-$HOME/scratch/wmt_length_data_v2}"
export CKPT_ROOT="${CKPT_ROOT:-retrain-wmt-v2}"
MODELS="${MODELS:-da qe}"
MIXES="${MIXES:-frac000 frac100}"

echo "plan:"
echo "  1. pools  → $DATA_DIR  (srun CPU, ~30-60 min; per-side 480 + src+mt <= 508)"
echo "  2. mixes  → $DATA_DIR/mixes"
echo "  3. arms   → MODELS='$MODELS' MIXES='$MIXES', ckpts → ~/scratch/checkpoints/$CKPT_ROOT"
[[ "${RUN:-0}" == "1" ]] || { echo "dry run — set RUN=1 to execute."; exit 0; }

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-comet-bio}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"

if [[ ! -f "$DATA_DIR/all_train.csv" ]]; then
  srun --partition=cpu_devel --cpus-per-task=8 --mem=32G --time=03:00:00 \
    python "$EXP/prepare_data.py" --output_dir "$DATA_DIR"
fi
if [[ ! -f "$DATA_DIR/mixes/manifest.json" ]]; then
  srun --partition=cpu_devel --cpus-per-task=4 --mem=16G --time=01:00:00 \
    python "$EXP/make_mixtures.py" --data_dir "$DATA_DIR"
fi

# design guard: constant total across mixes, concat budget respected
python - "$DATA_DIR" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
man = json.load(open(d / "mixes" / "manifest.json"))
totals = {m: sum(v["counts"][p] for p in ("sent", "agg", "native"))
          for m, v in man["mixes"].items()}
assert len(set(totals.values())) == 1, f"unequal mix totals: {totals}"
print(f"mixes OK: {len(totals)} mixes x {next(iter(totals.values())):,} rows, "
      f"seed {man['seed']}, pools {man['pool_sizes']}")
PY

if [[ "${SUBMIT:-0}" == "1" ]]; then
  SUBMIT=1 MODELS="$MODELS" MIXES="$MIXES" "$EXP/slurm/launch_sweep.sh"
else
  echo "data ready — submit with:"
  echo "  SUBMIT=1 DATA_DIR=$DATA_DIR CKPT_ROOT=$CKPT_ROOT MODELS='$MODELS' MIXES='$MIXES' $EXP/slurm/launch_sweep.sh"
fi
# eval scripts take CKPT_ROOT as a full path
EVAL_ROOT="$HOME/scratch/checkpoints/$CKPT_ROOT"
echo "evaluate later with:"
echo "  sbatch --export=ALL,CKPT_ROOT=$EVAL_ROOT,VAL_DATA_DIR=$DATA_DIR,NEWER_THAN=$(date +%F),SELECT=best $EXP/slurm/eval_correlation.sh"
echo "  sbatch --export=ALL,CKPT_ROOT=$EVAL_ROOT,NEWER_THAN=$(date +%F),SELECT=best $EXP/slurm/eval_metadoceval.sh"
