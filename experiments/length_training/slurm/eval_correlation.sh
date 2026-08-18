#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# eval_correlation.sh — agreement with human judgement per input length, for
# every arm of the length-composition sweep plus the two public baselines.
#
# Two lenses, both run by default:
#   val      ~/scratch/wmt_length_data — the SAME validation split early stopping
#            monitors (sent k=1, agg k=2,3,4,6, native k=0, 15 language pairs).
#            Selection-coupled, so it is a development signal, not a headline
#            number — but it is the one directly comparable to `val_kendall`.
#   heldout  ~/scratch/wmt_eval_portion — WMT22/23/24/25 portions never trained
#            on; the reportable numbers.
#
# Predictions are cached per (model, exact input rows) AND validated against the
# checkpoint the label currently resolves to, so re-running after a retraining
# rescores the changed arms instead of replaying the old ones.
#
# Usage:
#   sbatch experiments/length_training/slurm/eval_correlation.sh
#   LENS=val sbatch experiments/length_training/slurm/eval_correlation.sh
#   ARMS="frac000 frac100" sbatch .../eval_correlation.sh
#   MODELS="da-frac040=/path/to.ckpt" sbatch .../eval_correlation.sh
#
# Tunables:
#   LENS         val | heldout | both        (default both)
#   MODELS       explicit label=ckpt list    (default: discover every arm)
#   ARMS         substrings to keep          (e.g. "frac000 frac100")
#   SELECT       best | last                 (default best, by val_kendall)
#   NEWER_THAN   skip checkpoints older than this ISO date / file
#   CKPT_ROOT    checkpoint root             (default ~/scratch/checkpoints/retrain-wmt)
#   OUT_DIR      results dir                 (default results/length_training)
# ──────────────────────────────────────────────────────────────────────────────
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
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

EXP=experiments/length_training
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV:-comet-bio}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
[[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]] && export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"

CKPT_ROOT="${CKPT_ROOT:-$HOME/scratch/checkpoints/retrain-wmt}"
OUT_DIR="${OUT_DIR:-results/length_training}"
LENS="${LENS:-both}"

# ── which checkpoints ─────────────────────────────────────────────────────────
if [[ -n "${MODELS:-}" ]]; then
  ARM_MODELS="$MODELS"
else
  DISCOVER=(--root "$CKPT_ROOT" --select "${SELECT:-best}")
  [[ -n "${ARMS:-}" ]]       && DISCOVER+=(--arms $ARMS)
  [[ -n "${NEWER_THAN:-}" ]] && DISCOVER+=(--newer_than "$NEWER_THAN")
  ARM_MODELS="$(python "$EXP/list_arms.py" "${DISCOVER[@]}" | tr '\n' ' ')"
fi
BASELINES="da-base=Unbabel/wmt22-comet-da qe-base=Unbabel/wmt22-cometkiwi-da"

echo "═══════════════════════════════════════════════════════════"
echo " Node   : $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo " Lens   : $LENS"
echo " Arms   : $(wc -w <<<"$ARM_MODELS") checkpoints + 2 baselines"
echo "═══════════════════════════════════════════════════════════"

run_lens () {
  local name="$1" data_dir="$2"
  echo; echo "### lens: $name ($data_dir) ###"
  srun python "$EXP/eval_correlation.py" \
      --models $BASELINES $ARM_MODELS \
      --data_dir "$data_dir" \
      --batch_size "${BATCH_SIZE:-32}" \
      --cache_dir "$OUT_DIR/pred_cache_$name" \
      --output "$OUT_DIR/correlation_$name.json"
}

[[ "$LENS" == "val"     || "$LENS" == "both" ]] && run_lens val     "$HOME/scratch/wmt_length_data"
[[ "$LENS" == "heldout" || "$LENS" == "both" ]] && run_lens heldout "$HOME/scratch/wmt_eval_portion"

echo; echo "Done → $OUT_DIR/correlation_*.json"
echo "Table: python $EXP/analyze.py --results $OUT_DIR/correlation_heldout.json"
