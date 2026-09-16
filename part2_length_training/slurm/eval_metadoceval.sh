#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# eval_metadoceval.sh — document-level meta-evaluation of the sweep's arms on
# the MetaDocEval contrastive test set (Dahan, Bawden & Yvon, EAMT 2026).
#
# For each perturbation and context window w, the accuracy is the share of
# documents the metric scores higher in their original than in their perturbed
# version. Chance is 50%. `sentence_splitting` is quality-preserving by design,
# so its "accuracy" is a false-positive rate — the script flags it as such.
#
# This is the lens that answers whether long-text training buys genuine
# discourse-error detection rather than a rescaled score: segment correlation
# cannot separate the two, because discourse negatives keep sentence-level
# fluency intact.
#
# Clones the test set if missing. Window scoring is deduplicated across
# perturbations and cached per model, and a cache entry is only reused when it
# was produced by the checkpoint the label currently points at — so re-running
# after a retraining rescores the changed arms instead of replaying old scores.
#
# Usage:
#   sbatch experiments/length_training/slurm/eval_metadoceval.sh
#   ARMS="frac000 frac100" sbatch .../eval_metadoceval.sh
#   MODELS="da-frac040=/path/to.ckpt" WINDOWS="1 6" sbatch .../eval_metadoceval.sh
#
# Tunables:
#   MODELS       explicit label=ckpt list    (default: discover every arm)
#   ARMS         substrings to keep          (e.g. "frac000 frac100")
#   SELECT       best | last                 (default best, by val_kendall)
#   NEWER_THAN   skip checkpoints older than this ISO date / file
#   WINDOWS      context window sizes        (default "1 3 6 9")
#   CKPT_ROOT    checkpoint root             (default ~/scratch/checkpoints/retrain-wmt)
#   DATA_DIR     test-set clone              (default ~/scratch/metadoceval-testset)
#   OUT_DIR      results dir                 (default results/length_training)
# ──────────────────────────────────────────────────────────────────────────────
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
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

EXP=experiments/length_training
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV:-comet-bio}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
mkdir -p "$HF_HOME"
[[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]] && export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"

CKPT_ROOT="${CKPT_ROOT:-$HOME/scratch/checkpoints/retrain-wmt}"
DATA_DIR="${DATA_DIR:-$HOME/scratch/metadoceval-testset}"
OUT_DIR="${OUT_DIR:-results/length_training}"
WINDOWS="${WINDOWS:-1 3 6 9}"

# ── test set ──────────────────────────────────────────────────────────────────
if [[ ! -f "$DATA_DIR/data/en_fr_aya.json" ]]; then
  echo "### Cloning MetaDocEval test set → $DATA_DIR ###"
  git clone --depth 1 https://github.com/nicolasdahan/metadoceval-testset.git "$DATA_DIR"
else
  echo "Test set present: $DATA_DIR"
fi

# ── which checkpoints ─────────────────────────────────────────────────────────
if [[ -n "${MODELS:-}" ]]; then
  ARM_MODELS="$MODELS"
else
  DISCOVER=(--root "$CKPT_ROOT" --select "${SELECT:-best}")
  [[ -n "${ARMS:-}" ]]       && DISCOVER+=(--arms $ARMS)
  [[ -n "${NEWER_THAN:-}" ]] && DISCOVER+=(--newer_than "$NEWER_THAN")
  ARM_MODELS="$(python "$EXP/list_arms.py" "${DISCOVER[@]}" | tr '\n' ' ')"
fi
# cometkiwi is reference-free; eval_metadoceval.py detects that and drops `ref`
BASELINES="da-base=Unbabel/wmt22-comet-da qe-base=Unbabel/wmt22-cometkiwi-da"

echo "═══════════════════════════════════════════════════════════"
echo " Node    : $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo " Windows : $WINDOWS"
echo " Arms    : $(wc -w <<<"$ARM_MODELS") checkpoints + 2 baselines"
echo "═══════════════════════════════════════════════════════════"

srun python "$EXP/eval_metadoceval.py" \
    --data_dir "$DATA_DIR" \
    --models $BASELINES $ARM_MODELS \
    --windows $WINDOWS \
    --batch_size "${BATCH_SIZE:-32}" \
    --cache_dir "$OUT_DIR/mde_cache" \
    --output "$OUT_DIR/metadoceval.json" \
    --run_name "${RUN_NAME:-metadoceval}" \
    ${WANDB_PROJECT:+--wandb_project "$WANDB_PROJECT"}

echo; echo "Done → $OUT_DIR/metadoceval.json (+ plots/)"
