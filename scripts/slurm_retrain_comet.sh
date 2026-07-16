#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# slurm_retrain_comet.sh — COMET retraining experiments (two arms).
#
#   VARIANT=paragraph     continue Unbabel/wmt22-comet-da on the length-balanced
#                         paragraph mix (isolates the DATA effect)
#   VARIANT=encoder_swap  retrain from raw encoder weights on the same mix
#                         (isolates the ENCODER effect; default
#                         microsoft/infoxlm-large — override with $PRETRAINED /
#                         $ENCODER_MODEL, e.g. PRETRAINED=xlm-roberta-large for
#                         the controlled from-scratch baseline)
#
# Builds the paragraph data (scripts/prepare_paragraph_data.py) if missing,
# then trains via scripts/train_wandb.py with W&B logging.
#
# Usage:
#   VARIANT=paragraph    sbatch scripts/slurm_retrain_comet.sh
#   VARIANT=encoder_swap sbatch scripts/slurm_retrain_comet.sh
#   VARIANT=encoder_swap PRETRAINED=xlm-roberta-large RUN_NAME=xlmr-scratch \
#       sbatch scripts/slurm_retrain_comet.sh
#   # resume after timeout:
#   VARIANT=paragraph RESUME=<ckpt> WANDB_RUN_ID=<id> sbatch scripts/slurm_retrain_comet.sh
#
# Tunables (env at submit time):
#   VARIANT         paragraph | encoder_swap        (default paragraph)
#   CONDA_ENV       conda env                       (default comet-bio)
#   RUN_NAME        W&B run name                    (default <variant>-<date>)
#   WANDB_PROJECT   W&B project                     (default comet-retrain)
#   DATA_DIR        paragraph data dir              (default ~/scratch/paragraph_mqm)
#   PRETRAINED      HF encoder id (encoder_swap)    (default from the YAML)
#   ENCODER_MODEL   COMET encoder class             (XLM-RoBERTa | RemBERT | XLM-RoBERTa-XL)
#   RESUME          checkpoint to resume from
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=comet-retrain
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# ── environment ───────────────────────────────────────────────────────────────
CONDA_BASE="$HOME/miniconda3"
if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
else
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-comet-bio}"
fi

export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME"
# gated/hub auth under redirected HF_HOME (token lives in the default location)
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

VARIANT="${VARIANT:-paragraph}"
DATA_DIR="${DATA_DIR:-$HOME/scratch/paragraph_mqm}"
CKPT_DIR="$HOME/scratch/checkpoints/retrain/$VARIANT"
mkdir -p "$CKPT_DIR"

export WANDB_PROJECT="${WANDB_PROJECT:-comet-retrain}"
export WANDB_RUN_NAME="${RUN_NAME:-${VARIANT}-$(date +%Y%m%d-%H%M)}"
export WANDB_TAGS="retrain,${VARIANT}"
export WANDB_SAVE_DIR="$CKPT_DIR"
if ! wandb status &>/dev/null; then
  echo "W&B not configured — setting WANDB_MODE=offline"
  export WANDB_MODE=offline
fi

echo "═══════════════════════════════════════════════════════════"
echo " Node     : $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo " Variant  : $VARIANT"
echo " Data     : $DATA_DIR"
echo " Ckpts    : $CKPT_DIR"
echo " W&B      : project=$WANDB_PROJECT run=$WANDB_RUN_NAME mode=${WANDB_MODE:-online}"
echo "═══════════════════════════════════════════════════════════"
nvidia-smi || true

# ── 1. paragraph data (build once) ─────────────────────────────────────────────
# en/de/es/fr/ru only — zh/ja/th dropped (see slurm_block_xsim.sh rationale;
# the configs list the matching 8 CSVs).
LANG_PAIRS="${LANG_PAIRS:-en-de de-en en-es es-en en-fr fr-en en-ru ru-en}"
if [[ ! -f "$DATA_DIR/all_train.csv" ]]; then
  echo; echo "### Building length-balanced paragraph data ###"
  srun python scripts/prepare_paragraph_data.py \
      --output_dir "$DATA_DIR" --lang_pairs $LANG_PAIRS \
      --wandb_project "$WANDB_PROJECT"
else
  echo "Paragraph data present: $DATA_DIR"
fi

# ── 2. variant-specific config / checkpoint ────────────────────────────────────
EXTRA_ARGS=()
case "$VARIANT" in
  paragraph)
    CFG=configs/models/comet_paragraph_continue.yaml
    if [[ -z "${RESUME:-}" ]]; then
      BASE_CKPT=$(python - 2>/dev/null <<'EOF'
from comet import download_model
print(download_model("Unbabel/wmt22-comet-da"))
EOF
)
      [[ -f "$BASE_CKPT" ]] || { echo "ERROR: could not download wmt22-comet-da"; exit 1; }
      echo "Base checkpoint: $BASE_CKPT"
      EXTRA_ARGS+=(--load_from_checkpoint "$BASE_CKPT")
    fi
    ;;
  encoder_swap)
    CFG=configs/models/comet_encoder_swap.yaml
    [[ -n "${PRETRAINED:-}" ]] && EXTRA_ARGS+=(--regression_metric.init_args.pretrained_model "$PRETRAINED")
    [[ -n "${ENCODER_MODEL:-}" ]] && EXTRA_ARGS+=(--regression_metric.init_args.encoder_model "$ENCODER_MODEL")
    ;;
  *) echo "Unknown VARIANT=$VARIANT (use paragraph | encoder_swap)"; exit 1 ;;
esac

if [[ -n "${RESUME:-}" ]]; then
  echo "Resuming from: $RESUME"
  EXTRA_ARGS+=(--load_from_checkpoint "$RESUME" --resume_from_checkpoint "$RESUME")
fi

# ── 3. train ───────────────────────────────────────────────────────────────────
echo; echo "### Training ($VARIANT) ###"
srun python scripts/train_wandb.py \
    --cfg "$CFG" \
    --seed_everything "${SEED:-42}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$CKPT_DIR/$WANDB_RUN_NAME.log"

echo; echo "Done. Checkpoints under $CKPT_DIR (see wandb/<run>/checkpoints/)."
echo "Next: compare correlation-vs-length across models with scripts/eval_length_correlation.py"
