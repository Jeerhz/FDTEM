#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# finetune_bio_mqm.sh
#
# End-to-end pipeline:
#   1. Download & preprocess Bio-MQM data
#   2. Download the base COMET checkpoint (wmt22-comet-da)
#   3. Finetune via comet-train with W&B logging
#   4. Save the best checkpoint path for upload
#
# Usage:
#   bash scripts/finetune_bio_mqm.sh [--gpus 2] [--run_name my_run]
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── parse optional CLI flags  ──────────────────────────────────────────────────
GPUS=1
RUN_NAME="comet-bio-mqm-$(date +%Y%m%d-%H%M)"
SEED=42
OUTPUT_DIR="$HOME/scratch/bio_mqm"
CHECKPOINT_DIR="$HOME/scratch/checkpoints/bio_mqm"
RESUME_CKPT=""      # set via --resume <path>
WANDB_RUN_ID_ARG="" # set via --wandb_run_id <id>

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)          GPUS="$2";             shift 2 ;;
    --run_name)      RUN_NAME="$2";         shift 2 ;;
    --output_dir)    OUTPUT_DIR="$2";       shift 2 ;;
    --checkpoint_dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --resume)        RESUME_CKPT="$2";     shift 2 ;;
    --wandb_run_id)  WANDB_RUN_ID_ARG="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

echo "═══════════════════════════════════════════════════════════"
echo " COMET Bio-MQM finetuning"
echo "  GPUs        : $GPUS"
echo "  Run name    : $RUN_NAME"
echo "  Data dir    : $OUTPUT_DIR"
echo "  Checkpoints : $CHECKPOINT_DIR"
echo "═══════════════════════════════════════════════════════════"

# ── 0. ensure we are at the project root ─────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ── 1. prepare Bio-MQM data ───────────────────────────────────────────────────
echo ""
echo "── Step 1: Preparing Bio-MQM data ──────────────────────────"
python scripts/prepare_bio_mqm_data.py \
    --output_dir "$OUTPUT_DIR" \
    --write_combined

# ── 2. download base COMET checkpoint ────────────────────────────────────────
echo ""
echo "── Step 2: Downloading base model checkpoint ───────────────"
# download_model() already returns the full .ckpt path directly — no walking needed.
# Redirect stderr so tqdm/logging don't pollute the captured stdout.
BASE_CKPT=$(python - 2>/dev/null <<'EOF'
from comet import download_model
print(download_model("Unbabel/wmt22-comet-da"))
EOF
)

if [[ -z "$BASE_CKPT" || ! -f "$BASE_CKPT" ]]; then
  echo "ERROR: Could not resolve base checkpoint (got: '${BASE_CKPT}')."
  echo "       Try: python -c \"from comet import download_model; print(download_model('Unbabel/wmt22-comet-da'))\""
  exit 1
fi
echo "Base checkpoint: $BASE_CKPT"

# ── 3. patch trainer.yaml for the requested GPU count ────────────────────────
# We use a temp copy so we don't dirty the committed config.
TRAINER_TMP="$(mktemp /tmp/trainer_XXXXXX.yaml)"
sed "s/^  devices:.*/  devices: $GPUS/" configs/trainer.yaml > "$TRAINER_TMP"

# ── 4. patch model config checkpoint dir ─────────────────────────────────────
mkdir -p "$CHECKPOINT_DIR"
MODEL_CFG_TMP="$(mktemp /tmp/bio_mqm_cfg_XXXXXX.yaml)"
sed "s|model_checkpoint: ../model_checkpoint.yaml|model_checkpoint: ../model_checkpoint.yaml|" \
    configs/models/bio_mqm_finetune.yaml > "$MODEL_CFG_TMP"

# ── 5. W&B env vars ───────────────────────────────────────────────────────────
export WANDB_PROJECT="${WANDB_PROJECT:-comet-bio-mqm}"
export WANDB_RUN_NAME="$RUN_NAME"
export WANDB_TAGS="bio-mqm,comet,finetuning"
# Disable W&B if not logged in (set WANDB_MODE=disabled to always skip)
if ! wandb status &>/dev/null; then
  echo "W&B not configured — setting WANDB_MODE=offline"
  export WANDB_MODE=offline
fi

# ── 6. run training ───────────────────────────────────────────────────────────
# We use scripts/train_wandb.py instead of comet-train because jsonargparse
# 3.13.1 cannot instantiate WandbLogger from YAML (Union[Logger,bool,None] bug).
export WANDB_SAVE_DIR="$CHECKPOINT_DIR"
[[ -n "$WANDB_RUN_ID_ARG" ]] && export WANDB_RUN_ID="$WANDB_RUN_ID_ARG"

RESUME_ARGS=()
if [[ -n "$RESUME_CKPT" ]]; then
  echo "Resuming from checkpoint: $RESUME_CKPT"
  RESUME_ARGS+=(--load_from_checkpoint "$RESUME_CKPT" --resume_from_checkpoint "$RESUME_CKPT")
else
  RESUME_ARGS+=(--load_from_checkpoint "$BASE_CKPT")
fi

echo ""
echo "── Step 3: Running training with W&B logging ───────────────"
python scripts/train_wandb.py \
    --cfg configs/models/bio_mqm_finetune.yaml \
    "${RESUME_ARGS[@]}" \
    --seed_everything "$SEED" \
    2>&1 | tee "$CHECKPOINT_DIR/${RUN_NAME}.log"

# ── 7. locate best checkpoint ────────────────────────────────────────────────
BEST_CKPT=$(find . -name "*.ckpt" -newer "$BASE_CKPT" \
    | grep -v last | sort -t= -k2 -n | tail -1)
if [[ -z "$BEST_CKPT" ]]; then
  BEST_CKPT=$(find . -name "*.ckpt" -newer "$BASE_CKPT" | tail -1)
fi

echo ""
echo "── Done ──────────────────────────────────────────────────────"
echo "Best checkpoint : $BEST_CKPT"
echo ""
echo "To upload to Hugging Face:"
echo "  python scripts/upload_to_huggingface.py \\"
echo "      --checkpoint  \"$BEST_CKPT\" \\"
echo "      --repo_id     \"<your-hf-username>/comet-bio-mqm\" \\"
echo "      --run_name    \"$RUN_NAME\""

# Save best ckpt path for CI / downstream scripts
echo "$BEST_CKPT" > "$CHECKPOINT_DIR/best_checkpoint.txt"
