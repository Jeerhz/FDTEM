#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# slurm_retrain_comet.sh — COMET retraining experiments (three arms).
#
#   VARIANT=paragraph     continue Unbabel/wmt22-comet-da on the length-balanced
#                         paragraph mix (isolates the DATA effect)
#   VARIANT=encoder_swap  retrain from raw encoder weights on the same mix
#                         (isolates the ENCODER effect; default
#                         microsoft/infoxlm-large — override with $PRETRAINED /
#                         $ENCODER_MODEL, e.g. PRETRAINED=xlm-roberta-large for
#                         the controlled from-scratch baseline)
#   VARIANT=mix           continue wmt22-comet-da on ONE sentence-fraction mix
#                         (MIX=frac000 … frac100, built by
#                         scripts/make_sentence_mix.py — the "long-only +
#                         graded sentence data" sweep)
#
# FROZEN=1 (any variant) keeps the encoder frozen for the whole run (only the
# regression head trains); default is the config's schedule (0.3 epochs frozen,
# then full finetuning).
#
# Builds the paragraph data (scripts/prepare_paragraph_data.py) and, for
# VARIANT=mix, the mixes (scripts/make_sentence_mix.py) if missing, then trains
# via scripts/train_wandb.py with W&B logging.
#
# Usage:
#   VARIANT=paragraph    sbatch scripts/slurm_retrain_comet.sh
#   VARIANT=encoder_swap sbatch scripts/slurm_retrain_comet.sh
#   VARIANT=encoder_swap PRETRAINED=xlm-roberta-large RUN_NAME=xlmr-scratch \
#       sbatch scripts/slurm_retrain_comet.sh
#   # sentence-fraction sweep, both encoder regimes:
#   for MIX in frac000 frac010 frac020 frac040 frac060 frac080 frac100; do
#     for FROZEN in 0 1; do
#       VARIANT=mix MIX=$MIX FROZEN=$FROZEN sbatch scripts/slurm_retrain_comet.sh
#     done
#   done
#   # resume after timeout:
#   VARIANT=paragraph RESUME=<ckpt> WANDB_RUN_ID=<id> sbatch scripts/slurm_retrain_comet.sh
#
# Tunables (env at submit time):
#   VARIANT         paragraph | encoder_swap | mix  (default paragraph)
#   MIX             mix subdir (VARIANT=mix)        (e.g. frac020)
#   MIX_DIR         mixes root                      (default $DATA_DIR/mixes)
#   FROZEN          1 = encoder frozen all run      (default 0)
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
MIX_DIR="${MIX_DIR:-$DATA_DIR/mixes}"
FROZEN="${FROZEN:-0}"

ARM_NAME="$VARIANT"
if [[ "$VARIANT" == "mix" ]]; then
  [[ -n "${MIX:-}" ]] || { echo "VARIANT=mix requires MIX=fracNNN"; exit 1; }
  ARM_NAME="mix-$MIX"
fi
[[ "$FROZEN" == "1" ]] && ARM_NAME="$ARM_NAME-frozen"

CKPT_DIR="$HOME/scratch/checkpoints/retrain/$ARM_NAME"
mkdir -p "$CKPT_DIR"

export WANDB_PROJECT="${WANDB_PROJECT:-comet-retrain}"
export WANDB_RUN_NAME="${RUN_NAME:-${ARM_NAME}-$(date +%Y%m%d-%H%M)}"
export WANDB_TAGS="retrain,${ARM_NAME}"
export WANDB_SAVE_DIR="$CKPT_DIR"
if ! wandb status &>/dev/null; then
  echo "W&B not configured — setting WANDB_MODE=offline"
  export WANDB_MODE=offline
fi

echo "═══════════════════════════════════════════════════════════"
echo " Node     : $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo " Variant  : $VARIANT (arm: $ARM_NAME, frozen=$FROZEN)"
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

# mixes need the unbalanced pools ({lp}_trainpool.csv, added with the sweep) —
# older data dirs must be regenerated once
if [[ "$VARIANT" == "mix" ]]; then
  if ! ls "$DATA_DIR"/*_trainpool.csv &>/dev/null; then
    echo "No *_trainpool.csv in $DATA_DIR — rebuilding paragraph data"
    srun python scripts/prepare_paragraph_data.py \
        --output_dir "$DATA_DIR" --lang_pairs $LANG_PAIRS \
        --wandb_project "$WANDB_PROJECT"
  fi
  if [[ ! -f "$MIX_DIR/$MIX/all_train.csv" ]]; then
    echo; echo "### Building sentence-fraction mixes ###"
    srun python scripts/make_sentence_mix.py --data_dir "$DATA_DIR" --out_dir "$MIX_DIR"
  fi
  [[ -f "$MIX_DIR/$MIX/all_train.csv" ]] || { echo "ERROR: mix $MIX not found in $MIX_DIR"; exit 1; }
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
  mix)
    # same recipe as `paragraph` (continue wmt22-comet-da), only the train CSVs
    # point at one sentence-fraction mix; validation stays the shared val set
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
  *) echo "Unknown VARIANT=$VARIANT (use paragraph | encoder_swap | mix)"; exit 1 ;;
esac

# ── 2b. derived config (mix train_data / frozen encoder) ───────────────────────
# Overrides go through a generated YAML rather than CLI flags: jsonargparse
# 3.13.1 is unreliable for list-valued init_args, and the generated file doubles
# as the exact record of what this run trained on ($CKPT_DIR/config.yaml).
if [[ "$VARIANT" == "mix" || "$FROZEN" == "1" ]]; then
  GEN_CFG="$CKPT_DIR/config.yaml"
  MIX_PATH=""
  [[ "$VARIANT" == "mix" ]] && MIX_PATH="$MIX_DIR/$MIX"
  python - "$CFG" "$GEN_CFG" "$MIX_PATH" "$FROZEN" <<'EOF'
import sys, glob, os
import yaml

base_cfg, out_cfg, mix_path, frozen = sys.argv[1:5]
cfg = yaml.safe_load(open(base_cfg))
base_dir = os.path.dirname(os.path.abspath(base_cfg))

# sub-config references are relative to the base config's directory — make them
# absolute so the generated file can live anywhere
for key in ("trainer", "early_stopping", "model_checkpoint"):
    v = cfg.get(key)
    if isinstance(v, str):
        cfg[key] = os.path.normpath(os.path.join(base_dir, v))

metric_key = next(k for k, v in cfg.items()
                  if isinstance(v, dict) and "class_path" in v)
init = cfg[metric_key]["init_args"]

if mix_path:
    files = sorted(f for f in glob.glob(os.path.join(mix_path, "*_train.csv"))
                   if os.path.basename(f) != "all_train.csv")
    if not files:
        sys.exit(f"ERROR: no per-LP *_train.csv in {mix_path}")
    init["train_data"] = files
    print(f"train_data → {len(files)} files from {mix_path}")

if frozen == "1":
    # comet unfreezes once epoch_nr >= nr_frozen_epochs — this never fires
    init["nr_frozen_epochs"] = 1000
    init["keep_embeddings_frozen"] = True
    print("encoder frozen for the whole run (nr_frozen_epochs=1000)")

os.makedirs(os.path.dirname(os.path.abspath(out_cfg)), exist_ok=True)
with open(out_cfg, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
print(f"generated config → {out_cfg}")
EOF
  CFG="$GEN_CFG"
fi

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
