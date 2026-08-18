#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# train.sh — one arm of the length-composition sweep.
#
# Continues a public base metric on ONE sentence-fraction mix:
#   MODEL=da  → Unbabel/wmt22-comet-da     (ref-based, RegressionMetric)
#   MODEL=qe  → Unbabel/wmt22-cometkiwi-da (ref-free,  UnifiedMetric)
#   MIX       → frac000 … frac100, frac000nat, frac000agg
#   FROZEN=1  → encoder frozen for the whole run (head-only), default 0
#
# The full grid is 2 × 9 × 2 = 36 arms; submit it with slurm/launch_sweep.sh.
#
# ── What changed on 2026-08-18 (read this before comparing to older runs) ─────
# Arms used to receive one CSV per language pair as `train_data`. COMET selects
# the epoch's file itself (`train_data[epoch % len(train_data)]`) and Lightning
# never rebuilt the loader (`reload_dataloaders_every_n_epochs: 0`), so every
# arm trained on `train_data[0]` alone — between 205 and 6,940 rows depending on
# the arm, instead of the 24,000-row mix — and the arms were not comparable.
# Arms now train on the mix's single `all_train.csv`, so one epoch is the whole
# mix in every arm, and the budget below is identical across the grid.
# Checkpoints from before that date are not comparable with these.
#
# Usage:
#   MODEL=da MIX=frac040 FROZEN=0 sbatch experiments/length_training/slurm/train.sh
#   MODEL=qe MIX=frac000 FROZEN=1 sbatch experiments/length_training/slurm/train.sh
#   # resume after a timeout:
#   MODEL=da MIX=frac040 RESUME=<ckpt> WANDB_RUN_ID=<id> sbatch .../train.sh
#
# Tunables (env at submit time):
#   MODEL          da | qe                          (default da)
#   MIX            mix subdir                       (required)
#   FROZEN         1 = encoder frozen all run       (default 0)
#   MAX_EPOCHS     passes over the mix (the budget) (default 6)
#   MAX_STEPS      hard optimizer-step cap          (default -1 = unbounded)
#   PATIENCE       early-stopping patience (checks) (default 3)
#   VAL_INTERVAL   validations per epoch as a frac  (default 1.0 = once/epoch)
#   DATA_DIR       WMT pools                        (default ~/scratch/wmt_length_data)
#   MIX_DIR        mixes root                       (default $DATA_DIR/mixes)
#   VAL_FILES      validation CSVs                  (default $DATA_DIR/all_val.csv)
#   CKPT_ROOT      checkpoints subdir               (default retrain-wmt)
#   SEED           training seed                    (default 42)
#   CONDA_ENV      conda env                        (default comet-bio)
#   RUN_NAME       W&B run name                     (default <arm>-<date>)
#   WANDB_PROJECT  W&B project                      (default comet-retrain-wmt)
#   RESUME         checkpoint to resume from
#
# The Bio-MQM paragraph / encoder-swap variants that used to live here moved to
# archive/bio_paragraph_pipeline/ along with their data builders.
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=comet-retrain
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

EXP=experiments/length_training

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

MODEL="${MODEL:-da}"
DATA_DIR="${DATA_DIR:-$HOME/scratch/wmt_length_data}"
MIX_DIR="${MIX_DIR:-$DATA_DIR/mixes}"
FROZEN="${FROZEN:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-6}"
MAX_STEPS="${MAX_STEPS:--1}"
PATIENCE="${PATIENCE:-3}"
VAL_INTERVAL="${VAL_INTERVAL:-1.0}"
VAL_FILES="${VAL_FILES:-$DATA_DIR/all_val.csv}"

[[ -n "${MIX:-}" ]] || { echo "MIX=fracNNN is required (see --help in the header)"; exit 1; }

case "$MODEL" in
  da) CFG="$EXP/configs/comet_da.yaml"; BASE_ID="Unbabel/wmt22-comet-da";     ARM_NAME="mix-$MIX" ;;
  qe) CFG="$EXP/configs/comet_qe.yaml"; BASE_ID="Unbabel/wmt22-cometkiwi-da"; ARM_NAME="kiwi-mix-$MIX" ;;
  *)  echo "Unknown MODEL=$MODEL (use da | qe)"; exit 1 ;;
esac
[[ "$FROZEN" == "1" ]] && ARM_NAME="$ARM_NAME-frozen"

CKPT_DIR="$HOME/scratch/checkpoints/${CKPT_ROOT:-retrain-wmt}/$ARM_NAME"
mkdir -p "$CKPT_DIR"

# A fresh run must not leave an earlier sweep's runs in the arm directory:
# checkpoint selection takes the best val_kendall across everything it finds, so
# a superseded run would silently be the one evaluated. Rename, never delete.
if [[ -z "${RESUME:-}" && "${KEEP_PREVIOUS_RUNS:-0}" != "1" ]]; then
  if compgen -G "$CKPT_DIR/*/*/checkpoints" >/dev/null; then
    OLD="$CKPT_DIR.superseded-$(date +%Y%m%d-%H%M%S)"
    echo "Moving previous runs out of the way: $CKPT_DIR -> $OLD"
    mv "$CKPT_DIR" "$OLD"
    mkdir -p "$CKPT_DIR"
  fi
fi

export WANDB_PROJECT="${WANDB_PROJECT:-comet-retrain-wmt}"
export WANDB_RUN_NAME="${RUN_NAME:-${ARM_NAME}-$(date +%Y%m%d-%H%M)}"
export WANDB_TAGS="retrain-wmt,${ARM_NAME},budget${MAX_EPOCHS}ep"
export WANDB_SAVE_DIR="$CKPT_DIR"
if ! wandb status &>/dev/null; then
  echo "W&B not configured — setting WANDB_MODE=offline"
  export WANDB_MODE=offline
fi

echo "═══════════════════════════════════════════════════════════"
echo " Node     : $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo " Arm      : $ARM_NAME (model=$MODEL, mix=$MIX, frozen=$FROZEN)"
echo " Base     : $BASE_ID"
echo " Data     : $MIX_DIR/$MIX"
echo " Budget   : max_epochs=$MAX_EPOCHS max_steps=$MAX_STEPS patience=$PATIENCE"
echo " Ckpts    : $CKPT_DIR"
echo " W&B      : project=$WANDB_PROJECT run=$WANDB_RUN_NAME mode=${WANDB_MODE:-online}"
echo "═══════════════════════════════════════════════════════════"
nvidia-smi || true

# ── 1. data ───────────────────────────────────────────────────────────────────
if [[ ! -f "$DATA_DIR/all_train.csv" ]]; then
  echo "ERROR: no pools in $DATA_DIR — build them first:"
  echo "  python $EXP/prepare_data.py --output_dir $DATA_DIR"
  exit 1
fi
if [[ ! -f "$MIX_DIR/$MIX/all_train.csv" ]]; then
  echo; echo "### Building sentence-fraction mixes ###"
  srun python "$EXP/make_mixtures.py" --data_dir "$DATA_DIR" --out_dir "$MIX_DIR"
fi
[[ -f "$MIX_DIR/$MIX/all_train.csv" ]] || { echo "ERROR: mix $MIX not found in $MIX_DIR"; exit 1; }

# ── 2. base checkpoint ────────────────────────────────────────────────────────
EXTRA_ARGS=()
if [[ -z "${RESUME:-}" ]]; then
  BASE_CKPT=$(BASE_ID="$BASE_ID" python - 2>/dev/null <<'EOF'
import os
from comet import download_model
print(download_model(os.environ["BASE_ID"]))
EOF
)
  [[ -f "$BASE_CKPT" ]] || { echo "ERROR: could not download $BASE_ID"; exit 1; }
  echo "Base checkpoint: $BASE_CKPT"
  EXTRA_ARGS+=(--load_from_checkpoint "$BASE_CKPT")
fi

# ── 3. per-arm config ─────────────────────────────────────────────────────────
# Overrides go through a generated YAML rather than CLI flags: jsonargparse
# 3.13.1 is unreliable for list-valued init_args, and the generated file doubles
# as the exact record of what this run trained on ($CKPT_DIR/config.yaml).
GEN_CFG="$CKPT_DIR/config.yaml"
FROZEN_FLAG=()
[[ "$FROZEN" == "1" ]] && FROZEN_FLAG+=(--frozen)
python "$EXP/make_config.py" \
    --base_cfg "$CFG" \
    --out "$GEN_CFG" \
    --mix_dir "$MIX_DIR/$MIX" \
    --val_files $VAL_FILES \
    --max_epochs "$MAX_EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --val_check_interval "$VAL_INTERVAL" \
    --patience "$PATIENCE" \
    --save_top_k 1 \
    "${FROZEN_FLAG[@]}"
CFG="$GEN_CFG"

if [[ -n "${RESUME:-}" ]]; then
  echo "Resuming from: $RESUME"
  EXTRA_ARGS+=(--load_from_checkpoint "$RESUME" --resume_from_checkpoint "$RESUME")
fi

# ── 4. train ──────────────────────────────────────────────────────────────────
echo; echo "### Training $ARM_NAME ###"
srun python "$EXP/train.py" \
    --cfg "$CFG" \
    --seed_everything "${SEED:-42}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$CKPT_DIR/$WANDB_RUN_NAME.log"

# Guard against the failure this script was rewritten to prevent: COMET logs the
# file it loads each time the train dataloader is built. With a single-file
# train_data there must be exactly one such line, naming all_train.csv.
LOADS=$(grep -c "^.*Loading .*\.csv\.$" "$CKPT_DIR/$WANDB_RUN_NAME.log" || true)
echo "Train files loaded during the run: $LOADS"
grep "Loading .*\.csv\.$" "$CKPT_DIR/$WANDB_RUN_NAME.log" | sort -u || true

# comet.load_from_checkpoint requires <run-dir>/hparams.yaml, which the
# WandbLogger layout does not write — generate it from the checkpoint itself
# so evaluation scripts can load every arm without manual repair.
srun python - "$CKPT_DIR" <<'EOF'
import glob, os, sys, yaml, torch
for ck in glob.glob(os.path.join(sys.argv[1], "*", "*", "checkpoints", "last.ckpt")):
    run_dir = os.path.dirname(os.path.dirname(ck))
    hp_file = os.path.join(run_dir, "hparams.yaml")
    if os.path.exists(hp_file):
        continue
    c = torch.load(ck, map_location="cpu", weights_only=False, mmap=True)
    hp = {}
    for k, v in dict(c["hyper_parameters"]).items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        try:
            yaml.safe_dump({k: v}); hp[k] = v
        except Exception:
            hp[k] = str(v)
    hp.setdefault("class_identifier",
                  "unified_metric" if "input_segments" in hp else "regression_metric")
    with open(hp_file, "w") as fh:
        yaml.safe_dump(hp, fh, sort_keys=False)
    print(f"wrote {hp_file}")
EOF

echo; echo "Done. Checkpoints under $CKPT_DIR."
echo "Next: evaluate with"
echo "  sbatch $EXP/slurm/eval_correlation.sh   # validation + held-out, per k"
echo "  sbatch $EXP/slurm/eval_metadoceval.sh   # discourse-error detection"
