#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# slurm_block_xsim.sh — xSIM++ on concatenated blocks (Experiment B).
#
# Query = source block; pool = every true target block + hard negatives built by
# applying ONE xSIM++ perturbation (causality / entity / number) to ONE sentence
# of a block. Sweeps k ∈ {2,3,4,5} to measure how single-sentence error
# sensitivity dilutes with block length, across the COMET encoder, its raw XLM-R
# backbone, and the dedicated aligners LaBSE / E5.
#
# Produces, in results/block_xsim/:
#   plots/xsim_vs_xsimpp.png        xsim vs xsim++ error against block length
#   plots/detection_vs_length.png   P[gold closer than perturbed] vs k
#   plots/error_by_category.png     which perturbation category fools each model
#   plots/detection_by_position.png sensitivity vs where the error sits
#   block_xsim.json                 all numbers
# and logs to W&B (project $WANDB_PROJECT).
#
# Usage:
#   sbatch scripts/slurm_block_xsim.sh
#   LANGS="de es fr ru zh ja" K_LIST="2 3 4 5 6" sbatch scripts/slurm_block_xsim.sh
#
# Tunables (env at submit time):
#   CONDA_ENV      conda env                (default comet-bio)
#   BIO_CKPT       Bio-MQM COMET .ckpt      (default: epoch-12 4yeqp7cn/last.ckpt)
#   RETRAIN_CKPT   optional retrained ckpt to add to the zoo (Experiment A output)
#   LANGS          non-pivot ISO langs      (default: de es fr ru; cased only)
#   K_LIST         block lengths            (default: 2 3 4 5)
#   SPLITS         FLORES+ splits to pool   (default: dev devtest)
#   DIRECTION      en2xx | xx2en            (default: en2xx)
#   VARIANTS       hard negatives per (block, position, category) (default 2)
#   WANDB_PROJECT  W&B project              (default comet-block-xsim)
#
# FLORES+ is gated: be logged in to the HF Hub and have accepted the terms at
# https://huggingface.co/datasets/openlanguagedata/flores_plus.
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=block-xsim
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

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

export WANDB_PROJECT="${WANDB_PROJECT:-comet-block-xsim}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME"
# gated FLORES+ auth under redirected HF_HOME (token lives in the default location)
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

LANGS="${LANGS:-de es fr ru}"
K_LIST="${K_LIST:-2 3 4 5}"
SPLITS="${SPLITS:-dev devtest}"
DIRECTION="${DIRECTION:-en2xx}"
VARIANTS="${VARIANTS:-2}"
OUT=results/block_xsim
mkdir -p "$OUT"

# ── encoder zoo ────────────────────────────────────────────────────────────────
DEFAULT_BIO="$HOME/scratch/checkpoints/bio_mqm/comet-bio-mqm/4yeqp7cn/checkpoints/last.ckpt"
BIO_CKPT="${BIO_CKPT:-$DEFAULT_BIO}"
ENCODERS=( "comet:Unbabel/wmt22-comet-da" "hf-mean:xlm-roberta-large" "labse" "e5" )
[[ -f "$BIO_CKPT" ]] && ENCODERS+=( "comet:$BIO_CKPT" )
[[ -n "${RETRAIN_CKPT:-}" && -f "$RETRAIN_CKPT" ]] && ENCODERS+=( "comet:$RETRAIN_CKPT" )

echo "═══════════════════════════════════════════════════════════"
echo " Node      : $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo " Task      : block xSIM++  direction=$DIRECTION"
echo " FLORES+   : splits=[$SPLITS]  langs=[$LANGS]  k=[$K_LIST]  variants=$VARIANTS"
echo " Encoders  : ${ENCODERS[*]}"
echo " W&B       : project=$WANDB_PROJECT mode=${WANDB_MODE:-online}"
echo "═══════════════════════════════════════════════════════════"
nvidia-smi || true

# ── sanity check — gated FLORES+ access ─────────────────────────────────────────
echo; echo "### Sanity check — FLORES+ access ###"
srun python - <<'PY'
from datasets import load_dataset
ds = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
print(f"FLORES+ OK: eng_Latn/devtest = {len(ds)} rows")
PY

# ── run ────────────────────────────────────────────────────────────────────────
echo; echo "### Block xSIM++ ###"
srun python scripts/run_block_xsim.py \
  --encoders "${ENCODERS[@]}" \
  --langs $LANGS --k_list $K_LIST --splits $SPLITS \
  --direction "$DIRECTION" --variants_per_position "$VARIANTS" \
  --flores_source plus \
  --output "$OUT/block_xsim.json" --wandb_project "$WANDB_PROJECT"

echo; echo "Done → $OUT/  (headline plot: $OUT/plots/xsim_vs_xsimpp.png)"
