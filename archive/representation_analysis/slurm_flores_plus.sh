#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# slurm_flores_plus.sh — COMET-encoder translation-alignment study on *official*
# FLORES+ (OLDI, openlanguagedata/flores_plus), comparing the COMET encoder with
# LaBSE (and XLM-R / E5) via xSIM and xSIM++.
#
# What it runs (both on the official FLORES+ benchmark, --flores_source plus):
#   E1  cross-lingual alignment  : LASER-style xSIM retrieval error en↔xx
#                                  (+ E3 length scaling over pseudo-paragraphs)
#   E2  xSIM++ error-sensitivity : P[cos(src,tgt⁺) > cos(src,tgt⁻)] on injected
#                                  errors (number / omission / reorder / mistrans.)
#
# Encoder zoo:
#   comet:Unbabel/wmt22-comet-da      the subject (COMET encoder, layer-attn+avg pool)
#   comet:<BIO_CKPT>                  Bio-MQM-finetuned COMET (default: epoch-12 ckpt)
#   hf-mean:xlm-roberta-large         COMET's untrained backbone (control)
#   labse                             contrastive parallel-sentence aligner
#   e5                                modern contrastive text embedder
#
# All metrics go to Weights & Biases (project $WANDB_PROJECT) + results/repana/.
#
# Usage:
#   sbatch scripts/slurm_flores_plus.sh
#   BIO_CKPT=/path/model.ckpt LANGS="en de fr ru zh ja" sbatch scripts/slurm_flores_plus.sh
#
# FLORES+ is gated: you must be logged in to the HF Hub (`hf auth login`) and have
# accepted the terms at https://huggingface.co/datasets/openlanguagedata/flores_plus.
#
# Tunables (env at submit time):
#   CONDA_ENV       conda env                (default comet-bio)
#   BIO_CKPT        Bio-MQM COMET .ckpt      (default: epoch-12 4yeqp7cn/last.ckpt)
#   LANGS           space-separated ISO langs(default: en de es fr ru zh)
#   SPLIT           FLORES+ split            (default: devtest)
#   WANDB_PROJECT   W&B project              (default: comet-flores-plus)
#   WANDB_MODE      set offline if no network on the node
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=flores-plus
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
mkdir -p logs

# ── environment ───────────────────────────────────────────────────────────────
CONDA_BASE="$HOME/miniconda3"
if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
else
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-comet-bio}"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-comet-flores-plus}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME"

# FLORES+ is gated. huggingface_hub.get_token() reads $HF_HOME/token, so when
# HF_HOME is redirected to scratch the default ~/.cache/huggingface/token is
# missed and downloads fall back to anonymous (→ 401). Export HF_TOKEN explicitly
# so the datasets/hf:// download path authenticates.
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

SPLIT="${SPLIT:-devtest}"
LANGS="${LANGS:-en de es fr ru zh}"
OUT=results/flores_plus
mkdir -p "$OUT"

# ── Bio-MQM checkpoint (≥10 epochs): default to the epoch-12 last.ckpt ──────────
DEFAULT_BIO="$HOME/scratch/checkpoints/bio_mqm/comet-bio-mqm/4yeqp7cn/checkpoints/last.ckpt"
BIO_CKPT="${BIO_CKPT:-$DEFAULT_BIO}"

ENCODERS=( "comet:Unbabel/wmt22-comet-da" "hf-mean:xlm-roberta-large" "labse" "e5" )
if [[ -f "$BIO_CKPT" ]]; then
  ENCODERS=( "comet:Unbabel/wmt22-comet-da" "comet:$BIO_CKPT" "hf-mean:xlm-roberta-large" "labse" "e5" )
else
  echo "WARNING: Bio-MQM checkpoint not found at $BIO_CKPT — running without COMET-bio."
fi

echo "═══════════════════════════════════════════════════════════"
echo " Node    : $(hostname)   GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo " Job     : ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?})"
echo " FLORES+ : split=$SPLIT  langs=[$LANGS]"
echo " Bio ckpt: ${BIO_CKPT}$( [[ -f "$BIO_CKPT" ]] && echo '' || echo '  (MISSING)')"
echo " WANDB   : project=$WANDB_PROJECT mode=${WANDB_MODE:-online}"
echo "═══════════════════════════════════════════════════════════"
nvidia-smi || true

# ── confirm gated FLORES+ access before the long encoder loop ──────────────────
echo; echo "### Sanity check — official FLORES+ access (gated dataset) ###"
srun python - "$SPLIT" <<'PY'
import sys
from datasets import load_dataset
split = sys.argv[1]
ds = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split=split)
print(f"FLORES+ OK: eng_Latn/{split} = {len(ds)} rows; example id={ds[0]['id']!r}")
PY

# ── E1 + E3 — cross-lingual alignment & length scaling (xSIM) ──────────────────
# lengths 1 2 4: FLORES+ articles are short (no article spans 8 sentences, so k=8
# yields 0 parallel pseudo-docs). The runner skips any length with <2 docs anyway.
echo; echo "### E1 + E3 — xSIM alignment & length scaling on FLORES+ ###"
srun python scripts/run_retrieval.py \
  --flores_source plus --split "$SPLIT" \
  --encoders "${ENCODERS[@]}" --langs $LANGS --lengths 1 2 4 \
  --output "$OUT/e1_e3_retrieval.json" --wandb_project "$WANDB_PROJECT"

# ── E2 — xSIM++ error-sensitivity trade-off ────────────────────────────────────
echo; echo "### E2 — xSIM++ error-sensitivity on FLORES+ ###"
srun python scripts/run_error_sensitivity.py \
  --flores_source plus --split "$SPLIT" \
  --encoders "${ENCODERS[@]}" --langs $LANGS \
  --output "$OUT/e2_error_sensitivity.json" --wandb_project "$WANDB_PROJECT"

echo; echo "All FLORES+ experiments done → $OUT/  (W&B project: $WANDB_PROJECT)"
