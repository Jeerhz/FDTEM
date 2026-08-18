#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# slurm_matched_core.sh — matched-core length-sensitivity suite (Q1 + Q2 + Q3).
#
# The same CORE (a FLORES+ sentence pair + its single-perturbation negatives)
# is planted in fillers of total length L ∈ {60,120,240,480} reference tokens,
# four filler types φ (inert / neutral / distractor / natural — the dilution
# gradient) and three core positions π (start / mid / end). Intra-item design:
# any performance change is attributable to length, not content.
#
# Step 1 — run_matched_core.py (frozen encoder zoo = arm A₀):
#   Q1: LSI (Δ detection per doubling of L, within-item slopes + bootstrap CI),
#       L50; Q2: detection by error category and by core position π;
#   geometry: core-retention cosine cos(x(L), bare core).
# Step 2 — run_matched_core_layers.py (XLM-R, every hidden layer):
#   Q3: at which layer the core signal dies, and how the input embedding
#   drifts from the bare-core embedding, per L and φ.
#
# Produces results/matched_core/{matched_core.json, layer_probe.json, plots/}.
#
# Usage:
#   sbatch experiments/length_isolation/slurm/matched_core.sh
#
# Tunables (env at submit time):
#   CONDA_ENV      conda env                (default comet-bio)
#   BIO_CKPT       Bio-MQM COMET .ckpt      (default: epoch-12 4yeqp7cn/last.ckpt)
#   LANGS          non-pivot ISO langs      (default: de es fr ru)
#   N_ITEMS        cores per language       (default 120)
#   WANDB_PROJECT  W&B project              (default comet-matched-core)
#
# FLORES+ is gated: be logged in to the HF Hub and have accepted the terms.
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=matched-core
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

export WANDB_PROJECT="${WANDB_PROJECT:-comet-matched-core}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME"
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

LANGS="${LANGS:-de es fr ru}"
N_ITEMS="${N_ITEMS:-120}"
OUT=results/matched_core
mkdir -p "$OUT"

# ── encoder zoo ────────────────────────────────────────────────────────────────
DEFAULT_BIO="$HOME/scratch/checkpoints/bio_mqm/comet-bio-mqm/4yeqp7cn/checkpoints/last.ckpt"
BIO_CKPT="${BIO_CKPT:-$DEFAULT_BIO}"
ENCODERS=( "comet:Unbabel/wmt22-comet-da" "hf-mean:xlm-roberta-large" "labse" "e5" )
[[ -f "$BIO_CKPT" ]] && ENCODERS+=( "comet:$BIO_CKPT" )

echo "═══════════════════════════════════════════════════════════"
echo " Node      : $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo " Task      : matched-core length sensitivity (Q1+Q2), layer probe (Q3)"
echo " Grid      : L={60,120,240,480} ref tokens · φ=4 · π=3 · items=$N_ITEMS/lang"
echo " Langs     : [$LANGS]   Encoders: ${ENCODERS[*]}"
echo " W&B       : project=$WANDB_PROJECT mode=${WANDB_MODE:-online}"
echo "═══════════════════════════════════════════════════════════"
nvidia-smi || true

echo; echo "### Sanity check — FLORES+ access ###"
srun python - <<'PY'
from datasets import load_dataset
ds = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split="devtest")
print(f"FLORES+ OK: eng_Latn/devtest = {len(ds)} rows")
PY

echo; echo "### Step 1/2 — matched-core eval (encoder zoo) ###"
srun python experiments/length_isolation/run_matched_core.py \
  --encoders "${ENCODERS[@]}" \
  --langs $LANGS --n_items "$N_ITEMS" \
  --flores_source plus \
  --output "$OUT/matched_core.json" --wandb_project "$WANDB_PROJECT"

echo; echo "### Step 2/2 — layer-wise probe (Q3, XLM-R) ###"
srun python experiments/length_isolation/run_matched_core_layers.py \
  --model xlm-roberta-large --langs de fr \
  --flores_source plus \
  --output "$OUT/layer_probe.json"

echo; echo "Done → $OUT/  (headline: $OUT/plots/detection_vs_length.png, lsi_bars.png)"
