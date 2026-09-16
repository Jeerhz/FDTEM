#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# comet_align.sh — align translations with the COMET SCORE instead of with the
# encoder's cosine similarity, on the same FLORES+ blocks as xsim.sh.
#
# The block-xSIM++ experiment retrieves with cos(embed(src), embed(candidate)).
# That measures the encoder. The metric people actually report is the encoder
# PLUS a regression head trained on human judgement, so this run swaps the
# decision rule for argmax COMET(src, candidate) and changes nothing else — same
# articles, same non-overlapping blocks, same single-error hard negatives.
#
# Reference-free checkpoints only for the score rule: at alignment time the
# reference IS the translation being retrieved, so a reference-based metric
# cannot play without being handed the answer. COMET-DA and its arms take part
# through encoder-cos:, which is the comparison they were already in.
#
# The candidate set is the reference block plus ITS OWN perturbed variants. The
# other blocks' true targets — the classic xsim distractors — are excluded on
# purpose: telling one article from another is an easy, different skill, and
# leaving it in lets a model look sensitive to the injected edit when it is only
# recognising the topic.
#
# Produces, in results/comet_align/:
#   comet_align.json        duel + shortlist accuracy per (scorer, language, k),
#                           broken down by perturbation category and by the
#                           position of the perturbed sentence in the block
#   plots/comet_align.png   the three headline curves against k
#
# Usage:
#   sbatch experiments/length_isolation/slurm/comet_align.sh
#   K_LIST="1 2 3 4 5 6" LANGS="de es fr ru" sbatch .../comet_align.sh
#   QE_CKPT=<kiwi arm ckpt> DA_CKPT=<da arm ckpt> sbatch .../comet_align.sh
#
# Tunables (env at submit time):
#   CONDA_ENV     conda env                   (default comet-bio)
#   LANGS         non-pivot ISO langs         (default de es fr ru; cased only)
#   K_LIST        block lengths               (default 1 2 3 4 5)
#   SPLITS        FLORES+ splits              (default dev devtest)
#   NEGATIVES     hard negatives per block    (default 6 — this sets the cost)
#   SHORTLIST     candidates per retrieval    (default 3 = gold + 2 of
#                 ITS OWN negatives; no other-block distractors)
#   QE_CKPT       a QE arm to add on BOTH rules
#   DA_CKPT       a DA arm to add on encoder-cos only (it has no ref-free mode)
#   WANDB_PROJECT W&B project                 (default comet-align)
#
# FLORES+ is gated: be logged in to the HF Hub and have accepted the terms at
# https://huggingface.co/datasets/openlanguagedata/flores_plus.
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=comet-align
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

CONDA_BASE="$HOME/miniconda3"
if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
else
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-comet-bio}"
fi

export WANDB_PROJECT="${WANDB_PROJECT:-comet-align}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME"
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

LANGS="${LANGS:-de es fr ru}"
K_LIST="${K_LIST:-1 2 3 4 5}"
SPLITS="${SPLITS:-dev devtest}"
NEGATIVES="${NEGATIVES:-6}"
SHORTLIST="${SHORTLIST:-3}"
OUT=results/comet_align
mkdir -p "$OUT"

# The published pair, on both decision rules, is the minimum comparison:
# CometKiwi can do both; COMET-DA can only do the cosine.
SCORERS=( "comet-score:Unbabel/wmt22-cometkiwi-da"
          "encoder-cos:Unbabel/wmt22-cometkiwi-da"
          "encoder-cos:Unbabel/wmt22-comet-da" )
[[ -n "${QE_CKPT:-}" && -f "$QE_CKPT" ]] && SCORERS+=( "comet-score:$QE_CKPT" "encoder-cos:$QE_CKPT" )
[[ -n "${DA_CKPT:-}" && -f "$DA_CKPT" ]] && SCORERS+=( "encoder-cos:$DA_CKPT" )

echo "═══════════════════════════════════════════════════════════"
echo " Node      : $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo " Task      : alignment by COMET score vs by encoder cosine"
echo " FLORES+   : splits=[$SPLITS]  langs=[$LANGS]  k=[$K_LIST]"
echo " Protocols : duel (1 own negative) + shortlist ($SHORTLIST own candidates)"
echo " Scorers   : ${SCORERS[*]}"
echo "═══════════════════════════════════════════════════════════"
nvidia-smi || true

# How many COMET forward passes this is about to cost, before it costs them.
echo; echo "### Dry run — task sizes ###"
srun python experiments/length_isolation/run_comet_align.py --dry_run \
  --langs $LANGS --k_list $K_LIST --splits $SPLITS \
  --negatives_per_block "$NEGATIVES" --shortlist_size "$SHORTLIST"

echo; echo "### Alignment ###"
srun python experiments/length_isolation/run_comet_align.py \
  --scorers "${SCORERS[@]}" \
  --langs $LANGS --k_list $K_LIST --splits $SPLITS \
  --negatives_per_block "$NEGATIVES" --shortlist_size "$SHORTLIST" \
  --batch_size "${BATCH_SIZE:-32}" \
  --emb_cache_dir "$OUT/emb_cache" \
  --output "$OUT/comet_align.json" \
  --wandb_project "$WANDB_PROJECT"

echo; echo "Done → $OUT/  (plot: $OUT/plots/comet_align.png)"
