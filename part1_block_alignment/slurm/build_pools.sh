#!/usr/bin/env bash
# Build the part-1 data on CPU: FLORES -> blocks -> candidate pools (data/).
#
#   sbatch part1_block_alignment/slurm/build_pools.sh
#   LANGS="de es fr ru zh" K_LIST="1 2 3 4 5" BACKEND=spacy sbatch part1_block_alignment/slurm/build_pools.sh
#
# Tunables: SOURCE (plus|raw), LANGS, K_LIST, SPLITS, DIRECTION (en2xx|xx2en),
#           VARIANTS (negatives per position and category), BACKEND (spacy|heuristic|auto).
# FLORES+ is gated: `hf auth login` once and accept the terms at
# https://huggingface.co/datasets/openlanguagedata/flores_plus. The spacy backend
# needs `python -m spacy download <lang>_core_*_sm` for every language.
#SBATCH --job-name=build-pools
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cpu_devel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
source common/cluster_env.sh

SOURCE="${SOURCE:-plus}"
LANGS="${LANGS:-de es fr ru}"
K_LIST="${K_LIST:-1 2 3 4 5}"
SPLITS="${SPLITS:-dev devtest}"
DIRECTION="${DIRECTION:-en2xx}"
VARIANTS="${VARIANTS:-2}"
BACKEND="${BACKEND:-spacy}"

srun python -m part1_block_alignment.load_flores --source "$SOURCE" --langs en $LANGS --splits $SPLITS
srun python -m part1_block_alignment.build_blocks --source "$SOURCE" --splits $SPLITS --k_list $K_LIST
srun python -m part1_block_alignment.perturb --source "$SOURCE" --splits $SPLITS --langs $LANGS \
  --direction "$DIRECTION" --k_list $K_LIST --backend "$BACKEND" --variants_per_position "$VARIANTS"
