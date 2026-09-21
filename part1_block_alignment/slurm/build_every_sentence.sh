#!/usr/bin/env bash
# Build the every-sentence dataset on CPU (perturb_every_sentence.py): FLORES windows,
# distractors perturbed in every sentence, English source token counts -> data/.
# Needs data/flores_<source>_<split>.json (load_flores, first step of build_pools.sh).
#
#   sbatch part1_block_alignment/slurm/build_every_sentence.sh
#   LANGS="de es fr ru" K_LIST="1 2 3 4 5" N_DISTRACTORS=6 sbatch part1_block_alignment/slurm/build_every_sentence.sh
#
# Tunables: SOURCE (plus|raw), LANGS, K_LIST (windows are max(K_LIST) sentences), SPLITS,
#           N_DISTRACTORS (per window, at most), BACKEND (spacy|heuristic|auto).
#SBATCH --job-name=build-every-sentence
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cpu_devel
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
source common/cluster_env.sh

SOURCE="${SOURCE:-plus}"
LANGS="${LANGS:-de es fr ru}"
K_LIST="${K_LIST:-1 2 3 4 5}"
SPLITS="${SPLITS:-dev devtest}"
N_DISTRACTORS="${N_DISTRACTORS:-6}"
BACKEND="${BACKEND:-spacy}"

srun python -m part1_block_alignment.perturb_every_sentence --source "$SOURCE" --splits $SPLITS \
  --langs $LANGS --k_list $K_LIST --n_distractors "$N_DISTRACTORS" --backend "$BACKEND"
