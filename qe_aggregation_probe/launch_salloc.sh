#!/usr/bin/env bash
# Launched inside tmux: prefetch on the login node, then salloc+srun the probe.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate comet-bio
export HF_HOME=$HOME/scratch/hf_cache
mkdir -p "$HF_HOME" logs
[ -s "$HF_HOME/token" ] || cp ~/.cache/huggingface/token "$HF_HOME/token" 2>/dev/null
cd ~/FDTEM/qe_aggregation_probe

echo "══ prefetch WMT24++ ($(date))"
python -c "from datasets import load_dataset
for lp in ['en-de_DE','en-zh_CN','en-es_MX','en-fr_FR']:
    load_dataset('google/wmt24pp', lp, split='train'); print('  ok', lp)"

echo "══ prefetch COMET checkpoints ($(date))"
python -c "from comet import download_model

download_model('Unbabel/wmt22-comet-da'); print('  ok comet')"

echo "══ salloc ($(date))"
exec salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
  srun python -u run.py --config config.yaml --perturb_backend rule --metrics cometkiwi comet
