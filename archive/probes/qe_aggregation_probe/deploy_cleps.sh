#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# deploy_cleps.sh — sync qe_aggregation_probe to the Cleps cluster and submit
# the sbatch job. Run from anywhere on the LOCAL machine (needs Inria VPN or
# on-site network so that `ssh cleps` works).
#
#     ./deploy_cleps.sh                     # sync + auto-setup + sbatch
#     ./deploy_cleps.sh --metrics cometkiwi # extra args forwarded to run.py
#     CLEPS_HOST=abensale@cleps.paris.inria.fr ./deploy_cleps.sh
#
# What it does on the cluster:
#   1. rsync this package into $REMOTE_DIR/qe_aggregation_probe
#   2. ensure the conda env has datasets/openai/sentencepiece (login node has
#      internet); warn if no HF token (wmt22-cometkiwi-da is license-gated)
#   3. prefetch the WMT24++ subsets on the login node (small, avoids surprises
#      if a compute node has no internet)
#   4. pick the perturbation backend: llm if OPENAI_API_KEY is available
#      (locally exported or in the remote ~/.bashrc), else rule + warning
#   5. sbatch slurm_agg_probe.sh and print the job id + queue state
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HOST="${CLEPS_HOST:-cleps}"
REMOTE_DIR="${CLEPS_REMOTE_DIR:-FDTEM}"          # relative to $HOME on the cluster
CONDA_ENV="${CONDA_ENV:-comet-bio}"
LOCAL_PKG="$(cd "$(dirname "$0")" && pwd)"
EXTRA=("$@")

echo "══ 1/3 sync → $HOST:~/$REMOTE_DIR/qe_aggregation_probe"
ssh "$HOST" "mkdir -p ~/$REMOTE_DIR/qe_aggregation_probe"
rsync -az --delete \
  --exclude results --exclude results_smoke --exclude __pycache__ \
  --exclude '.venv*' --exclude logs \
  "$LOCAL_PKG/" "$HOST:$REMOTE_DIR/qe_aggregation_probe/"

echo "══ 2/3 remote setup + 3/3 submit"
LOCAL_KEY="${OPENAI_API_KEY:-}"
ssh "$HOST" REMOTE_DIR="$REMOTE_DIR" CONDA_ENV="$CONDA_ENV" \
    LOCAL_KEY="$LOCAL_KEY" EXTRA="${EXTRA[*]:-}" 'bash -s' <<'REMOTE'
set -euo pipefail
cd ~/"$REMOTE_DIR"/qe_aggregation_probe
mkdir -p logs

# ── conda env ────────────────────────────────────────────────────────────────
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh" 2>/dev/null \
  || source /etc/profile.d/conda.sh 2>/dev/null || true
if ! conda activate "$CONDA_ENV" 2>/dev/null; then
  echo "!! conda env '$CONDA_ENV' not found. Available:"; conda env list || true
  exit 1
fi
python - <<'PY' || pip install -q "datasets>=2.14" "openai>=1.30" sentencepiece
import datasets, openai, sentencepiece  # noqa
PY

export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
mkdir -p "$HF_HOME"

# ── HF token check (wmt22-cometkiwi-da is license-gated) ─────────────────────
if [[ -z "${HF_TOKEN:-}" && ! -s "$HF_HOME/token" && ! -s ~/.cache/huggingface/token ]]; then
  echo "!! WARNING: no HuggingFace token found — Unbabel/wmt22-cometkiwi-da is"
  echo "   license-gated and will fail to download. Run: huggingface-cli login"
fi

# ── prefetch WMT24++ (login node has internet; subsets are small) ────────────
python - <<'PY' || echo "!! WMT24++ prefetch failed (compute node will retry)"
from datasets import load_dataset
for lp in ["en-de_DE", "en-zh_CN", "en-es_MX", "en-fr_FR"]:
    load_dataset("google/wmt24pp", lp, split="train")
    print("  prefetched", lp)
PY

# ── perturbation backend: llm needs an API key at job runtime ────────────────
KEY="${OPENAI_API_KEY:-$LOCAL_KEY}"
BACKEND_ARGS=()
EXPORT_ARGS="--export=ALL"
if [[ -n "$KEY" ]]; then
  echo "── OPENAI_API_KEY found → perturb backend: llm"
  EXPORT_ARGS="--export=ALL,OPENAI_API_KEY=$KEY"
else
  echo "!! No OPENAI_API_KEY (local or remote) → submitting with the RULE backend"
  echo "   (deterministic length-preserving edits: valid lower-bound run)."
  echo "   For the LLM arm: export OPENAI_API_KEY and rerun deploy (scores are"
  echo "   cached — only the perturbation-dependent items get rescored)."
  BACKEND_ARGS=(--perturb_backend rule)
fi

# ── submit ───────────────────────────────────────────────────────────────────
JOB=$(sbatch --parsable "$EXPORT_ARGS" slurm_agg_probe.sh \
      ${BACKEND_ARGS[@]+"${BACKEND_ARGS[@]}"} $EXTRA)
echo "── submitted job $JOB"
squeue -j "$JOB" -o "%.10i %.12P %.14j %.8T %.10M %.6D %R" || true
echo "── logs: ~/$REMOTE_DIR/qe_aggregation_probe/logs/agg_probe-$JOB.out"
REMOTE
