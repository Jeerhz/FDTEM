#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# launch_long.sh — the four-arm grid, trained far past the wave-1 budget.
#
# The four arms differ ONLY in what kind of text they are made of, at the same
# number of rows (see make_mixtures.py --total_policy pure):
#
#   frac100      100 % single sentences                    "phrases"
#   frac000agg   100 % concatenated same-document windows  "phrases concaténées"
#   frac000nat   100 % natively long documents             "documents natifs"
#   frac000       50 / 50 concatenated + native            "mixte"
#
# Wave 1 gave every arm 6 epochs (~4,500 optimizer steps), which was enough to
# show the arms move but not enough to separate them. This launcher gives them
# MAX_EPOCHS=60 (~45,000 steps) with PATIENCE=10, i.e. ten times the budget.
#
# ── Why the jobs are chained ──────────────────────────────────────────────────
# 60 epochs does not fit in the partition's 2-day wall clock. Each arm is
# therefore submitted as a CHAIN of jobs linked by --dependency=afterany, every
# one of them carrying RESUME=auto: the first finds no checkpoint and starts
# from the published base, each later one picks up its arm's own last.ckpt and
# the W&B run id encoded in that path, so the whole chain is one run and one set
# of curves. Lightning counts max_epochs GLOBALLY across a resume, so every link
# carrying MAX_EPOCHS=60 converges on 60 epochs total — the chain stops itself.
# Links that start after the arm has already finished exit in minutes.
#
# --dependency=afterany (not afterok) is deliberate: a job killed by the wall
# clock exits non-zero, and that is exactly the case the next link exists for.
#
# ── Usage (login node) ────────────────────────────────────────────────────────
#   experiments/length_training/slurm/launch_long.sh                 # dry run
#   SUBMIT=1 experiments/length_training/slurm/launch_long.sh        # send it
#   SUBMIT=1 MODELS=da ARMS=frac000nat .../launch_long.sh            # one cell
#
# Tunables: MODELS ("da qe") · ARMS · FROZEN_LEVELS ("0") · MAX_EPOCHS (60) ·
#           PATIENCE (10) · CHAIN (4 links/arm) · DATA_DIR · MIX_DIR ·
#           CKPT_ROOT · SEED · plus anything train.sh reads.
#
# Build the mixes first with slurm/deploy_four_arms.sh — this script refuses to
# submit if the four arms do not all exist at the same size.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"

EXP=experiments/length_training

MODELS="${MODELS:-da qe}"
ARMS="${ARMS:-frac100 frac000agg frac000nat frac000}"
FROZEN_LEVELS="${FROZEN_LEVELS:-0}"
CHAIN="${CHAIN:-4}"

export MAX_EPOCHS="${MAX_EPOCHS:-60}"
export PATIENCE="${PATIENCE:-10}"
export DATA_DIR="${DATA_DIR:-$HOME/scratch/wmt_length_data_v2}"
export MIX_DIR="${MIX_DIR:-$DATA_DIR/mixes_pure}"
export CKPT_ROOT="${CKPT_ROOT:-retrain-wmt-v3}"
SMALL_GPUS="${SMALL_GPUS:-gpu001,gpu004,gpu011,gpu014}"

# ── design guard ──────────────────────────────────────────────────────────────
# The budget is only a controlled factor if an epoch is the same number of
# examples in every arm. Check it against the manifest rather than trusting that
# the mixes were built with the right policy.
python - "$MIX_DIR" $ARMS <<'PY'
import json, sys
from pathlib import Path
mix_dir, arms = Path(sys.argv[1]), sys.argv[2:]
man_path = mix_dir / "manifest.json"
if not man_path.exists():
    raise SystemExit(f"no manifest at {man_path} — build the mixes first:\n"
                     f"  experiments/length_training/slurm/deploy_four_arms.sh")
man = json.load(open(man_path))
missing = [a for a in arms if a not in man["mixes"]]
if missing:
    raise SystemExit(
        f"mixes {missing} are absent from {man_path}.\n"
        f"  present: {sorted(man['mixes'])}\n"
        f"  A pure-pool arm needs a whole N from one pool; rebuild with\n"
        f"  make_mixtures.py --total_policy pure --arms {' '.join(arms)}")
totals = {a: sum(man["mixes"][a]["counts"][p] for p in ("sent", "agg", "native"))
          for a in arms}
if len(set(totals.values())) != 1:
    raise SystemExit(f"arms have unequal row counts, so an epoch means different "
                     f"things in each: {totals}")
n = next(iter(totals.values()))
print(f"mixes OK: {len(arms)} arms x {n:,} rows "
      f"(policy={man.get('total_policy', 'mix')}, seed={man['seed']})")
for a in arms:
    c = man["mixes"][a]["counts"]
    print(f"  {a:12} sent={c['sent']:>6,}  agg={c['agg']:>6,}  native={c['native']:>6,}")
PY

STEPS_PER_EPOCH=$(python - "$MIX_DIR" "$(echo $ARMS | cut -d' ' -f1)" <<'PY'
import csv, sys
csv.field_size_limit(10 ** 9)
p = f"{sys.argv[1]}/{sys.argv[2]}/all_train.csv"
n = sum(1 for _ in csv.reader(open(p, newline="", encoding="utf-8"))) - 1
print(max(1, n // 32))   # batch 4 x accum 8
PY
)

echo
echo "═══════════════════════════════════════════════════════════"
echo " Four-arm grid, long budget"
echo "   arms      : $ARMS"
echo "   models    : $MODELS   frozen levels: $FROZEN_LEVELS"
echo "   budget    : MAX_EPOCHS=$MAX_EPOCHS  PATIENCE=$PATIENCE"
echo "               ~$STEPS_PER_EPOCH optimizer steps/epoch"
echo "               → ~$((MAX_EPOCHS * STEPS_PER_EPOCH)) steps/arm"
echo "   chain     : $CHAIN linked jobs per arm (2-day wall clock each)"
echo "   data      : $MIX_DIR"
echo "   ckpts     : ~/scratch/checkpoints/$CKPT_ROOT"
echo "═══════════════════════════════════════════════════════════"

n=0
for model in $MODELS; do
  for arm in $ARMS; do
    for frozen in $FROZEN_LEVELS; do
      dep=""
      for link in $(seq 1 "$CHAIN"); do
        args=(--export="ALL,MODEL=$model,MIX=$arm,FROZEN=$frozen,RESUME=auto,\
MAX_EPOCHS=$MAX_EPOCHS,PATIENCE=$PATIENCE,DATA_DIR=$DATA_DIR,MIX_DIR=$MIX_DIR,\
CKPT_ROOT=$CKPT_ROOT")
        [[ "$frozen" == "0" ]] && args+=(--exclude="$SMALL_GPUS")
        [[ -n "$dep" ]] && args+=(--dependency="afterany:$dep")
        n=$((n + 1))
        if [[ "${SUBMIT:-0}" == "1" ]]; then
          jid=$(sbatch --parsable "${args[@]}" "$EXP/slurm/train.sh")
          echo "submitted $jid  model=$model arm=$arm frozen=$frozen link=$link/$CHAIN${dep:+ after $dep}"
          dep="$jid"
        else
          echo "would submit: model=$model arm=$arm frozen=$frozen link=$link/$CHAIN${dep:+ after \$prev}"
          dep="PREV"
        fi
      done
    done
  done
done

echo
if [[ "${SUBMIT:-0}" == "1" ]]; then
  echo "$n job(s) submitted across $((n / CHAIN)) arm(s)."
  echo "Watch:   squeue -u $USER"
  echo "Then:    sbatch --export=ALL,CKPT_ROOT=$HOME/scratch/checkpoints/$CKPT_ROOT,\
VAL_DATA_DIR=$DATA_DIR,SELECT=best $EXP/slurm/eval_correlation.sh"
else
  echo "$n job(s) — dry run, set SUBMIT=1 to send them."
fi
