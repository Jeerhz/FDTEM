# Cluster runbook — COMET length-sensitivity experiments

Instructions for a Claude Code session running **on the cluster**, in the FDTEM
repo. Read this file end to end before running anything.

Companion document: `docs/RETRAIN_AND_BLOCK_XSIM.md` has the scientific design
and the meaning of each result. This file is the operational procedure.

---

## 1. What we are trying to find out

COMET is trained on **single sentences** but is routinely applied to
**paragraphs and documents**. Two things could go wrong, and we want to separate
them:

1. **Is the length behaviour a property of the training data?** If we continue
   training COMET on long text only, does it lose its agreement with human
   judgement on short text — and how much sentence-level data is needed to keep
   both? That is the **sentence-fraction sweep** (Experiment A2).
2. **Does training on long text buy genuine document-level competence, or just
   a rescaled score?** Human correlation cannot answer this, because
   discourse-level errors are built to preserve sentence-level fluency. That is
   what **MetaDocEval** (A3) and **block-xSIM++** (B) test.

The experimental lever is the **composition** of the training mix: fraction
`f ∈ {0,10,20,40,60,80,100}%` of single-sentence examples, the rest being
multi-sentence windows, **at constant total size** so that only composition
varies. Crossed with two encoder regimes — encoder frozen (only the regression
head learns) vs unfrozen (the default schedule) — that is **14 training arms**.

Everything is evaluated through three lenses that answer different questions:

| lens | script | question |
|------|--------|----------|
| segment correlation vs length | `eval_length_correlation.py` | does it still track human MQM, per window size k? |
| document contrastive accuracy | `eval_metadoceval.py` | does it detect discourse errors, and does context dilute them? |
| block-xSIM++ | `run_block_xsim.py` | is the *encoder* still sensitive to one buried error? |

---

## 2. Ground rules for this session

- **Do not commit or push.** Report changes; the user decides.
- **Do not delete or overwrite** anything under `~/scratch` without asking.
  Checkpoints and staged datasets are expensive to rebuild.
- **Never fabricate results.** If a job is still queued or running, say so. Do
  not infer numbers from partial logs.
- **Long jobs are `sbatch`, not foreground.** Submit, record the job id, move on.
  Never block a session waiting for a 2-day job.
- If a step fails, **read the log before resubmitting**. Logs are
  `logs/<jobname>-<jobid>.{out,err}`.
- Report progress with job ids and real paths, so the user can check
  independently.

---

## 3. Preflight

Run these checks first and report anything that fails. Do not start Phase 0
until all pass.

```bash
cd ~/FDTEM                      # adjust if the repo lives elsewhere
git status --short              # expect a clean-ish tree; report surprises

# environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate comet-bio
python -c "import comet, torch, pandas, scipy, matplotlib; \
           print('comet', comet.__version__, 'torch', torch.__version__, \
                 'cuda', torch.cuda.is_available())"

# cluster
sinfo -s                        # confirm the 'gpu' partition name and limits
squeue -u "$USER"               # anything of ours already running?

# credentials
wandb status                    # offline mode is acceptable, just note it
ls ~/.cache/huggingface/token   # needed for gated FLORES+ / CometKiwi

# disk — this is the one people forget
df -h ~/scratch
du -sh ~/scratch/* 2>/dev/null | sort -h | tail
```

**Disk budget.** Each training arm keeps `save_top_k=2` plus `last.ckpt`, and a
COMET/XLM-R-large checkpoint is roughly 2.3 GB — about **7 GB per arm, ~100 GB
for all 14**. If `~/scratch` cannot absorb that, tell the user before launching
and propose either fewer arms or `save_top_k=1`.

---

## 4. Phase 0 — build the data, exactly once

**This phase is blocking and must not be parallelised.** The training script
will build missing data itself, so if you submit 14 jobs into an empty data
directory they will all detect it missing and race to write the same CSVs.
Build first, submit second.

Note that `~/scratch/paragraph_mqm` may already exist from earlier work but
**predates the `*_trainpool.csv` output** that the sweep samples from. If those
files are absent the directory must be regenerated.

```bash
ls ~/scratch/paragraph_mqm/*_trainpool.csv 2>/dev/null | head
```

If that returns nothing, rebuild (interactively, ~30–60 min, CPU-bound —
tokenisation dominates):

```bash
python scripts/prepare_paragraph_data.py \
    --output_dir ~/scratch/paragraph_mqm \
    --lang_pairs en-de de-en en-es es-en en-fr fr-en en-ru ru-en \
    --wandb_project comet-retrain
```

Then build the mixes and **verify them before training on them**:

```bash
python scripts/make_sentence_mix.py --data_dir ~/scratch/paragraph_mqm

cat ~/scratch/paragraph_mqm/mixes/MANIFEST.md
```

Check and report these four things — they are the design guarantees, and if any
is violated the whole sweep is confounded:

1. every `fracNNN` has the **same total** number of windows per language pair;
2. the `k1` column equals the advertised percentage of that total;
3. the long window sizes are split evenly within each mix;
4. `manifest.json` records the seed and an md5 per CSV.

---

## 5. Phase 1 — baseline evaluations (no training required)

Run this **while training is queued**. It costs little, and it gives the
reference curves that every retrained arm is compared against. If the baselines
look wrong, we learn it now rather than after two days of GPU.

```bash
sbatch scripts/slurm_metadoceval.sh          # wmt22-comet-da + cometkiwi
```

This clones the MetaDocEval test set into `~/scratch/metadoceval-testset` if
missing (~36 MB), then scores it. Expect a few hours: ~105k unique window
inputs per model after deduplication.

**Two things to check in the output and report:**

- The script prints the contrastive-pair counts it actually found and
  cross-checks them against the clone's own README. It will warn that
  `sentence_shuffling` and `sentence_splitting` are smaller than advertised.
  **This is expected** — it is a known README regression upstream (commit
  `1ee6788` had the correct values, `d3e8dc6` overwrote them), not a broken
  download. All results use the data actually present.
- `sentence_splitting` is **quality-preserving by design**: high accuracy there
  is a false-positive rate, not a success. The plot labels it.

Segment-level baseline, in parallel (no slurm wrapper — use `srun`, adapt flags
to what `sinfo -s` reported):

```bash
srun --partition=gpu --gres=gpu:1 --mem=64G --time=04:00:00 \
  python scripts/eval_length_correlation.py \
    --models wmt22=Unbabel/wmt22-comet-da \
             bio=$HOME/scratch/checkpoints/bio_mqm/comet-bio-mqm/4yeqp7cn/checkpoints/last.ckpt \
    --data_dir ~/scratch/paragraph_mqm \
    --output results/retrain/length_correlation.json \
    --wandb_project comet-retrain
```

---

## 6. Phase 2 — training, wave 1 (the headline arms)

Do **not** submit all 14 at once on the first try. Send the four arms that
answer the main question, confirm they train, then fill in the sweep. A config
error costs one hour this way instead of fourteen.

```bash
for MIX in frac000 frac100; do        # long-only vs sentence-only control
  for FROZEN in 0 1; do               # head-only vs full finetuning
    VARIANT=mix MIX=$MIX FROZEN=$FROZEN sbatch scripts/slurm_retrain_comet.sh
  done
done
squeue -u "$USER"
```

Each job writes to `~/scratch/checkpoints/retrain/mix-<frac>[-frozen]/` — one
directory per arm, so they never collide — and drops a generated `config.yaml`
there recording exactly what it trained on. Wall time is up to 2 days.

**Within ~15 minutes of the first job starting, verify and report:**

```bash
tail -40 logs/comet-retrain-<jobid>.out
```

- the `Variant`/`arm`/`frozen` header line matches what you intended;
- `train_data` in `~/scratch/checkpoints/retrain/mix-frac000/config.yaml` points
  at the `mixes/frac000/` CSVs, **not** the default `paragraph_mqm` ones;
- for a `FROZEN=1` arm, `nr_frozen_epochs: 1000` is in that config;
- loss is decreasing and `val_kendall` is being logged.

Only once wave 1 is training cleanly, submit the rest:

```bash
for MIX in frac010 frac020 frac040 frac060 frac080; do
  for FROZEN in 0 1; do
    VARIANT=mix MIX=$MIX FROZEN=$FROZEN sbatch scripts/slurm_retrain_comet.sh
  done
done
```

---

## 7. Phase 3 — evaluate the checkpoints

As arms finish, evaluate them. Both evaluation scripts cache predictions per
model, so re-running with an extra checkpoint only scores the new one — it is
cheap to re-run as arms complete, and there is no need to wait for all 14.

```bash
# every finished mix-* arm, plus the baselines, auto-discovered
SWEEP=1 sbatch scripts/slurm_metadoceval.sh
```

Segment correlation across arms (extend the `--models` list as checkpoints
appear; label each arm exactly as its directory is named, so plots stay
readable):

```bash
srun --partition=gpu --gres=gpu:1 --mem=64G --time=06:00:00 \
  python scripts/eval_length_correlation.py \
    --models wmt22=Unbabel/wmt22-comet-da \
             frac000=$HOME/scratch/checkpoints/retrain/mix-frac000/*/*/checkpoints/last.ckpt \
             frac020=... frac100=... \
    --data_dir ~/scratch/paragraph_mqm \
    --output results/retrain/length_correlation.json \
    --wandb_project comet-retrain
```

Encoder-level check on the most interesting arm (usually `frac000`):

```bash
RETRAIN_CKPT=$HOME/scratch/checkpoints/retrain/mix-frac000/<proj>/<run>/checkpoints/last.ckpt \
    sbatch scripts/slurm_block_xsim.sh
```

---

## 8. Failure playbook

**Job hit the 2-day wall clock.** Resume onto the same W&B run so the curves
stay continuous; the run id is in the job's log header:

```bash
VARIANT=mix MIX=frac000 FROZEN=0 \
  RESUME=<path-to-last.ckpt> WANDB_RUN_ID=<id> \
  sbatch scripts/slurm_retrain_comet.sh
```

**CUDA OOM.** Inputs run to ~480 tokens. Lower `batch_size` in
`configs/models/comet_paragraph_continue.yaml` and raise
`accumulate_grad_batches` in `configs/trainer_wandb.yaml` by the same factor, so
the effective batch is unchanged — otherwise the arms are no longer comparable
to each other.

**EarlyStopping fires immediately on resume.** Add `--reset_early_stopping`
(supported by `scripts/train_wandb.py`).

**`make_sentence_mix.py` exits with "pool exhausted".** A deliberate assertion,
not a crash: the constant-total design could not be met. Report the language
pair and counts; the fix is `--per_lp_total` or dropping a language pair, and
that is the user's call.

**MetaDocEval clone fails.** Clone it manually and pass `DATA_DIR=`. Do not
hand-edit the JSON.

**Several jobs crash instantly with a data error.** Almost certainly the Phase 0
race: they were submitted before the data existed. Build the data, then
resubmit.

---

## 9. What to report back

Once results exist, give the user a short written summary — not raw JSON:

1. **The headline number:** does `frac000` (long-only) lose sentence-level
   Kendall τ at k=1 relative to `wmt22-comet-da`, and by how much?
2. **The restoration point:** the smallest sentence fraction that recovers
   baseline k=1 correlation. This is the practically useful result.
3. **Frozen vs unfrozen:** if the frozen arms barely move while the unfrozen
   ones do, the length behaviour lives in the encoder weights rather than the
   regression head.
4. **MetaDocEval:** for each arm, does lexical-perturbation accuracy *rise* with
   window size `w`? In the paper no metric genuinely does — only CometKiwi
   climbs, and barely to chance. A retrained arm that climbs above chance would
   be a real finding, so double-check it before claiming it.
5. **The splitting false-positive rate** across `w` — flag if retraining made it
   worse.

Include job ids, checkpoint paths, and the result JSON paths so anything can be
re-derived. State plainly which arms have **not** finished.
