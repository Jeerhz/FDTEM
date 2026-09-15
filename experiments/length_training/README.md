# Length-composition sweep — how much of a metric's training data must be sentences?

Continue a public MT metric on training mixes that differ **only** in how much
of their text is single sentences versus multi-sentence spans and whole
documents, then ask what that costs and what it buys.

| factor | levels |
|---|---|
| base metric | `Unbabel/wmt22-comet-da` (ref-based) · `Unbabel/wmt22-cometkiwi-da` (ref-free) |
| sentence fraction *f* | 0, 10, 20, 40, 60, 80, 100 % + two long-mass ablations (`frac000nat`, `frac000agg`) |
| encoder regime | unfrozen (`nr_frozen_epochs=0.3`) · frozen (head-only) |

2 × 9 × 2 = **36 arms**, plus the two baselines evaluated as released.

---

## Correction, 2026-08-18 — the first sweep did not measure composition

The 2026-08-14 runs are **not interpretable** and their ranking should not be
reported. The cause is in how training files were passed, not in the mixes.

COMET chooses an epoch's training file itself:

```python
# comet/models/base.py :: train_dataloader
data_path = self.hparams.train_data[self.current_epoch % len(self.hparams.train_data)]
```

and Lightning only calls `train_dataloader()` again when
`reload_dataloaders_every_n_epochs >= 1`. Our trainer config sets it to `0`, so
the loader was built **once**. The config generator passed one CSV per language
pair, so every arm trained on `train_data[0]` alone — one language pair's slice
of its mix — for its entire run:

| arm | file actually trained on | rows | share of the 24,000-row mix |
|---|---|---|---|
| `frac100` | `en-de_train.csv` | 6,940 | 29% |
| `frac000agg` | `en-de_train.csv` | 6,480 | 27% |
| `frac000nat` | `cs-de_train.csv` | 2,232 | 9% |
| `frac000` | `cs-de_train.csv` | 1,131 | 5% |
| `frac080` | `cs-de_train.csv` | 205 | 0.9% |

Two confounds followed, either one fatal on its own:

* **Unequal training budget.** Early stopping had patience 20 *epochs*, and an
  epoch was 205–6,940 examples depending on the arm. The arms that looked best
  are the arms that trained on the most data — the result ranking reproduces
  the budget ranking.
* **Unequal language coverage.** `frac100` and `frac000agg` trained on en-de;
  every other arm trained on cs-de. The "sentences vs long text" contrast was
  also an "en-de vs cs-de" contrast.

The training logs confirm it directly — one `Loading …csv.` line per run, next
to 60+ epochs:

```
$ grep -c "Loading /home" logs/comet-retrain-5190528.out     # 1
$ grep -oE "Epoch [0-9]+" logs/comet-retrain-5190528.out | sort -uV | tail -1   # Epoch 63
```

This was inherited from the Bio-MQM pipeline, where every mix had the same 8
language pairs. There the file count was constant, so the defect was invisible;
the WMT pools have 3 and 13 pairs, which made it decisive.

### The fix

Arms train on the mix's single `all_train.csv`. One epoch is then the whole
24,000-row mix in every arm, and the budget below is the same across the grid.
Three guards keep it that way:

* `make_config.py` refuses a multi-file `train_data` unless explicitly forced.
* `comet/models/base.py` warns at `setup()` when `train_data` has more than one
  file, saying which files will be silently ignored.
* `slurm/train.sh` renames a previous sweep's runs out of the arm directory
  before a fresh run, so checkpoint selection cannot pick a superseded run, and
  reports how many training files the run actually loaded.

Evaluation caches are keyed by label, and labels outlive the checkpoints behind
them, so `fdtem.comet_io` now records the checkpoint fingerprint alongside each
cache entry and rescores when it does not match. Without that, re-evaluating
`da-frac040` after retraining would silently replay the old run's scores.

**Still open — the language-coverage asymmetry.** The pools come from different
releases and overlap on almost nothing:

```
coverage[sent  ]:  3 pairs — en-de en-ru zh-en
coverage[agg   ]:  3 pairs — en-de en-ru zh-en
coverage[native]: 13 pairs — cs-de cs-uk en-ar en-bho en-cs en-et en-is en-ja en-mas en-ru en-sr en-uk en-zh
shared by all pools: 1 — en-ru
```

So *f* still moves the language distribution as well as the length
distribution. `make_mixtures.py --match_lp_coverage` restricts every pool to the
shared pairs, but as the numbers above show that means **one** language pair —
not a viable design. The realistic options are to analyse per language pair, to
add a WMT25 en-de/zh-en document pool so the overlap is usable, or to accept
and state the confound. The correction above does not decide this.

---

## Correction, 2026-08-24 — QE arms trained on truncated long inputs

CometKiwi (UnifiedMetric) encodes `<s> mt </s></s> src </s>` as **one** sequence
hard-truncated at 512 tokens; the data filter capped each side at 480 tokens
separately. Result: 7.8% of native and 10.7% of k=6 training rows (and 7.1% of
the WMT25 eval documents) overflowed for the QE arms, silently dropping the tail
of the source — on exactly the long inputs the sweep studies. DA arms
(sides encoded separately) were unaffected.

Fix: `prepare_data.py --max_concat_tokens` (default 508 = 512 − 4 special
tokens) now also drops rows with src+mt over the QE budget, from the **shared**
pools, so DA and QE keep training on byte-identical rows. `windows()` also
gained a seg_id-contiguity guard (11/72,447 zh-en windows spanned a gap).

Redeploy on rebuilt pools (v1 data and checkpoints stay untouched):

```bash
RUN=1          experiments/length_training/slurm/deploy_concat_fix.sh  # pools+mixes → ~/scratch/wmt_length_data_v2
RUN=1 SUBMIT=1 experiments/length_training/slurm/deploy_concat_fix.sh  # ... + submit arms (ckpts → retrain-wmt-v2)
```

---

## Extension, 2026-08-29 — four arms, a long budget, and two new lenses

The wave-1 arms moved but did not separate: 6 epochs (~4,500 optimizer steps) is
enough to show that composition does something and not enough to say what. The
follow-up fixes four things at once.

### 1. Four arms, named by what they are made of

| arm | training text | the question it answers |
|---|---|---|
| `frac100` | 100 % single sentences | the control: continued training that changes nothing about length |
| `frac000agg` | 100 % concatenated same-document windows | does *synthetic* length help? |
| `frac000nat` | 100 % natively long documents | does *real* document text help, and differently? |
| `frac000` | 50 / 50 concatenated + native | does the mixture beat either pure source? |

`frac000nat` and `frac000agg` had never trained, and the reason was a silent
one: a pure arm takes its whole N from ONE pool, while the default sizing rule
`N = min(sent, 2·agg, 2·native)` only guarantees half of N from each long pool.
`make_mixtures.py` logged `infeasible — skipped` and carried on, so a four-way
design quietly became a two-way one.

Fixed both ways round: `--total_policy pure` sizes N as `min(sent, agg, native)`,
the largest N at which all four arms hold the same number of rows, and an
infeasible arm is now a hard error unless `--allow_skip` is passed.

Measured on the v2 pools: sent 59,139 · agg 156,112 · native 39,244, so the pure
policy allows N = 39,244 and the 24,000 cap is what actually binds. **All four
arms are at N = 24,000** — the same epoch size as wave 1, so the new numbers are
comparable both to each other and to the earlier runs.

### 2. Ten times the budget, chained across the wall clock

`MAX_EPOCHS=60` (~45,000 steps) with `PATIENCE=10`. That does not fit in the
partition's 2-day limit, so each arm is submitted as a chain of jobs linked by
`--dependency=afterany`, each carrying `RESUME=auto`: the first starts from the
published base, every later one picks up its arm's own `last.ckpt` and the W&B
run id encoded in that path, so the chain is one run and one set of curves.
Lightning counts `max_epochs` globally across a resume, so the chain stops
itself at 60 epochs rather than doing 60 per link.

```bash
RUN=1          experiments/length_training/slurm/deploy_four_arms.sh   # mixes only (pools are reused)
RUN=1 SUBMIT=1 experiments/length_training/slurm/deploy_four_arms.sh   # ... + submit the grid
```

Pools are **not** rebuilt: the v2 pools already carry the concatenation-aware
token filter. Only the mixes are new (`mixes_pure/`), and checkpoints go to
`retrain-wmt-v3`, leaving v1 and v2 untouched.

### 3. The uncontrolled arm

`make_uncontrolled_mix.py` builds one mix out of everything: both pool splits
*and* the held-out evaluation portions. `slurm/launch_uncontrolled.sh` trains it
unfrozen from step 0, at ten times the continue-training learning rates, with
early stopping effectively off. It is the far end of the axis — the published
metric saw none of this, the four arms are a controlled step away, this is as
far as continued training goes — and it is what makes the size of the
composition effects readable.

**Its correlation numbers measure memorisation and are not results.** The mix
directory carries a `CONTAMINATED` marker, the manifest carries
`contaminated: true`, `train.sh` prints the marker in the job log and tags the
W&B run. MetaDocEval is a different corpus and nothing from it enters the mix,
so that lens stays honest for this arm — which is the reason to build it.

### 4. Two new lenses on the same predictions

`eval_length_profile.py` (run automatically by `slurm/eval_correlation.sh`,
`PROFILE=0` to skip) reuses the correlation lens's prediction cache and reports
what a correlation cannot:

* **the distribution of the scores per k** — mean, spread, quantiles, histogram,
  and `spread_ratio_vs_k1`. Kendall τ is invariant to any monotone squashing, so
  a metric whose long-text scores collapse into a narrow band keeps its τ and
  loses its resolution. This is the number that shows it.
* **length in tokens, not in sentences** — every quantity again against the
  XLM-R token count of the real input, in bins shared by every model and file.
  `n_mt` for the DA arms (three sides encoded separately) and `n_concat` for the
  QE arms (src+mt packed into one 512-token sequence — the budget is visible in
  the last bin).

Figures: `python report/figures/make_length_figures.py` → `report/figures/length/`.
It also emits the **per-phenomenon** MetaDocEval grids, absolute and as a delta
against each family's published metric: micro-averaging over a context window
lets a phenomenon that improves cancel a phenomenon that degrades, and the
average comes out flat.

---

## Running it

### The 2026-08-29 wave, in order

```bash
# 0. pools already exist (v2, concat-filtered) — nothing to rebuild.
#    Check what is in them before spending GPU on them:
python experiments/length_training/report_mix_composition.py

# 1. mixes for the four arms, all at the same N
RUN=1 experiments/length_training/slurm/deploy_four_arms.sh

# 2. the four arms x two families, 60 epochs, chained across the wall clock
SUBMIT=1 experiments/length_training/slurm/launch_long.sh

# 3. the contaminated arm (separate: different data, different schedule)
RUN=1 SUBMIT=1 experiments/length_training/slurm/launch_uncontrolled.sh

# 4. evaluate as arms finish — correlation + score distributions + token profile
sbatch --export=ALL,CKPT_ROOT=$HOME/scratch/checkpoints/retrain-wmt-v3,\
VAL_DATA_DIR=$HOME/scratch/wmt_length_data_v2 \
    experiments/length_training/slurm/eval_correlation.sh
sbatch --export=ALL,CKPT_ROOT=$HOME/scratch/checkpoints/retrain-wmt-v3 \
    experiments/length_training/slurm/eval_metadoceval.sh

# 5. figures (runs anywhere the JSONs are; skips whatever is not there yet)
python report/figures/make_length_figures.py
```

The alignment question — does the COMET *score* retrieve translations better
than the cosine of the encoder it is built on — lives in experiment 2:
`sbatch experiments/length_isolation/slurm/comet_align.sh`, with `QE_CKPT`
pointing at a finished QE arm.

### The original sweep

```bash
# 1. pools (once)
python experiments/length_training/prepare_data.py --output_dir ~/scratch/wmt_length_data

# 2. mixes (once) — prints the language-coverage report quoted above
python experiments/length_training/make_mixtures.py --data_dir ~/scratch/wmt_length_data

# 3. training: one arm, or the grid
MODEL=da MIX=frac040 FROZEN=0 sbatch experiments/length_training/slurm/train.sh
experiments/length_training/slurm/launch_sweep.sh                 # dry run
SUBMIT=1 MIXES="frac000 frac100" experiments/length_training/slurm/launch_sweep.sh   # first wave

# 4. evaluation (arms are discovered from the checkpoint root)
sbatch experiments/length_training/slurm/eval_correlation.sh      # validation + held-out
sbatch experiments/length_training/slurm/eval_metadoceval.sh      # discourse errors
python experiments/length_training/analyze.py --bootstrap
```

### Budget

Identical for every arm, and now meaningful because every epoch is the same size:

| | wave 1 (2026-08) | four-arm wave (`launch_long.sh`) |
|---|---|---|
| epoch | 24,000 rows = the whole mix | N rows = the whole mix, `--total_policy pure` |
| optimizer steps / epoch | 750 (batch 4 × accum 8 × 1 GPU) | N / 32, printed at submit time |
| `MAX_EPOCHS` | 6 → ≤ 4,500 steps | 60 → ~45,000 steps |
| `PATIENCE` | 3 validation checks | 10 validation checks |
| wall clock | one 2-day job | a chain of 4 jobs, `RESUME=auto` |

Early stopping is a guard against divergence, not the thing that sets the
budget — that is the point of the correction. Raise `MAX_EPOCHS` or set
`MAX_STEPS` to change it, but change it for the whole grid at once.

## Evaluation lenses

| lens | data | status | question |
|---|---|---|---|
| **validation** | `~/scratch/wmt_length_data` — `sent`/`agg`/`native` `*_val.csv`, 21,988 rows, 15 pairs | *selection-coupled*: the split early stopping monitors | development signal, directly comparable to `val_kendall`; τ per k ∈ {0,1,2,3,4,6} |
| **held-out** | `~/scratch/wmt_eval_portion` — WMT22/23/24/25 portions | fully held out | the reportable per-k numbers |
| **MetaDocEval** | contrastive test set (Dahan, Bawden & Yvon, EAMT 2026) | fully held out, different corpus | does long-text training buy *discourse-error detection*, or only a rescaled score? Accuracy per perturbation and context window w ∈ {1,3,6,9} |

`sentence_splitting` in MetaDocEval is quality-preserving by design, so its
"accuracy" is a false-positive rate; the script labels it as such.

## Files

| | |
|---|---|
| `prepare_data.py` | WMT22 segments, aggregated windows, WMT25 documents → one schema |
| `make_mixtures.py` | constant-size mixes; `--total_policy pure` for the four-arm grid; language-coverage report; `manifest.json` |
| `make_uncontrolled_mix.py` | the contaminated everything-mix, with its marker |
| `make_config.py` | per-arm training config: one train file, one budget |
| `list_arms.py` | trained arms → `label=checkpoint` for the shell |
| `train.py` | COMET training with a W&B logger |
| `eval_correlation.py` · `eval_metadoceval.py` · `analyze.py` | the three lenses and the results table |
| `eval_length_profile.py` | score distributions per k, and everything again per input-token bin |
| `report_mix_composition.py` | what is actually inside each pool and mix, per k |
| `slurm/` | `train.sh` · `launch_sweep.sh` · `deploy_four_arms.sh` · `launch_long.sh` · `launch_uncontrolled.sh` · `eval_correlation.sh` · `eval_metadoceval.sh` |
| `report/figures/make_length_figures.py` | per-phenomenon MetaDocEval, score distributions, token axis |
