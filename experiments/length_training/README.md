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

## Running it

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

| | |
|---|---|
| epoch | 24,000 rows = the whole mix |
| optimizer steps / epoch | 750 (batch 4 × accum 8 × 1 GPU) |
| `MAX_EPOCHS` | 6 → ≤ 144,000 examples, ≤ 4,500 steps |
| `PATIENCE` | 3 validation checks (validation runs once per epoch) |

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
| `make_mixtures.py` | constant-size mixes; language-coverage report; `manifest.json` |
| `make_config.py` | per-arm training config: one train file, one budget |
| `list_arms.py` | trained arms → `label=checkpoint` for the shell |
| `train.py` | COMET training with a W&B logger |
| `eval_correlation.py` · `eval_metadoceval.py` · `analyze.py` | the three lenses and the results table |
| `slurm/` | `train.sh` · `launch_sweep.sh` · `eval_correlation.sh` · `eval_metadoceval.sh` |
