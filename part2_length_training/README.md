# Part 2 — how much of a metric's training data must be sentences?

Continue a public MT metric on training mixes that differ **only** in how much of
their text is single sentences versus multi-sentence windows and whole documents,
then ask what that costs and what it buys.

| factor | levels |
|---|---|
| base metric (`arms.py: FAMILIES`) | `da` = `Unbabel/wmt22-comet-da` (ref-based) · `qe` = `Unbabel/wmt22-cometkiwi-da` (ref-free) |
| mix (`arms.py: MIX_SPECS`) | `frac000` … `frac100` (percentage of single sentences, long mass split 50/50 windows / documents), `frac000agg` (windows only), `frac000nat` (documents only), `uncontrolled` (everything, contaminated) |
| encoder regime | unfrozen (`nr_frozen_epochs=0.3`) · frozen (`--frozen`, head-only) |
| budget (`arms.py: TRAIN_PRESETS`) | `default` 60 epochs / patience 10 · `wave1` 6 / 3 · `uncontrolled` |

An arm is an `ArmLabel` (`models.py`): `da-frac000agg-frozen` in results,
`mix-frac000agg-frozen` as a checkpoint directory under `~/scratch/checkpoints/<ckpt_root>/`.

## Pipeline

```
load_wmt_pools.py      WMT22 MQM segments (sent, k=1) + aggregated windows (agg, k∈{2,3,4,6})
                       + WMT25 native documents (native, k=0)  -> ~/scratch/wmt_length_data_v2/*.csv
load_heldout_sets.py   WMT23/24 paragraph MQM sets              -> ~/scratch/wmt_eval_portion/heldout-*_val.csv
load_metadoceval.py    clone of the MetaDocEval test set        -> ~/scratch/metadoceval-testset
make_mixtures.py       named mixes at constant N (+ uncontrolled) -> <data_dir>/mixes_pure/<mix>/all_train.csv, manifest.json
inspect_mixtures.py    what is actually in each pool and mix, per k
load_model.py          the published checkpoint + base config of a family
train.py               one arm -> common.train_comet (W&B, RESUME=auto)
eval_validation.py     Kendall/Spearman/Pearson per (file, k)  -> results/correlation_<lens>.json
eval_length_profile.py score distributions per k and per token bin -> results/length_profile_<lens>.json
eval_metadoceval.py    contrastive accuracy per perturbation and window -> results/metadoceval.json
analyze.py             the results table (+ document bootstrap)
figures/make_figures.py the 16 report figures -> figures/
```

Every script is `python -m part2_length_training.<script> --help`; the slurm wrappers
only set the environment. Results are the pydantic models of `models.py`
(`CorrelationResults`, `LengthProfileResults`, `MetaDocEvalResults`); see
`results/README.md` for the provenance of the committed files.

## Running it

```bash
# 0. data, once (30-60 min CPU, tokenisation-bound)
python -m part2_length_training.load_wmt_pools --output_dir ~/scratch/wmt_length_data_v2
python -m part2_length_training.load_heldout_sets --output_dir ~/scratch/wmt_eval_portion
python -m part2_length_training.load_metadoceval

# 1. the four-arm grid: builds the missing mixes, checks the design guard, submits chains
part2_length_training/slurm/launch_arms.sh                    # dry run
RUN=1 SUBMIT=1 part2_length_training/slurm/launch_arms.sh     # frac100 frac000agg frac000nat frac000, da + qe
SUBMIT=1 ARMS=uncontrolled PRESET=uncontrolled part2_length_training/slurm/launch_arms.sh

# 2. one arm by hand
MODEL=da MIX=frac040 FROZEN=0 sbatch part2_length_training/slurm/train.sh
MODEL=da MIX=frac040 RESUME=auto sbatch part2_length_training/slurm/train.sh    # continue after a wall clock

# 3. evaluate as arms finish (checkpoints are discovered from CKPT_ROOT)
sbatch part2_length_training/slurm/eval_validation.sh      # val + heldout lenses + length profile
sbatch part2_length_training/slurm/eval_metadoceval.sh
python -m part2_length_training.analyze --lens heldout --bootstrap

# 4. figures (anywhere the JSONs are; skips what is missing)
python -m part2_length_training.figures.make_figures
```

The wave-1 ladder (nine mixes, frozen and unfrozen, 6 epochs) is
`PRESET=wave1 ARMS="frac000 frac010 frac020 frac040 frac060 frac080 frac100" FROZEN_LEVELS="0 1" CHAIN=1 MIX_DIR=$HOME/scratch/wmt_length_data_v2/mixes SUBMIT=1 part2_length_training/slurm/launch_arms.sh`.
The alignment question (does the COMET *score* retrieve translations better than the
cosine of its encoder) is part 1: `QE_CKPT=<kiwi arm> sbatch part1_block_alignment/slurm/evaluate_comet_score.sh`.

## Evaluation lenses

| lens | data | status | question |
|---|---|---|---|
| **validation** | `~/scratch/wmt_length_data_v2/*_val.csv` (15 pairs) | selection-coupled: the split early stopping monitors | development signal, comparable to `val_kendall`; τ per k ∈ {0,1,2,3,4,6} |
| **held-out** | `~/scratch/wmt_eval_portion` — WMT22/23/24/25 portions | fully held out | the reportable per-k numbers |
| **MetaDocEval** | contrastive test set (Dahan, Bawden & Yvon, EAMT 2026) | held out, different corpus | does long-text training buy discourse-error detection? accuracy per perturbation and window w ∈ {1,3,6,9} |

`sentence_splitting` in MetaDocEval is quality-preserving by design, so its "accuracy"
is a false-positive rate; the scripts label it as such. Kendall τ is invariant to any
monotone squashing of the scale, so `eval_length_profile.py` also reports the
**distribution** of the scores per k and per XLM-R token bin (`n_mt` for DA, `n_concat`
= src+mt for the 512-token CometKiwi budget).

Known gap: `load_heldout_sets.py` writes the WMT23/24 files; the `wmt22-*` and
`wmt25-*` portions of `~/scratch/wmt_eval_portion` were assembled by hand in 2026-08 and
have no producer in the repository.

## The design guarantees, and how they were broken once

**One train file per arm (2026-08-18).** COMET picks an epoch's file as
`train_data[epoch % len(train_data)]` and Lightning only rebuilds the loader when
`reload_dataloaders_every_n_epochs >= 1` (ours is 0). The 2026-08-14 sweep passed one
CSV per language pair, so every arm trained on `train_data[0]` alone — 205 to 6,940 rows
depending on the arm instead of the 24,000-row mix — with unequal budgets and unequal
language coverage. Its rankings reproduce the budget ranking and are not results
(`results/wave1_wandb_curve_diagnosis.json` keeps the curves). Arms now train on the
mix's single `all_train.csv`; `common/train_config.py` refuses multi-file lists;
`train.py` moves a previous run out of the arm directory so checkpoint selection cannot
pick a superseded one; evaluation caches record the checkpoint fingerprint.

**QE inputs are one sequence (2026-08-24).** CometKiwi encodes `<s> mt </s></s> src </s>`
as one 512-token sequence; a per-side cap of 480 let 8-11 % of long rows overflow and
silently lose the tail of the source. `load_wmt_pools.py --max_concat_tokens` (default
508) drops those rows from the shared pools, so DA and QE train on identical rows.

**Equal N across arms (2026-08-29).** A pure arm takes its whole N from one pool;
`--total_policy pure` sizes N as min(sent, agg, native) (the 24,000 cap binds on the v2
pools), and an infeasible arm is a hard error unless `--allow_skip`. `launch_arms.sh`
checks the manifest (`make_mixtures.py --verify`) before submitting, and chains `CHAIN`
jobs per arm with `--dependency=afterany` and `RESUME=auto`: one W&B run, one set of
curves, and Lightning's global `max_epochs` stops the chain.

**Language coverage is still confounded with composition.** The sentence and window
pools come from WMT22 (3 pairs), the document pool from WMT25 (13 pairs); they share
one pair. `make_mixtures.py --match_lp_coverage` restricts to the shared pairs (not a
viable design); analyse per language pair or state the confound.

**The uncontrolled arm** (`make_mixtures.py --arms uncontrolled`) folds the validation
split and the held-out portions into training, unfrozen from step 0 at ten times the
continue-training rates. It is the far end of the axis and makes the composition
effects readable. Its mix directory carries a `CONTAMINATED` marker, its manifest
`contaminated: true`, and its W&B run the tag `contaminated`; its correlation numbers
measure memorisation and are not results. MetaDocEval is a different corpus and stays
honest for it.

## Files

| | |
|---|---|
| `models.py` · `arms.py` | data rows, mixes, `ArmLabel`, presets, the three result models; the families, `MIX_SPECS`, `TRAIN_PRESETS` |
| `load_wmt_pools.py` · `load_heldout_sets.py` · `load_metadoceval.py` | data |
| `make_mixtures.py` · `inspect_mixtures.py` | named compositions (+ `--verify`, `--arms uncontrolled`) and what is inside them |
| `load_model.py` · `train.py` · `list_arms.py` | the model to fine-tune, one arm, the trained arms as `label=checkpoint` |
| `eval_validation.py` · `eval_length_profile.py` · `eval_metadoceval.py` · `analyze.py` | the three lenses and the table |
| `configs/comet_da.yaml` · `configs/comet_qe.yaml` | the two base configs (sub-configs in `common/configs/`) |
| `slurm/` | `train.sh` · `launch_arms.sh` · `eval_validation.sh` · `eval_metadoceval.sh` |
| `figures/make_figures.py` | per-phenomenon MetaDocEval, score distributions, token axis |
| `PROTOCOL.md` | the pre-registered research questions and their corrections |
