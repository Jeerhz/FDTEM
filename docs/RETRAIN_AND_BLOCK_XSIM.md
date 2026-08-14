# COMET retraining & block-level xSIM++

Two experiments on how COMET behaves on **longer text**. The Lp-norm aggregation
probe (`qe_aggregation_probe/`, formerly "Exp 0") has been **dropped**.

Shared conventions: models & datasets live under `~/scratch`; FLORES+ is the
gated `openlanguagedata/flores_plus` (account `AdleBenSalem` is allow-listed);
everything logs to W&B. The Bio-MQM ≥10-epoch checkpoint is
`~/scratch/checkpoints/bio_mqm/comet-bio-mqm/4yeqp7cn/checkpoints/last.ckpt`.

---

## Experiment A — retrain COMET (better encoder / paragraph data)

**Question.** Two knobs on COMET's quality-vs-length behaviour:
1. **Encoder** — how much of COMET is its ageing XLM-R backbone? Swap in a
   better-aligned encoder of the same size.
2. **Training data** — COMET is trained on short segments. Continue it on
   *paragraph-level* text and see whether correlation with human quality holds up
   (or improves) as text grows.

### Data — length-balanced paragraph mix (`scripts/prepare_paragraph_data.py`)

Bio-MQM segments carry `(system, doc_id, seg_id)`, so consecutive same-document
segments form real paragraphs. For every document we take sliding windows of
`k ∈ {1,2,3,4,6}` consecutive segments, concatenate `src`/`mt`/`ref`, and score
the window by the **mean per-segment MQM penalty** (z-normalised + sigmoid,
per language pair, fit on train). At `k=1` this is exactly the sentence-level
Bio-MQM score, so all lengths share one scale.

- Also writes `{lp}_trainpool.csv` — **all** train windows before balancing,
  the sampling stock for the sentence-fraction sweep (Experiment A2).
- **Length-balanced**: every `k` contributes the same number of *train* windows
  (the scarcest length sets the count, or `--per_length N`). So the model sees
  "as many segments of each length as we have", per the brief — short and long
  segments in equal measure, `k=1` retained so sentence-level quality is not
  sacrificed.
- Train windows overlap (stride 1 = augmentation); **val windows are
  non-overlapping** (stride k) and never subsampled, so `k`-wise correlations
  use everything available.
- Windows over `--max_tokens` (480) on any side are dropped (512 encoder limit).
- Outputs `{lp}_train.csv` / `{lp}_val.csv` (+ `all_*`, `stats.json`, plots).

```bash
python scripts/prepare_paragraph_data.py \
    --output_dir ~/scratch/paragraph_mqm --wandb_project comet-retrain
```

### Training arms

| arm | config | what changes | baseline it isolates against |
|-----|--------|--------------|------------------------------|
| **paragraph** | `configs/models/comet_paragraph_continue.yaml` | continue `wmt22-comet-da` on the paragraph mix | the public `wmt22-comet-da` and the sentence-only bio ckpt |
| **encoder_swap** | `configs/models/comet_encoder_swap.yaml` | retrain the head on `microsoft/infoxlm-large` (drop-in XLM-R-large-compatible, extra cross-lingual contrastive pretraining) | the same recipe with `pretrained_model=xlm-roberta-large` |

`infoxlm-large` is the default swap because it is architecturally identical to
XLM-R-large (same size, same tokenizer class) so it needs no COMET code change;
`google/rembert` and `facebook/xlm-roberta-xl` are documented alternatives
(set `encoder_model` accordingly). The encoder arm starts from **raw** encoder
weights (COMET's head is trained fresh), so its controlled baseline is XLM-R
trained the same way — not `wmt22-comet-da`.

```bash
# builds paragraph data if missing, then trains
VARIANT=paragraph    sbatch scripts/slurm_retrain_comet.sh
VARIANT=encoder_swap sbatch scripts/slurm_retrain_comet.sh
# controlled encoder baseline:
VARIANT=encoder_swap PRETRAINED=xlm-roberta-large RUN_NAME=xlmr-scratch \
    sbatch scripts/slurm_retrain_comet.sh
```

Resume after a timeout: `VARIANT=… RESUME=<ckpt> WANDB_RUN_ID=<id> sbatch …`.

### Evaluation — correlation vs length (`scripts/eval_length_correlation.py`)

For each model and each `(lp, k)` cell of the paragraph **val** set, predict and
report Pearson/Spearman/Kendall against the gold score. Headline plot: **Kendall
τ vs k**, one line per model. A model that generalises to long text degrades less
(or improves) as `k` grows.

```bash
python scripts/eval_length_correlation.py \
    --models wmt22=Unbabel/wmt22-comet-da \
             bio=$HOME/scratch/checkpoints/bio_mqm/comet-bio-mqm/4yeqp7cn/checkpoints/last.ckpt \
             paragraph=$HOME/scratch/checkpoints/retrain/paragraph/<run>/checkpoints/last.ckpt \
             infoxlm=$HOME/scratch/checkpoints/retrain/encoder_swap/<run>/checkpoints/last.ckpt \
    --data_dir ~/scratch/paragraph_mqm \
    --output results/retrain/length_correlation.json \
    --wandb_project comet-retrain
```

---

## Experiment A2 — sentence-fraction sweep (long-only training, frozen vs unfrozen)

**Question.** If COMET is continued **only on long text**, does it lose
sentence-level correlation with human judgement — and how much sentence-level
data does it take to keep both? Two encoder regimes separate "the head adapted"
from "the encoder itself moved": `FROZEN=1` trains the regression head only;
the default unfreezes the encoder after 0.3 epochs (the YAML schedule).

### Data — sentence-fraction mixes (`scripts/make_sentence_mix.py`)

From the unbalanced window pools (`{lp}_trainpool.csv`, written by
`prepare_paragraph_data.py`), build one training set per sentence share
f ∈ {0, 10, 20, 40, 60, 80, 100} %: `f·N` single sentences (k=1) plus the rest
long windows (k ∈ {2,3,4,6}, split evenly), at **constant total N per LP** —
composition is the only thing that varies, never the amount of data.
`frac000` is the long-only arm of the headline test; `frac100` is the
sentence-only control.

Built for verifiability (the mixes are the object under test):
- **nested sampling** — each (lp,k) pool is shuffled once with the seed and
  every mix takes a prefix, so two mixes differ only by prefix lengths;
- `mixes/MANIFEST.md` + `manifest.json` + per-mix `counts.json` record the
  source (Bio-MQM repo, window construction, scoring), the per-(lp,k) counts,
  the seed, and an md5 per CSV — regenerate with the same seed to audit;
- validation is **never mixed**: every arm is evaluated on the same shared
  `{lp}_val.csv` (paragraph windows = concatenations of consecutive segments
  of the same document, scored by the mean per-segment MQM penalty).

```bash
python scripts/make_sentence_mix.py --data_dir ~/scratch/paragraph_mqm
```

### Training (extends `scripts/slurm_retrain_comet.sh`)

```bash
for MIX in frac000 frac010 frac020 frac040 frac060 frac080 frac100; do
  for FROZEN in 0 1; do
    VARIANT=mix MIX=$MIX FROZEN=$FROZEN sbatch scripts/slurm_retrain_comet.sh
  done
done
```

Checkpoints land in `~/scratch/checkpoints/retrain/mix-<frac>[-frozen]/`, one
directory per arm so the 14 runs never collide. Same recipe as the `paragraph`
arm (continue `wmt22-comet-da`), only the train CSVs change; `FROZEN=1` sets
`nr_frozen_epochs=1000` + `keep_embeddings_frozen`, so the encoder never
unfreezes and only the regression head learns.

The overrides are written into a generated `$CKPT_DIR/config.yaml` rather than
passed as CLI flags (jsonargparse 3.13.1 is unreliable for list-valued
`init_args`), which also leaves a per-run record of exactly what was trained.

### Evaluation — three lenses per checkpoint

1. **Correlation vs length** (`scripts/eval_length_correlation.py`) — the k=1
   cells are the headline: does the long-only model (frac000) drop on
   sentence-level human correlation, and which fraction restores it?
2. **MetaDocEval** (`scripts/eval_metadoceval.py`) — document-level contrastive
   accuracy; see the dedicated section below.
3. **Block-xSIM++** (Experiment B) — pass the checkpoint:
   `RETRAIN_CKPT=.../mix-frac000/<run>/checkpoints/last.ckpt sbatch scripts/slurm_block_xsim.sh`.

### What each result would mean

- **frac000 k=1 τ drops below the wmt22 baseline** → training distribution
  drives sentence-level competence; the sweep's restoration point says how
  cheaply it is bought back.
- **frozen arm flat across fracs, unfrozen arm moves** → the length behaviour
  lives in the encoder weights, not the head — connects to the
  representation-analysis line.
- **MetaDocEval accuracy rises where the baseline COMET is flat** → paragraph
  training bought genuine discourse sensitivity, not just a rescaled score.

---

## Experiment A3 — MetaDocEval (`scripts/eval_metadoceval.py`)

**Question.** Does a COMET retrained on paragraph-level data actually detect
**discourse-level** errors, or does it only shift its score scale? Human
correlation cannot answer this: the perturbations here are designed to preserve
sentence-level fluency while breaking coherence across sentences.

### Test set and protocol

[MetaDocEval](https://github.com/nicolasdahan/metadoceval-testset) (Dahan,
Bawden & Yvon, EAMT 2026) — WMT24++ documents (en→fr/es/de, systems AYA23 and
GEMINI-1.5-PRO, ~11.3 segments per document) paired with perturbed variants:

| kind | perturbations |
|------|---------------|
| structural | `sentence_removal`, `sentence_repetition`, `sentence_shuffling`, `sentence_splitting` |
| lexical | `tense_consistency`, `lexical_consistency`, `conjunction_substitution`, `pronoun_swap_{singular,plural}` |

Scoring follows the paper: **SLIDE(w,1)** — every window of `w` consecutive
segments, stride 1, concatenated and scored; window scores averaged within a
document; **accuracy = P[score(original) > score(perturbed)]**, chance 50%, with
a paired two-sided t-test on per-document differences. Sweeping
`w ∈ {1,3,6,9}` separates "detects the error" from "dilutes it with context".

Two reading rules that are easy to get wrong:

- **`sentence_splitting` is quality-preserving by design.** High accuracy there
  is a *false-positive rate*, not a success — the script labels it as such in
  both the log and the plot. The paper's concerning finding is exactly that
  COMET/CometKiwi penalise splits *more* as context grows.
- **`doc_id` is local to each perturbation category**, not a global document id
  (verified: segment counts and `sys` text conflict across categories for the
  same id). Documents are only ever grouped within a category.

### Known gap in the released test set

Seven of the nine categories match the README's counts exactly, but two do not:

| category | README (commit `d3e8dc6`) | data, and README at `1ee6788` |
|----------|--------------------------:|------------------------------:|
| `sentence_shuffling` | 1,929 | 1,290 |
| `sentence_splitting` | 700 | **118** |

This is a README regression, not missing data. Commit `1ee6788` ("report actual
contrastive-pair counts") listed exactly the values the data contains; the later
`d3e8dc6` overwrote them with larger ones. The larger numbers look like
*attempted* perturbations rather than effective ones — shuffling a set of
positions leaves fixed points unchanged, and the splitting heuristic only fires
when a comma really separates two finite clauses — whereas the README defines
the column as `sys ≠ sys_perturbed` (`Levenshtein > 0`).

Consequence: splitting, the quality-preserving control, has 15–27 pairs per
(language pair × system) cell, so read it micro-averaged only.
`eval_metadoceval.py` **parses the clone's own README** and cross-checks it
against the data, warning on any mismatch and recording it under
`readme_mismatches` in the output JSON — so the check fixes itself if upstream
does, and catches a future regression. All reported numbers use the data.

### Run

```bash
# baselines only (clones the test set if missing)
sbatch scripts/slurm_metadoceval.sh
# every sentence-fraction checkpoint that exists, plus the baselines
SWEEP=1 sbatch scripts/slurm_metadoceval.sh
```

or directly:

```bash
python scripts/eval_metadoceval.py \
    --data_dir ~/scratch/metadoceval-testset \
    --models wmt22=Unbabel/wmt22-comet-da kiwi=Unbabel/wmt22-cometkiwi-da \
             longonly=$HOME/scratch/checkpoints/retrain/mix-frac000/<run>/checkpoints/last.ckpt \
    --windows 1 3 6 9 \
    --output results/metadoceval/accuracy.json --wandb_project comet-retrain
```

Parsing goes through the test set's **own** `scripts/load_data.py` (imported
from the clone, stdlib-only) rather than a reimplementation, and the clone's git
revision is recorded in the output JSON for provenance. Reference-free
checkpoints are detected automatically (the `ref` field is dropped). Window
triples are deduplicated before scoring — ~105k unique inputs instead of ~342k,
a 69% saving — and cached per model, so adding a checkpoint only scores the new
one. Plot: `results/metadoceval/plots/metadoceval_accuracy.png`, laid out like
Figure 1 of the paper (structural row, lexical row, dashed lines for
reference-free metrics).

### What each result would mean

- **Lexical accuracy rises with `w` for a retrained model but stays flat for
  `wmt22-comet-da`** → paragraph training bought real discourse sensitivity.
  In the paper only CometKiwi rises, and barely reaches chance at `w=9`.
- **Structural accuracy falls more slowly with `w`** → less dilution of a local
  error inside a long window; the same effect Experiment B measures on the
  encoder alone.
- **Splitting false-positive rate grows with `w`** → the model is penalising
  segmentation rather than quality, the failure mode the paper reports for
  both COMET (76%→89%) and CometKiwi (68%→85%).

---

## Experiment B — xSIM++ on concatenated blocks (`scripts/run_block_xsim.py`)

**Question.** Can COMET's encoder still spot a single perturbed sentence once it
is *diluted* inside a k-sentence block? This adapts xSIM++
(Chen et al. 2023, arXiv:2306.12907) from single sentences to blocks.

### Setup (`scripts/block_xsim_common.py`)

- FLORES+ articles → **non-overlapping** blocks of `k` consecutive sentences.
- **Query** = the source block (k source sentences concatenated in order).
- **Pool** = every true target block **+** hard negatives: for each block, apply
  **one** xSIM++ perturbation to **one** of its k sentences, leaving the other
  `k−1` intact.
- **Correct** = retrieving the concatenation of all k translations, in order,
  unperturbed. Retrieval uses the absolute-margin (plain cosine) rule, per the
  xSIM++ paper's footnote 6.

Sweeping `k ∈ {2,3,4,5}` turns "how sensitive is the encoder to one error?" into
"how fast does that sensitivity **dilute** with block length?".

### Perturbation categories (the three of xSIM++ §2.2)

| category | how |
|----------|-----|
| `causality` | antonym substitution, negation insertion/removal, modal strengthening (Tan et al. 2021) |
| `entity` | replace a named-entity-like span with one sampled from a corpus-wide same-language bank |
| `number` | replace digits / ordinals with different values |

**Deviation from the paper.** xSIM++ perturbs *English* using NLTK NER / spaCy /
WordNet. Here the perturbed side is whichever the pool is built from — by default
the **translations** (`--direction en2xx`) — so perturbations are implemented
self-contained and multilingually (no NLTK/spaCy/WordNet; the cluster sandbox
blocks the NLTK data download anyway). Entity detection is a casing + corpus-
frequency heuristic and is therefore **unavailable for case-less scripts (zh, ja,
th)**; `variant_stats` in the JSON reports per-category coverage so any gap is
explicit. Use `--direction xx2en` for the paper's original English-pool setup.

### Metrics (per encoder × language × k)

- `xsim_err` / `xsimpp_err` — error over the true-only pool vs the full pool.
- `error_breakdown` — of the mistakes, how many were "misaligned" (a *different*
  block) vs fooled by each perturbation category (xSIM++ Table 4 typology).
- `per_category_err`, `category_combos` — error with the pool restricted to one
  category, or each combination.
- `detection_rate` — `P[cos(query, gold) > cos(query, perturbed)]` over each
  block's own negatives (the push-apart probability), also split by category and
  by **position** of the perturbed sentence.
- `margin_vs_best_negative`, `margin_vs_best_own_perturbation`.

### Run

```bash
# inspect the generated hard negatives, no GPU:
python scripts/run_block_xsim.py --dry_run --langs de es fr ru --k_list 2 3 4 5

# full zoo on the cluster (COMET, bio-COMET, XLM-R, LaBSE, E5):
sbatch scripts/slurm_block_xsim.sh
# add a retrained checkpoint from Experiment A:
RETRAIN_CKPT=$HOME/scratch/checkpoints/retrain/paragraph/<run>/checkpoints/last.ckpt \
    sbatch scripts/slurm_block_xsim.sh
```

Plots in `results/block_xsim/plots/`: `xsim_vs_xsimpp`, `detection_vs_length`,
`error_by_category`, `detection_by_position`. Block embeddings are cached under
`results/block_xsim/emb_cache/`, so adding an encoder only embeds the new one.

### What each result would mean

- **xsim ≈ 0 but xsim++ rising with k** → the encoder aligns blocks fine but is
  progressively blind to a single buried error — the dilution the length-bias
  study is about.
- **aligners (LaBSE/E5) detect < COMET** → COMET's quality training bought
  error-sensitivity the pure aligners lack (the pull-together / push-apart
  contrast of the representation-analysis suite).
- **detection falling with the perturbed sentence's position** → the encoder
  under-weights later sentences in a long block.
