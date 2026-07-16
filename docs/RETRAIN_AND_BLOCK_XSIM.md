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
