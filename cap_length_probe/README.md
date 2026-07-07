# cap_length_probe — does COMET's encoder degrade with input length?

A reproducible benchmark measuring how the **COMET encoder's** text representations
behave as input length grows, **relative to the raw XLM-R backbone** COMET was
fine-tuned from. Downstream task: **Comparative Agendas Project (CAP) major-topic
classification** (21 classes). Both encoders are **frozen**; we probe them with a
lightweight linear classifier. The single independent variable is input length.

> **Hypothesis.** COMET's sentence-level QE fine-tuning specializes the encoder for
> short inputs, so probe accuracy from COMET features should degrade — or fail to
> improve — with length faster than from raw XLM-R. Because the two encoders share
> the **same backbone and tokenizer**, any gap in the accuracy-vs-length curve is
> attributable to COMET's fine-tuning, not architecture or size.

## Encoders (frozen, same backbone family)

| role | default pair (XLM-R-large) | alt pair (`--pair alt`, InfoXLM-large) |
|------|----------------------------|-----------------------------------------|
| `comet` | `Unbabel/wmt22-comet-da` | `Unbabel/wmt22-cometkiwi-da` |
| `raw`   | `FacebookAI/xlm-roberta-large` | `microsoft/infoxlm-large` |

The COMET encoder is the **underlying transformer extracted from the checkpoint**:

```python
from comet import download_model, load_from_checkpoint
m = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
hf_model  = m.encoder.model       # XLMRobertaModel
tokenizer = m.encoder.tokenizer   # XLMRobertaTokenizerFast
```

Both encoders run through **identical code** (`features.py`): `eval()`,
`torch.no_grad()`, fp16 on GPU, `output_hidden_states=True`, then **masked
mean-pool over the last hidden layer** (primary; `cls` and `all_layers_mean` behind
`features.pooling`). Features are z-scored with **train statistics only** in the
probe; optional L2-normalisation via `features.normalize_l2` (report both).
`run.py` prints a tokenizer-equality check so the token-length buckets are directly
comparable.

## Dataset & provenance

**Primary source — "Beyond Token Limits" (Sebők et al. 2025)**, the long-document
CAP release that specifically targets >512-token documents across 5 languages
(`en`, `hu`, `nl`, `fr`, `it`). Paper: arXiv [2509.10199](https://arxiv.org/abs/2509.10199);
replication on OSF [`w3fjn`](https://osf.io/w3fjn/). The labelled CSVs live in the
project's OneDrive store (not reachable via the public OSF API), so **stage them on
the cluster** and point `data.btl_dir` (or `$BTL_DIR`) at them. Expected schema
(from `00_data_clean.ipynb`): `text`, `label` (0-indexed CAP code), optional
`label_cap` (original CAP code), `language` (full names), and
`token_count_xlm_roberta_large`. `data.py` maps `label`→CAP major-topic code, drops
*No Policy Content* (999), maps language names→ISO, dedups, **keeps only docs with
≥512 XLM-R tokens** (the paired-design gate), and builds its own stratified split.

**Fallbacks** (auto, logged): `poltextlab_hf` (a small HF CAP set — short titles,
logic only) and `synthetic` (a seeded, torch-free multilingual generator used by the
local logic smoke test). If `btl_dir` is missing, `data.py` logs the miss and falls
back to `synthetic` rather than crashing.

CAP major topics targeted (21): Macroeconomics, Civil Rights, Health, Agriculture,
Labor, Education, Environment, Energy, Immigration, Transportation, Law & Crime,
Social Welfare, Housing, Domestic Commerce, Defense, Technology, Foreign Trade,
International Affairs, Government Operations, Public Lands, Culture.

## Length control (paired, in-domain prefixes)

Buckets `L ∈ {16, 32, 64, 128, 256, 512}` tokens (512 = XLM-R context limit). For
each retained document we feed the **first L tokens** (`max_length=L,
truncation=True`); the *same* document contributes to every bucket, so length is the
only thing that changes — label and topic are constant, prefixes stay coherent and
in-domain.

## Probing protocol

- **Classifier:** multinomial logistic regression (`class_weight="balanced"`,
  high `max_iter`) on standardized frozen features. 1-hidden-layer MLP behind
  `probe.classifier: mlp`.
- **Seeds:** ≥5; each seed draws a fixed-fraction train sub-sample
  (`probe.train_subsample_frac`) → mean ± std across seeds.
- **Regime 1 (primary):** train *and* test at each L → accuracy/F1 vs L per encoder.
- **Regime 2 (stability):** train at the shortest L, evaluate at all L.
- **Cache:** pooled features are extracted once per (encoder, document-set, L) and
  reused across probes/seeds (`results/feature_cache/`, keyed by content hash).

## Run it

```bash
# 0. install (into the COMET cluster env)
pip install -r requirements.txt

# 1. full run — one command reproduces every table and figure
python run.py --config config.yaml                 # default XLM-R pair
python run.py --config config.yaml --pair alt       # InfoXLM pair (if time permits)

# 2. cluster (Cleps / SLURM), submit from the repo root
BTL_DIR=$HOME/scratch/cap_btl sbatch cap_length_probe/slurm_cap_length.sh

# 3. smoke test (real encoders): 1 language, ~200 docs, 2 buckets
python run.py --config config.yaml --smoke

# 4. local LOGIC smoke test (no GPU, no downloads): mock features + synthetic data
python run.py --config config_smoke.yaml            # writes results_smoke/ (clearly marked)
```

Individual stages are runnable too: `python data.py`, `python features.py`,
`python probe.py`, `python analyze.py`, `python plots.py` (all take `--config`).

## Outputs (`results/`)

| path | what |
|------|------|
| `results.csv` | tidy: `encoder, lang, L, seed, acc, macro_f1` (+ `regime`, meta) |
| `predictions.csv.gz` | seed-0 per-item predictions (for bootstrap CIs) |
| `analysis/{summary,slopes,gap_by_L,per_language,regime2,geometry}.csv` | metrics, log-length slopes, COMET−XLM-R gap, geometry |
| `figures/accuracy_vs_length.png` | acc & macro-F1 vs L, one line per encoder, CI bands |
| `figures/gap_vs_length.png` | COMET − XLM-R gap vs length |
| `figures/per_language_accuracy.png` | per-language small multiples |
| `figures/regime2_stability.png` | train-short / test-across-length |
| `figures/geometry_vs_length.png` | norm / anisotropy / drift vs length |
| `SUMMARY.md` | one-paragraph answer + the two key plots |

`results_smoke/` holds the output of the local **logic** smoke test (mock backend);
its `SUMMARY.md` carries a loud "not a scientific result" banner.

## Reproducibility & acceptance

- Identical pooling, split, documents, probe, and seeds across both encoders;
  tokenizer-equality asserted at runtime.
- All seeds set; `run.py` logs library versions + GPU.
- Sanity checks printed before training: #docs per (lang, label), label balance,
  feature dim (expect 1024 for `*-large`), NaN/inf checks, and confirmation that
  **both encoders yield the same #items per bucket**.
- One command (`python run.py`) reproduces every table and figure; `results.csv` is
  populated for both encoders across all L and seeds; `SUMMARY.md` answers the
  question with the accuracy-vs-length and gap-vs-length plots; a documented
  fallback covers an unavailable CAP source.

Requirements: see [`requirements.txt`](requirements.txt) (torch, transformers,
unbabel-comet, scikit-learn, datasets, pandas, numpy, scipy, matplotlib, pyyaml).
