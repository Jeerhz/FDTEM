# Representation-analysis suite (E1–E5)

How does training XLM-R to **predict translation quality** (COMET) reshape its
multilingual representation space — and what happens when the text is longer than
the short segments COMET was trained on? This suite answers that by putting COMET
side-by-side with the encoders it is *not*: a raw XLM-R, and two pure
parallel-sentence aligners (LaBSE, multilingual-E5).

## The central question

> COMET is trained from an old multilingual encoder (XLM-R) but produces
> embeddings used to predict quality. One could also use those embeddings
> directly for cross-lingual similarity. The interesting tension: aligners
> (LaBSE/LASER/SONAR/E5) only **pull parallel sentences together**, whereas a
> quality metric must *also* **push apart pairs that are nearly parallel but
> contain errors**. How does COMET's encoder differ from the aligners?

Every experiment is framed around that pull-together / push-apart contrast.

## The encoder zoo (one identical interface — `repana_common.py`)

| spec | what it is | role |
|------|-----------|------|
| `comet:Unbabel/wmt22-comet-da` | COMET full pipeline (layerwise-attn + avg pool) | the subject |
| `comet:<bio.ckpt>` | your Bio-MQM-finetuned COMET | domain-adapted subject |
| `hf-mean:xlm-roberta-large` | raw XLM-R, mean pooled | the untrained control / COMET's backbone |
| `labse` | LaBSE (`setu4993/LaBSE`) | contrastive aligner |
| `e5` | `intfloat/multilingual-e5-base` | modern contrastive aligner |

All embeddings L2-normalised. Data = **FLORES-200 devtest** (multi-parallel, 1012
sentences/language), so every metric is computed on identical content across
languages and encoders.

## Experiments

| # | script | metric | reads on |
|---|--------|--------|----------|
| **E1** | `run_e1_e3_retrieval.py` | xsim retrieval error (en↔xx) | pull-together / alignment |
| **E3** | *(same script)* | xsim vs pseudo-doc length k=1,2,4,8 | behaviour past sentence scale |
| **E2** | `run_e2_error_sensitivity.py` | P[cos(src,tgt⁺)>cos(src,tgt⁻)] on injected errors | **push-apart** + the trade-off plot |
| **E4** | `run_e4_geometry.py` | anisotropy · language-id probe · alignment gap · COMET layer weights | shape of the space |
| **E5** | `run_e5_cka.py` | per-layer linear CKA (XLM-R→COMET→COMET-bio) + output-space CKA | *where* training moved reps |

**E3 pseudo-paragraphs**: consecutive same-article FLORES sentences are
concatenated (FLORES rows are in document order with a `URL` column marking
articles), keeping them parallel across languages. All encoders truncate at 512
tokens identically, so the question is how well each *uses* multi-sentence context.

**E2 hard negatives** (xsim++ style, language-agnostic): `number` (digit swap),
`delete` (omission), `swap` (adjacent reorder), `replace` (mistranslation). For
no-space languages (zh/ja/th) perturbations are character-level.

## Running on Cleps

Full suite (auto-discovers the Bio-MQM checkpoint under `~/scratch/checkpoints/bio_mqm`):

```bash
sbatch scripts/slurm_repana.sh
```

Single experiment (pass-through mode):

```bash
sbatch scripts/slurm_repana.sh python scripts/run_e2_error_sensitivity.py \
    --encoders comet:Unbabel/wmt22-comet-da labse e5 --langs en de fr ru zh
```

Useful env vars: `BIO_CKPT=/path/model.ckpt`, `LANGS="en de es fr ru zh ar hi"`,
`CONDA_ENV=comet-bio`, `WANDB_PROJECT=comet-repana`, `WANDB_MODE=offline`.

Outputs: `results/repana/*.json` + `results/repana/plots/*.png`. Sentence
embeddings are cached under `results/repana/emb_cache/` and shared across E1/E2/E4,
so reruns are cheap.

## What each result would mean

- **E1 high, E2 ~0.5** → a pure aligner: parallel sentences nearest, but blind to
  errors. Expected for LaBSE/E5.
- **E1 lower, E2 high** → COMET's signature: it sacrifices some raw alignment to
  encode error structure. This is the hypothesis the supervisor flagged.
- **E3** → if COMET's curve falls off faster than the aligners' as k grows, that
  quantifies the "trained on short segments" limitation.
- **E4 language-probe** → lower = more language-agnostic. Tells whether COMET made
  the space more or less language-separable than XLM-R / the aligners.
- **E5** → a CKA dip at the top layers for COMET vs XLM-R localises the change to
  the head region; a dip reaching the *lower* layers for COMET-bio vs COMET is a
  catastrophic-forgetting / multilinguality-loss signature.

## Notes / extensions
- LASER3 & SONAR were left out (fragile fairseq2 installs on the cluster). The
  zoo is one `build_embedder` branch away from adding them.
- To add languages, extend `--langs` (FLORES codes live in `FLORES_CODE`).
