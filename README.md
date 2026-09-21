# FDTEM — length and domain sensitivity of learned MT metrics

COMET-style metrics are trained on single sentences and applied to paragraphs and
documents. This repository holds the three experiments of the FDTEM internship that
ask what that costs, each self-contained in its own package, plus a `common/`
library they share.

| part | package | question | entry points |
|---|---|---|---|
| 1 | [`part1_block_alignment/`](part1_block_alignment/README.md) | Concatenate FLORES+ sentences into blocks of k sentences and add one xSIM++ perturbation to one sentence: can an encoder (cosine) or a metric (COMET score) still tell the reference block from the perturbed one as k grows? | `load_flores` → `build_blocks` → `perturb` → `evaluate_encoders` / `evaluate_duel` / `evaluate_comet_score`; one error per sentence: `perturb_every_sentence` → `evaluate_every_sentence` |
| 2 | [`part2_length_training/`](part2_length_training/README.md) | Continue COMET-DA / CometKiwi on training mixes that differ only in composition (sentences, concatenated windows, native documents): what does it buy on validation, held-out and MetaDocEval? | `load_wmt_pools` → `make_mixtures` → `train` → `eval_validation` / `eval_length_profile` / `eval_metadoceval` / `analyze` |
| 3 | [`part3_biomed_finetune/`](part3_biomed_finetune/README.md) | Fine-tune COMET on Bio-MQM (biomedical MQM annotations) and compare it with the published model per language pair. | `load_bio_mqm` → `slurm/finetune.sh` → `evaluate` → `upload_to_huggingface` |

Every script is run from the repository root as `python -m <package>.<script> --help`;
its slurm wrapper lives in `<package>/slurm/` and only sets the environment. Results
(JSON + plots) and figures (script + PDF/PNG) live inside the part that produced them.
`report/` holds the internship report and the HTML pages that read all parts.

## Layout

```
common/                   shared library: paths, HF/W&B auth, FLORES loader, encoder zoo,
                          COMET loading/scoring with caches, stats, training config + trainer
part1_block_alignment/    part 1: models.py, scripts, slurm/, data/ (ignored), results/, figures/
part2_length_training/    part 2: models.py, arms.py, scripts, configs/, slurm/, results/, figures/
part3_biomed_finetune/    part 3: models.py, scripts, configs/, slurm/, results/
report/                   main.tex + partie1_longueur.tex, figure and page builders
docs/                     cluster_runbook.md (operations), design_history.md (pre-refactor design notes)
tests/                    the committed result JSONs round-trip through their pydantic models
```

## `common/`

| module | contents |
|---|---|
| `paths.py` | `ROOT`, `SCRATCH` (`~/scratch`, override with `FDTEM_SCRATCH`), `CHECKPOINTS` |
| `auth.py` | `hf_token()` (token resolution + HF cache dirs), `init_wandb()`, `wandb_logger()` with offline fallback |
| `cluster_env.sh` | the one shell block every slurm script sources: repo root, conda/venv, HF, W&B |
| `flores.py` | `FloresCorpus` (pydantic), `load_flores("plus"\|"raw", …)`, language codes, `joiner()` |
| `encoders.py` | embedder zoo `comet:<id\|ckpt>`, `hf-mean:<id>`, `labse`, `e5`, with an on-disk embedding cache |
| `comet_models.py` | `resolve_checkpoint`, `load_comet`, `uses_reference`, `score` (prediction cache keyed by rows and checkpoint fingerprint), `best_checkpoint`, `write_hparams` |
| `stats.py` | `CorrelationCell`, `correlations`, document-level `tau` and `bootstrap_delta` |
| `train_config.py` | a self-contained training YAML from a base config; refuses more than one train file |
| `train_comet.py` | `train(...)`: base checkpoint, resume (`auto` continues the arm's last run), W&B logger, `hparams.yaml` |
| `configs/` | the Lightning trainer, early-stopping and checkpoint blocks shared by parts 2 and 3 |

## Install (cluster)

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate comet-bio
pip uninstall -y unbabel-comet          # the old editable install of the vendored copy
pip install "unbabel-comet==2.2.7"      # COMET now comes from PyPI
pip install -e ".[spacy]"               # this repo (editable) + the spaCy perturbation backend
python -m spacy download de_core_news_sm   # es fr ru zh en likewise; part 1 only
python -c "import comet, common; print(comet.__version__, common.__file__)"
hf auth login        # then accept the terms of openlanguagedata/flores_plus once
wandb login          # optional: jobs fall back to WANDB_MODE=offline
```

Slurm scripts source `common/cluster_env.sh`, which activates the env, redirects the HF
caches to `~/scratch/hf_cache`, re-exports the HF token and switches W&B offline when
needed. See [`docs/cluster_runbook.md`](docs/cluster_runbook.md) for the preflight checks.

Locally (no GPU) only the pure-Python side runs: a venv with `pydantic pandas numpy
scipy pyyaml matplotlib pytest` is enough for the models, the figure scripts, the pages
and `pytest tests`.

## Conventions

- Results are written and read through the pydantic models of `<package>/models.py`
  (`model_dump_json` / `model_validate_json`); figure scripts never re-derive JSON shapes.
- Predictions are cached per (model label, exact input rows) and validated against the
  checkpoint fingerprint, so re-running an evaluation only scores what changed; caches
  are never committed.
- Uncertainty is bootstrapped over **documents**, never over examples.
- A rank correlation is never the whole answer: part 2 also reports the distribution
  of the scores per length (`eval_length_profile`).
- A training mix built from data an evaluation set also contains carries a
  `CONTAMINATED` marker, `contaminated: true` in its manifest and a W&B tag; its
  correlation numbers are not results (`make_mixtures --arms uncontrolled`).
- One train file per run: COMET reads `train_data[epoch % len]` and Lightning never
  rebuilds the loader, so `common/train_config.py` refuses multi-file lists.

## Report

```bash
python -m part1_block_alignment.figures.make_figures
python -m part2_length_training.figures.make_figures
python report/figures/make_report_figures.py && python report/figures/make_answer_figures.py
python -m report.make_status_page   # also make_deck_length, make_deck_page, make_answer_page
cd report && latexmk -pdf main.tex
```
