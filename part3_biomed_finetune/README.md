# Part 3: COMET fine-tuning on Bio-MQM

Fine-tune `Unbabel/wmt22-comet-da` on the biomedical MQM annotations of
[Bio-MQM](https://github.com/amazon-science/bio-mqm-dataset) (Zouhar et al. 2024)
and compare the adapted checkpoint with the public one. Secondary to parts 1 and 2:
it reproduces the paper's setting and provides the fine-tuned encoder used in part 1.

```
load_bio_mqm.py           dataset clone -> <lp>_{train,val}.csv + all_{train,val}.csv
configs/finetune.yaml     RegressionMetric block (sub-configs in common/configs/)
slurm/finetune.sh         data -> common.train_comet -> evaluate.py
evaluate.py               base vs bio per language pair -> results/correlation.json + plot
slurm/evaluate.sh         evaluate.py alone
upload_to_huggingface.py  publish a checkpoint on the Hub
models.py                 BioMqmRow, BioEvalResults, MQM_WEIGHTS, LANG_PAIRS
```

## Data

`load_bio_mqm.py` clones (or pulls) the dataset, converts span annotations to a
segment penalty (critical -25, major -5, minor -1, averaged over annotators),
z-scores then sigmoids the penalty per language pair, and splits by document with
the dataset's `doc_id_splits.json`: dev documents -> train, test documents -> val.
Rows without a reference are dropped. CSV columns are the `BioMqmRow` fields
(`src, mt, ref, score, lp, system, doc_id, seg_id`); COMET reads the first four,
`evaluate.py` groups by `doc_id`.

```bash
python -m part3_biomed_finetune.load_bio_mqm --output_dir ~/scratch/bio_mqm [--lang_pairs en-fr fr-en]
```

## Authentication (once per machine)

```bash
hf auth login   # Hub downloads and the upload step
wandb login     # training curves; without it jobs run with WANDB_MODE=offline
```

`common/cluster_env.sh` redirects `HF_HOME` to scratch and re-exports the token.

## Run

```bash
sbatch part3_biomed_finetune/slurm/finetune.sh
```

The job writes the data if `all_train.csv` is missing, trains with
`common.train_comet`, then evaluates. Tunables (environment variables):

| variable | default | |
|---|---|---|
| `DATA_DIR` | `$FDTEM_SCRATCH/bio_mqm` | CSVs |
| `CKPT_DIR` | `$FDTEM_SCRATCH/checkpoints/bio_mqm` | run directory |
| `RUN_NAME` | `comet-bio-mqm-<date>` | W&B run name |
| `WANDB_PROJECT` | `comet-bio-mqm` | |
| `MAX_EPOCHS` / `PATIENCE` / `SEED` | 20 / 5 / 42 | |
| `RESUME` | empty | `auto` (newest `last.ckpt` of `CKPT_DIR`, same W&B run) or a `.ckpt` |
| `LANG_PAIRS` | empty = all ten | e.g. `"en-fr fr-en"` |

Example: `MAX_EPOCHS=1 LANG_PAIRS="en-fr" sbatch part3_biomed_finetune/slurm/finetune.sh`.

The base checkpoint is downloaded by `common.comet_models.resolve_checkpoint`;
`configs/finetune.yaml` keeps the model and learning rates, the launcher passes the
data paths. The old config listed the ten per-pair CSVs under `train_data`, but COMET
reads only `train_data[0]` (see `common/train_config.py`), so training silently used
one language pair. The config now points at `all_train.csv` / `all_val.csv`, and
`train_config` refuses multi-file lists.

## Evaluate

```bash
sbatch part3_biomed_finetune/slurm/evaluate.sh                 # BIO_CKPT=auto
python -m part3_biomed_finetune.evaluate --bio <ckpt> --data_dir ~/scratch/bio_mqm
```

Per language pair: Pearson / Spearman / Kendall of both models, mean over pairs,
and a paired bootstrap over documents (`doc_id`) of delta Kendall (bio - base).
Output: `results/correlation.json` (`BioEvalResults`) and
`results/plots/kendall_by_lp.png`. See `results/README.md` for the committed file.

## Upload

```bash
python -m part3_biomed_finetune.upload_to_huggingface --checkpoint auto \
    --repo_id <user>/comet-bio-mqm --run_name <RUN_NAME> [--private] [--wandb_run_id <id>]
```

Validates the checkpoint, exports it with `hparams.yaml` and the HF-format encoder,
writes a model card and uploads the folder. The raw `.ckpt` is included, so the model
loads with `load_from_checkpoint(download_model("<user>/comet-bio-mqm"))`.

## Checkpoints

```
~/scratch/checkpoints/bio_mqm/
  config.yaml, run_name
  <WANDB_PROJECT>/<run-id>/
    hparams.yaml               written after training (needed by load_from_checkpoint)
    checkpoints/
      epoch=N-step=S-val_kendall=0.xxx.ckpt   best by val_kendall (save_top_k 1)
      last.ckpt
```

`--bio auto` and `--checkpoint auto` pick the highest `val_kendall` file, falling back
to `last.ckpt`. Note: on the run of 2026-08 the useful (10+ epoch) checkpoint was
`last.ckpt`, not the `epoch=N` file, because early stopping kept an earlier epoch as
"best" on the in-domain validation set; pass the path explicitly in that case.
A fresh run moves a previous run directory aside as `bio_mqm.superseded-<timestamp>`.
