# 1 — Domain adaptation

Finetune COMET on **Bio-MQM** (biomedical MQM annotations, Zouhar et al. 2024)
and compare the adapted model with the public one. The domain is deliberately
far from the WMT news/general data COMET was trained on.

## Run

```bash
python experiments/domain_adaptation/prepare_data.py --output_dir ~/scratch/bio_mqm_data
sbatch experiments/domain_adaptation/slurm/finetune.sh
```

`configs/finetune.yaml` holds the model and schedule; the shared trainer /
early-stopping / checkpoint configs live in the repo-root `configs/`.

## Compare the models

Both the adapted and the public checkpoint are scored with experiment 3's
evaluator, which reports agreement with human judgement per input length:

```bash
python experiments/length_training/eval_correlation.py \
    --models base=Unbabel/wmt22-comet-da bio=<checkpoint> \
    --data_dir <a directory of *_val.csv>
```

## Note

The ≥10-epoch Bio-MQM checkpoint is `last.ckpt`, not the `epoch=N` files —
early stopping kept the older epochs as "best" on the in-domain validation set.
