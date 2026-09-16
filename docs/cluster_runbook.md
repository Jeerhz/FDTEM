# Cluster runbook

Operational notes for running the three parts on the cluster. The ground rules for
a Claude Code session are in `CLAUDE.md`; the science is in each part's README and,
for the pre-refactor history, in `docs/design_history.md`.

## Preflight

```bash
cd ~/FDTEM && git status --short
source ~/miniconda3/etc/profile.d/conda.sh && conda activate comet-bio
python -c "import comet, torch, common; print(comet.__version__, torch.cuda.is_available(), common.__file__)"
sinfo -s                          # the 'gpu' and 'cpu_devel' partitions
squeue -u "$USER"
wandb status                      # offline is acceptable, just note it
ls ~/.cache/huggingface/token     # gated FLORES+ / CometKiwi
df -h ~/scratch && du -sh ~/scratch/* 2>/dev/null | sort -h | tail
```

`comet` must come from site-packages (PyPI 2.2.7) and `common` from this repo; see the
install block of README.md if either is wrong.

## Disk budget

A COMET / XLM-R-large checkpoint is about 2.3 GB. Part 2 keeps `save_top_k=1` plus
`last.ckpt` per arm (about 5 GB per arm); a four-arm × two-family wave is about 40 GB
under `~/scratch/checkpoints/<ckpt_root>/`. Check before submitting a grid.

## Where each procedure lives

| part | build data | train | evaluate |
|---|---|---|---|
| 1 | `sbatch part1_block_alignment/slurm/build_pools.sh` | — | `sbatch part1_block_alignment/slurm/evaluate_{encoders,duel,comet_score}.sh` |
| 2 | `python -m part2_length_training.load_wmt_pools`, `load_heldout_sets`, `load_metadoceval` | `SUBMIT=1 part2_length_training/slurm/launch_arms.sh` | `sbatch part2_length_training/slurm/eval_{validation,metadoceval}.sh` |
| 3 | inside `finetune.sh` | `sbatch part3_biomed_finetune/slurm/finetune.sh` | `sbatch part3_biomed_finetune/slurm/evaluate.sh` |

Build data before submitting training: jobs submitted into an empty data directory
race to write the same CSVs.

## Failure playbook

- **Wall clock hit.** Part 2 arms are chained (`CHAIN` linked jobs with `RESUME=auto`),
  so the next link continues the same W&B run. For a single job:
  `RESUME=auto MODEL=… MIX=… sbatch part2_length_training/slurm/train.sh`.
- **CUDA OOM.** Lower `batch_size` in the part's base config
  (`part2_length_training/configs/comet_*.yaml`, `part3_biomed_finetune/configs/finetune.yaml`)
  and raise `accumulate_grad_batches` in `common/configs/trainer_wandb.yaml` by the same
  factor, so the effective batch and the comparability across arms are unchanged.
- **EarlyStopping fires immediately on resume.** Pass `--reset_early_stopping` to
  `common.train_comet`.
- **`make_mixtures` refuses an arm ("infeasible").** The constant-total design cannot be
  met from the pools; use `--total_policy pure`, a smaller `--total`, or `--allow_skip`
  and say so in the results. Never silently drop an arm.
- **MetaDocEval clone fails.** `python -m part2_length_training.load_metadoceval --data_dir …`
  or clone by hand; never hand-edit the JSON.
- **Checkpoint will not load ("hparams.yaml missing").** `common.comet_models.load_comet`
  rebuilds it from the checkpoint; `python -m common.train_comet` writes it after every run.

## What to report back

Short prose, not raw JSON: the headline number, its confidence interval (document
bootstrap), which arms are still running, and the job ids, checkpoint paths and result
JSON paths so everything can be re-derived. For part 2 in particular: k=1 Kendall τ of
each arm against the published metric, the held-out document numbers, and whether any
MetaDocEval accuracy rises with the window size (double-check before claiming it).
