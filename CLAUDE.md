# Working in this repository

Three research parts (`part1_block_alignment`, `part2_length_training`,
`part3_biomed_finetune`) plus `common/`; see README.md for what each does.

## Ground rules

- Everything that needs a GPU or the data runs on the cluster (`conda activate comet-bio`).
  Nothing is installed locally: use a venv with pydantic/pandas/numpy/matplotlib for models,
  figures, pages and `pytest tests`.
- Do not commit or push unless asked. Never delete or overwrite anything under `~/scratch`.
- Never fabricate results: if a job is queued or running, say so; do not read numbers off
  partial logs. Long jobs are `sbatch`, not foreground; report job ids and real paths.
- If a job fails, read `logs/<jobname>-<jobid>.{out,err}` before resubmitting.

## Code rules

- Scripts run from the repo root as `python -m <package>.<script>`; absolute imports only
  (`from common.x import …`); no sibling imports, no `sys.path` edits; the repo root is
  `common.paths.ROOT`.
- Results JSON go through the pydantic models in `<package>/models.py`; keep the models
  plain mirrors of the files (no validators that reshape data).
- Slurm scripts: `#SBATCH` header, `source common/cluster_env.sh`, env-var tunables with
  defaults, one `srun python -m …`; no heredocs.
- Keep code short and explicit; file names say what a script does; comments only where the
  reason is not obvious. Do not change a protocol, a default or a number without saying so.
