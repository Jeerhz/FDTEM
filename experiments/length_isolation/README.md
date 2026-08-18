# 2 — Isolating length

Take FLORES+ articles, concatenate `k` consecutive sentences into a block, and
ask an encoder to retrieve the correct translation of that block among hard
negatives that differ by **one** perturbed sentence (xSIM++ categories:
causality, entity, number).

As `k` grows the error is diluted — but so does everything else, which is why
the design controls the confounds instead of just growing `k`.

## Controlling for everything but length

| confound | control |
|---|---|
| the candidate pool grows with `k` | `run_duel.py` pins the pool to gold + **D** single-error negatives at every `k`, and averages over all `C(m, D)` subsets in closed form |
| blocks overlap | blocks are non-overlapping (stride = `k`) |
| longer blocks come from longer articles | per-`k` block counts are reported; long `k` is rare and must not be compared without CIs |
| "is it length or is it content?" | `run_matched_core*.py` keeps the **same** core sentence and only changes the filler around it (inert / neutral / distractor / natural) and its position |

## Run

```bash
sbatch experiments/length_isolation/slurm/xsim.sh          # dilution curve over k
sbatch experiments/length_isolation/slurm/duel.sh          # fixed-pool control
sbatch experiments/length_isolation/slurm/matched_core.sh  # same core, growing filler
```

Encoders are named uniformly: `comet:<hub-id-or-ckpt>`, `hf-mean:<hf-id>`,
`labse`, `e5`. Pass a checkpoint from experiment 3 to test a finetuned encoder.

## Files

| file | role |
|---|---|
| `blocks.py` | blocks, perturbations, candidate pools, evaluation |
| `run_xsim.py` | the dilution curve |
| `run_duel.py` | fixed-pool duel — the candidate-count control |
| `matched_core.py`, `run_matched_core*.py` | matched-core probes, incl. per-layer |
