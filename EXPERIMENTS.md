# FDTEM — length and domain sensitivity of learned MT metrics

Three experiments. Each is self-contained under `experiments/`, shares
`fdtem/`, and writes to `results/<experiment>/`.

| # | folder | question |
|---|--------|----------|
| 1 | [`experiments/domain_adaptation`](experiments/domain_adaptation) | What does finetuning COMET on an unseen domain (biomedical MQM) change? |
| 2 | [`experiments/length_isolation`](experiments/length_isolation) | Do encoders still spot a single error once it is buried in a longer block — with length as the *only* variable? |
| 3 | [`experiments/length_training`](experiments/length_training) | Does finetuning COMET on longer text help, and how should the mixture be composed? |

Experiment 3 evaluates its finetuned encoders with experiment 2's task, so the
two share `fdtem.encoders` and `fdtem.flores`.

## Shared library — `fdtem/`

| module | contents |
|---|---|
| `languages.py` | FLORES codes, the no-whitespace languages |
| `flores.py` | FLORES / FLORES+ loading, row-aligned, grouped into articles |
| `encoders.py` | the encoder zoo (`comet:`, `hf-mean:`, `labse`, `e5`) + embedding cache |
| `perturb.py` | sentence-level perturbations used to build hard negatives |
| `metrics.py` | xsim retrieval error |
| `comet_io.py` | checkpoint discovery, `hparams.yaml` repair, model loading, cached scoring |
| `stats.py` | correlations and **document-level** bootstrap |

The repo is installed editable (`unbabel_comet.pth`), so `import fdtem` works
from anywhere without touching the environment.

## Layout

```
fdtem/          shared library
experiments/    the three experiments (data prep, run, eval, slurm/, configs/)
comet/          vendored COMET, patched — the training library itself
configs/        COMET trainer / early-stopping / checkpoint configs
results/        JSON results + prediction caches
archive/        superseded code, kept for reference (see archive/README.md)
```

## Conventions

- Every experiment step is a plain `python experiments/<exp>/<step>.py --help`.
- Slurm wrappers live in `experiments/<exp>/slurm/` and only set environment
  and paths — the science stays in the Python entry points.
- Predictions are cached per (model, exact input rows); re-running an eval with
  one extra model only scores that model.
- Uncertainty is always bootstrapped over **documents**, never over examples:
  windows of one document share errors and text.
