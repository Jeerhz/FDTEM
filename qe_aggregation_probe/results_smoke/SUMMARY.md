# SUMMARY — Exp 0: what aggregation function do QE metrics implement?

> ⚠️ Contains the **mock** metric (logic smoke test). Its paragraph scores were
> SYNTHESISED as a power mean with true p = -4.0; the fitted p̂ below must
> recover it. Mock numbers are a pipeline check, not a scientific result.

**Protocol.** k-sentence paragraphs from contiguous, same-document WMT24++ post-edits; m ∈ {0..k} sentences perturbed with controlled severity×family errors (positions counterbalanced); s_i = score of sentence i alone, S = score of the concatenated paragraph; candidate aggregation functions fitted with affine calibration and compared by document-grouped cross-validated R².

## Verdicts

- **mock** (quality scale): best-fitting aggregation is **power_mean** (CV R² = 0.987); power-mean exponent k=2: -4.02 [-4.22, -3.82], k=4: -3.85 [-3.95, -3.75]; position-invariant; p̂ stable across k.

## Key outputs

- `analysis/candidate_fits.csv` — every candidate, every (metric, k)
- `analysis/p_by_k.csv` — the headline exponent p̂ with bootstrap CIs
- `analysis/dilution.csv` + `figures/dilution_*.png` — the null-model curve Δ(m,k,severity)
- `analysis/residual_by_k.csv` — what aggregation does NOT explain (length effect)
- `figures/` — obs-vs-pred, p-by-k, candidate comparison

## Reproduce

```bash
python run.py --config config.yaml
```