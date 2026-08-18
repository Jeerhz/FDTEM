# SUMMARY — Exp 0: what aggregation function do QE metrics implement?

> ⚠️ Contains the **mock** metric (logic smoke test). Its paragraph scores were
> SYNTHESISED as a power mean with true p = -4.0; the fitted p̂ below must
> recover it. Mock numbers are a pipeline check, not a scientific result.

**Protocol.** Nested paired families of contiguous WMT24++ post-edit segments: every scale k is a prefix of the same k_max window; each unit's perturbation (severity × error-family, one edit per segment, length-preserving) is generated once and frozen at all scales; perturbation sets grow as monotone chains, so m ∈ {0..k} is covered exactly once per (family, condition, k) with counterbalanced positions. s_i = score of sentence i alone, S = score of the concatenated prefix; candidates fitted with affine calibration and compared by document-grouped CV R² (k ≥ 2).

## Verdicts

- **mock** (quality scale): best-fitting aggregation is **power_mean** (CV R² = 0.988); power-mean exponent k=2: -3.92 [-4.12, -3.72], k=4: -3.99 [-4.19, -3.79]; position-invariant; p̂ stable across k.

## Key outputs

- `analysis/candidate_fits.csv` — every candidate, every (metric, k≥2)
- `analysis/p_by_k.csv` — the headline exponent p̂ with bootstrap CIs
- `analysis/dilution.csv` + `figures/dilution_*.png` — the null-model curve Δ(m,k,severity), k=1 = undiluted effect
- `analysis/clean_curve.csv` — error-free prefix score vs k (concatenation bias)
- `analysis/residual_by_k.csv` — what aggregation does NOT explain (length effect)
- `figures/` — obs-vs-pred, p-by-k, candidate comparison

## Reproduce

```bash
python run.py --config config.yaml
```