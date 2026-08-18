# SUMMARY — Exp 0: what aggregation function do QE metrics implement?

**Protocol.** Nested paired families of contiguous WMT24++ post-edit segments: every scale k is a prefix of the same k_max window; each unit's perturbation (severity × error-family, one edit per segment, length-preserving) is generated once and frozen at all scales; perturbation sets grow as monotone chains, so m ∈ {0..k} is covered exactly once per (family, condition, k) with counterbalanced positions. s_i = score of sentence i alone, S = score of the concatenated prefix; candidates fitted with affine calibration and compared by document-grouped CV R² (k ≥ 2).

## Verdicts

- **comet** (quality scale): best-fitting aggregation is **softmin** (CV R² = 0.820); power-mean exponent k=2: -0.71 [-1.01, -0.41], k=3: -0.65 [-0.95, -0.35], k=4: -0.74 [-0.94, -0.48], k=6: -0.82 [-1.12, -0.52]; position-invariant; p̂ stable across k.
- **cometkiwi** (quality scale): best-fitting aggregation is **softmin** (CV R² = 0.759); power-mean exponent k=2: -1.18 [-1.68, -0.68], k=3: -1.52 [-1.88, -1.12], k=4: -1.45 [-1.81, -1.05], k=6: -1.30 [-1.65, -1.00]; position-SENSITIVE; p̂ stable across k.

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