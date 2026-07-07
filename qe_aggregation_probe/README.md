# qe_aggregation_probe — Exp 0: estimate the aggregation function (the null model)

**Question.** If a k-sentence text contains m degraded sentences, how does the
paragraph-level QE score depend on the sentence-level scores?

```
score(paragraph) ≈ f(score(sent_1), ..., score(sent_k))     — which f?
```

Mean? Minimum (one nonsense sentence sinks everything)? A sum of MQM penalties?
We fit, among others, the **generalized power mean**

```
M_p(s) = ( (1/k) Σᵢ sᵢᵖ )^(1/p)
```

whose exponent p interpolates continuously between the arithmetic mean (p = 1),
the geometric mean (p → 0) and the minimum (p → −∞). **The fitted p is the
answer**, and the dilution curve Δ(m, k, severity) becomes the null model for
the rest of the length-bias study: any paragraph-score behaviour f can explain
is "just aggregation"; the frozen-f residual that grows with k is the candidate
length/encoder effect.

Natural sentences all score high and alike, so f is unidentifiable from clean
text — we inject a **controlled number m of degraded sentences** per block to
spread the part scores (cf. Zhang et al. 2026, arXiv:2510.22028; Dahan, Bawden
& Yvon 2026, MetaDocEval).

## Protocol

1. **Data** — WMT24++ (`google/wmt24pp`) human post-edits = the error-free
   baseline, en→{de, zh, es, fr} by default. Units are the aligned dataset
   segments (optionally bertalign-resegmented sentences, `data.resegment:
   bertalign`). Only **contiguous same-document** units are concatenated;
   blocks are non-overlapping k-unit windows, k ∈ {2, 3, 4, 6}. Paragraphs
   over 500 tokens are dropped (512-token encoder limit). A FLORES-200 source
   is available as a robustness set (`data.source: flores`).
2. **Perturbations** — for each block and each condition (minor/major ×
   accuracy/fluency) every unit gets one perturbed variant, generated with the
   published error-injection prompts of Zhang et al. (App. A.4) through any
   OpenAI-compatible endpoint (`perturb.backend: llm`), or with deterministic
   length-preserving rule edits (`rule`) for smoke tests. QC enforces minimal,
   **length-preserving** edits so quality is dissociated from length; a
   (block, condition) cell is dropped whole if any unit fails QC.
3. **Items** — per block: m = 0 (shared) and m = 1..k per condition, with the
   perturbed **positions counterbalanced deterministically**
   (`combinations(range(k), m)[block_idx mod C(k,m)]`).
4. **Scoring** — s_i = score of sentence i **alone** (single-sentence input);
   S = score of the concatenated paragraph. Metrics probed independently:
   CometKiwi (`Unbabel/wmt22-cometkiwi-da`), COMET (`wmt22-comet-da`, ref =
   the unperturbed post-edit), MetricX-24-QE (error scale — fitted on the
   penalty scale, where mean/sum/max-penalty are the natural candidates).
   All scores are disk-cached; reruns are free.
5. **Fit** — candidates: mean, min, max, median (+ sum on the penalty scale),
   power mean M_p (free p), soft-min/log-sum-exp (free temperature). Every
   candidate gets the same affine calibration S ≈ a + b·g(s) so AIC/BIC compare
   *shapes*. Model selection by GroupKFold-by-document CV R²; bootstrap CIs for
   p resampled by document. Note: within a fixed k, `sum` is affine-equivalent
   to `mean`; they separate only in the pooled fit across k
   (`candidate_fits_pooled.csv`).
6. **Controls** — position test (is f symmetric? residual ~ position of the
   m=1 error); severity/family interaction (p per condition); stability of p
   across k; frozen-f residual by k (the bridge to the length experiment);
   optional MixedLM on residuals.

## Run

```bash
# logic smoke test — no GPU, no API. The mock metric SYNTHESISES paragraph
# scores as a power mean with known true_p; the pipeline must recover it.
python run.py --config config.yaml --smoke

# full run (cluster). Needs $OPENAI_API_KEY (or any OpenAI-compatible endpoint
# via $PERTURB_BASE_URL + $PERTURB_MODEL) for the LLM perturbations.
python run.py --config config.yaml

# single metric / rule-based perturbations (free, lower-bound sanity check)
python run.py --config config.yaml --metrics cometkiwi --perturb_backend rule
sbatch slurm_agg_probe.sh
```

MetricX-24-QE is disabled by default; enable it in `config.yaml` after
`pip install git+https://github.com/google-research/metricx` and spot-check a
few scores against upstream `predict.py`.

## Outputs

| file | content |
|---|---|
| `results/items.csv` | tidy dataframe: metric, lang, k, m, severity, family, positions, s_1..s_k, S, token counts, truncated flag |
| `results/analysis/candidate_fits.csv` | all candidates × (metric, k): R², CV-R², AIC/BIC, p̂ + CI |
| `results/analysis/p_by_k.csv` | headline power-mean exponent per (metric, k) |
| `results/analysis/p_by_condition.csv` | p̂ split by severity and error family |
| `results/analysis/position_test.csv` | symmetry control (slope, p-value, ΔR²) |
| `results/analysis/residual_by_k.csv` | S − f̂ with f frozen at the smallest k |
| `results/analysis/dilution.csv` | Δ(m, k, severity, family) — the null-model curve |
| `results/figures/` | obs-vs-pred, p-by-k, candidate CV comparison, dilution, frozen residual |
| `results/SUMMARY.md` | one verdict sentence per metric |

## Design notes

- **Determinism**: one global seed drives block sampling, counterbalancing and
  rule perturbations; LLM outputs and metric scores are content-hash cached, so
  a rerun reproduces the same items without new API/GPU work.
- **No candidate is hard-coded as ground truth**: all fits are reported, the
  "winner" is picked by grouped CV only.
- The human-annotator arm of Exp 0 (small sample: rate paragraphs with 1
  absurd sentence out of k, compare p_human vs p_metric) reuses `items.csv`
  directly — export the m ∈ {0, 1} subset for annotation.
