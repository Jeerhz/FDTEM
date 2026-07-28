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

## Nested paired design — the four control guarantees

The sampling unit is a **family**: a contiguous window of k_max = max(k_list)
aligned units. Every scale k is the k-prefix of the same family.

1. **Length is the only variable across scales.** Perturbations are
   *inherited*, never regenerated: the perturbed variant of unit i under
   condition c is produced once per family and frozen, so a larger prefix
   contains the *same* perturbation instances as the smaller one plus, at most,
   perturbations of the newly added segments (monotone chains, point 4).
   Growing k never swaps in edits of different difficulty.
2. **Identical candidate pool at every scale.** Only documents with ≥ k_max
   units whose k_max prefix passes the 500-token filter enter the pool, and
   every scale reuses that same pool — no "short blocks are plentiful, long
   blocks are rare" selection confound. (Trade-off, on purpose: the pool is
   restricted to longer documents; pool size and the share of long-enough docs
   are logged per language.)
3. **A failed perturbation never falls back to the clean sentence.** If QC
   rejects the perturbation of any unit, the whole (family, condition) cell is
   dropped at *all* scales — and item assembly asserts that every perturbed
   position differs textually from its clean counterpart.
4. **One perturbation per segment; controlled positions and types.** Each
   perturbed segment carries exactly one edit; all perturbed segments of an
   item share one condition (severity × error-family is a between-item
   factor). Positions follow a deterministic per-family permutation π (seeded
   by family identity, shared across conditions): perturbation sets are the
   nested chain P₁ = {π₁} ⊂ P₂ ⊂ … ⊂ P_k_max, realized at scale k as
   P_j ∩ [0,k). This yields exactly one item per m ∈ {0..k} per
   (family, condition, k) — full, balanced m-coverage at every scale — with
   position balance across families.

k = 1 is included as the trivial anchor: S ≡ s₁ by construction (identical
scoring input), so k=1 feeds the dilution table (the *undiluted* severity
effect size) and the clean curve, but is excluded from aggregation fits.

## Protocol

1. **Data** — WMT24++ (`google/wmt24pp`) human post-edits = the error-free
   baseline, en→{de, zh, es, fr} by default. Units are the aligned dataset
   segments (optionally bertalign-resegmented sentences, `data.resegment:
   bertalign`). Families are non-overlapping k_max windows of contiguous
   same-document units; k_list = {1, 2, 3, 4, 6}. A FLORES-200 source is
   available as a robustness set (`data.source: flores`).
2. **Perturbations** — per family × condition (minor/major ×
   accuracy/fluency), every unit gets one perturbed variant, generated with the
   published error-injection prompts of Zhang et al. (App. A.4) through any
   OpenAI-compatible endpoint (`perturb.backend: llm`), or with deterministic
   length-preserving rule edits (`rule`) for smoke tests. QC enforces minimal,
   **length-preserving** edits so quality is dissociated from length.
3. **Scoring** — s_i = score of sentence i **alone** (single-sentence input);
   S = score of the concatenated prefix. Metrics probed independently:
   CometKiwi (`Unbabel/wmt22-cometkiwi-da`), COMET (`wmt22-comet-da`, ref =
   the unperturbed post-edit prefix), MetricX-24-QE (error scale — fitted on
   the penalty scale, where mean/sum/max-penalty are the natural candidates).
   All scores are disk-cached; reruns are free.
4. **Fit** — candidates: mean, min, max, median (+ sum on the penalty scale),
   power mean M_p (free p), soft-min/log-sum-exp (free temperature). Every
   candidate gets the same affine calibration S ≈ a + b·g(s) so AIC/BIC compare
   *shapes*. Model selection by GroupKFold-by-document CV R²; bootstrap CIs for
   p resampled by document. Note: within a fixed k, `sum` is affine-equivalent
   to `mean`; they separate only in the pooled fit across k
   (`candidate_fits_pooled.csv`).
5. **Controls** — position test (is f symmetric? residual ~ position of the
   m=1 error); severity/family interaction (p per condition); stability of p
   across k; clean curve (error-free prefix score vs k — the concatenation
   bias of Zhang et al. measured on our paired families); frozen-f residual by
   k (the bridge to the length experiment); optional MixedLM on residuals.

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
| `results/items.csv` | tidy dataframe: metric, lang, k, m, condition, positions, chain step/plan, family/doc ids, s_1..s_k, S, token counts |
| `results/analysis/candidate_fits.csv` | all candidates × (metric, k≥2): R², CV-R², AIC/BIC, p̂ + CI |
| `results/analysis/p_by_k.csv` | headline power-mean exponent per (metric, k) |
| `results/analysis/p_by_condition.csv` | p̂ split by severity and error family |
| `results/analysis/position_test.csv` | symmetry control (slope, p-value, ΔR²) |
| `results/analysis/residual_by_k.csv` | S − f̂ with f frozen at the smallest k ≥ 2 |
| `results/analysis/dilution.csv` | Δ(m, k, severity, family); k=1 = undiluted severity effect |
| `results/analysis/clean_curve.csv` | error-free prefix score vs k (concatenation bias) |
| `results/figures/` | obs-vs-pred, p-by-k, candidate CV comparison, dilution, clean curve, frozen residual |
| `results/SUMMARY.md` | one verdict sentence per metric |

## Design notes

- **Determinism**: one global seed drives family sampling; chain permutations
  are seeded by family *identity* (not index), so adding/removing families
  never reshuffles the others; LLM outputs and metric scores are content-hash
  cached, so a rerun reproduces the same items without new API/GPU work.
- **No candidate is hard-coded as ground truth**: all fits are reported, the
  "winner" is picked by grouped CV only.
- The human-annotator arm of Exp 0 (small sample: rate paragraphs with 1
  absurd sentence out of k, compare p_human vs p_metric) reuses `items.csv`
  directly — export the m ∈ {0, 1} subset for annotation.
