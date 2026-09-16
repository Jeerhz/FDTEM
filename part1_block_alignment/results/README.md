# block xSIM++ — hard negatives only, spaCy perturbations

Run of `experiments/length_isolation/run_xsim.py` with the two protocol changes:
the candidate pool is **gold + single-edit hard negatives** (the other blocks'
true targets — the classic xsim distractors — are no longer candidates), and the
negatives come from the **spaCy backend** (real NER, morphology, parse-anchored
negation) rather than the casing/regex heuristics.

## Configuration

| | |
|---|---|
| encoder | `e5:intfloat/multilingual-e5-small`, CPU |
| corpus | FLORES-200 raw, `dev` + `devtest` (2009 aligned rows) |
| languages | de, es, fr, ru, **zh** |
| k | 2, 3, 4, 5 (non-overlapping blocks, stride = k) |
| negatives | 2 per (block, sentence position, category) |
| blocks | `--max_blocks 40` per split → 80 blocks per (lang, k) |
| backend | `--perturb_backend spacy`, seed 42 |

This is a **small-encoder, subsampled** run — it is not the cluster encoder zoo
(COMET / bio-COMET / XLM-R / LaBSE / E5-base). Reproduce at full scale with
`sbatch experiments/length_isolation/slurm/xsim.sh` (BACKEND defaults to spacy).
With 80 blocks per cell the binomial CI on an error near 0.8 is roughly ±0.09.

## Result — mean over the five languages

| k | negs/block | chance | **gold+own** | gold+all | xsim | xsim++ |
|--:|--:|--:|--:|--:|--:|--:|
| 2 | 8.4 | 0.886 | **0.475** | 0.475 | 0.000 | 0.475 |
| 3 | 12.2 | 0.919 | **0.657** | 0.657 | 0.000 | 0.657 |
| 4 | 16.0 | 0.938 | **0.800** | 0.800 | 0.000 | 0.800 |
| 5 | 16.2 | 0.936 | **0.875** | 0.875 | 0.000 | 0.875 |

Per-language `gold+own` error:

| | k=2 | k=3 | k=4 | k=5 |
|---|--:|--:|--:|--:|
| de | 0.412 | 0.600 | 0.787 | 0.875 |
| es | 0.425 | 0.713 | 0.838 | 0.812 |
| fr | 0.487 | 0.713 | 0.812 | 0.900 |
| ru | 0.487 | 0.600 | 0.738 | 0.912 |
| zh | 0.562 | 0.662 | 0.825 | 0.875 |

### 1. The classic xsim distractors contribute nothing

In **all 20 (language, k) cells**, `gold+own_perturbed`, `gold+all_perturbed` and
`true+perturbed` are equal to machine precision, and the true-only pool
(`xsim`) has error **exactly 0.000**. Picking the right *article* is trivial for
a modern multilingual encoder at these block lengths; every error is the encoder
preferring its own block's perturbed copy. Removing the other blocks' true
targets is therefore free — and keeping them was measuring nothing. See
`plots/pool_ablation.png`: three curves lie on top of each other and the classic
xsim curve is flat on zero.

### 2. Dilution, read against chance

The own pool grows with k (8.4 → 16.2 negatives per block), so chance error
rises too. Normalised skill, `(chance − err) / chance`:

| k | 2 | 3 | 4 | 5 |
|---|--:|--:|--:|--:|
| skill | 0.464 | 0.285 | 0.147 | **0.065** |

The encoder keeps less than a tenth of its headroom above chance once the edit
sits inside a five-sentence block. `margin_vs_best_own_perturbation` turns
**negative** from k = 3 on: on average the gold block is *further* from the query
than its best perturbed rival. Note the pairwise `detection_rate` stays high
(0.94 → 0.80) — one-vs-one hides what one-vs-many exposes.

### 3. Causality is the failure mode; entities and numbers survive

Per-category own pools (only that category's negatives, so each has its own
chance floor):

| k | causality err / chance | entity err / chance | number err / chance |
|--:|--:|--:|--:|
| 2 | 0.360 / 0.776 | 0.092 / 0.760 | 0.132 / 0.633 |
| 3 | 0.550 / 0.837 | 0.131 / 0.811 | 0.176 / 0.733 |
| 4 | 0.698 / 0.871 | 0.225 / 0.850 | 0.200 / 0.783 |
| 5 | 0.792 / 0.893 | 0.303 / 0.832 | 0.290 / 0.742 |

Negation and antonym flips are near chance by k = 5 while a swapped entity or
digit is still caught two times out of three. Surface-token changes survive
dilution; polarity changes do not.

### 4. zh only exists because of the spaCy backend

The casing heuristic yields **zero** entity negatives for uncased scripts. spaCy
NER harvested 1180 entities across 10 labels from the Chinese corpus, so zh is a
full participant here for the first time (2106/de, 1613/es, 1495/fr, 1353/ru).

## Caveat carried over

Both hard-negative pools still *grow* with k, which is why the chance line is
plotted. For a comparison across k at a genuinely fixed candidate count, use
`run_duel.py` — the own pool is its D = m (all-negatives) case.

## Files

| | |
|---|---|
| `block_xsim.json` | every metric, per encoder × language × k |
| `plots/hard_negative_error.png` | headline: own/all-negative error vs k, with chance |
| `plots/pool_ablation.png` | all four pools — the ablation of the classic distractors |
| `plots/error_by_category.png` | own-pool error per perturbation category |
| `plots/detection_vs_length.png`, `plots/detection_by_position.png` | pairwise detection |
