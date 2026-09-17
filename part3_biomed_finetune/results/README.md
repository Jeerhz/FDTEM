# Part 3 results

| file | produced | by | content |
|---|---|---|---|
| `correlation_bio_vs_base_2026-08-14.json` | 2026-08-14 | the old length-training evaluator (`eval_correlation.py`, now `part2_length_training/eval_validation.py`) | correlations of `wmt22` (Unbabel/wmt22-comet-da) and `bio` (the Bio-MQM fine-tuned checkpoint) with human scores on the archived Bio-MQM paragraph windows of `~/scratch/paragraph_mqm`, per language pair and window size k = 1..6 |

`correlation_bio_vs_base_2026-08-14.json` is the per-language-pair source of
`report/main.tex` table `tab:bio`. Its layout (`models -> label -> lp -> k -> cell`,
plus `_mean_by_k`) is the part-2 `CorrelationResults` shape, not `BioEvalResults`.

`evaluate.py` supersedes it: it scores the sentence-level `<lp>_val.csv` files
written by `load_bio_mqm.py` and writes `correlation.json` (`BioEvalResults`,
labels `base` / `bio`, document bootstrap of delta Kendall) plus
`plots/kendall_by_lp.png`. Prediction caches live under `cache/` (git-ignored).
