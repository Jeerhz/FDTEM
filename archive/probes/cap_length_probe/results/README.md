# results/

Populated by `python run.py --config config.yaml` (real frozen encoders, GPU):

- `results.csv`, `predictions.csv.gz`
- `analysis/*.csv` (summary, slopes, gap_by_L, per_language, regime2, geometry)
- `figures/*.png`
- `SUMMARY.md`

The `feature_cache/` subdir (cached pooled features, keyed by content hash) is
git-ignored. The local **logic** smoke test writes to `../results_smoke/` instead.
