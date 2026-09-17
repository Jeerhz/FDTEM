# Part 2 results — provenance

| file | wave | produced by | read by |
|---|---|---|---|
| `correlation_val.json` | wave 2 (2026-09-01) | `eval_validation.py --lens val` | `analyze.py --lens val`, `figures/make_figures.py` |
| `correlation_heldout.json` | wave 2 | `eval_validation.py --lens heldout` | `analyze.py`, `figures/make_figures.py`, `report/figures/make_report_figures.py`, `report/figures/make_answer_figures.py`, `report/make_status_page.py` |
| `length_profile_val.json` · `length_profile_heldout.json` | wave 2 | `eval_length_profile.py` | `figures/make_figures.py` |
| `metadoceval.json` | wave 2 | `eval_metadoceval.py` | `figures/make_figures.py`, `report/figures/make_answer_figures.py` |
| `token_length_by_k.json` | wave 2 | no producer in the repo (token histograms per k and lens) | `figures/make_figures.py` |
| `wave1_correlation_heldout.json` | wave 1 (2026-08-18, 30 arms, 6 epochs) | the then `eval_correlation.py` | `report/figures/make_report_figures.py` (baseline_tau_vs_k) |
| `wave1_metadoceval.json` | wave 0 (2026-08-17, Bio-MQM-era baselines `wmt22` / `kiwi`) | the then `eval_metadoceval.py` | `report/figures/make_report_figures.py` (metadoceval_base) |
| `wave1_wandb_curve_diagnosis.json` | waves 0 and 1 | no producer in the repo (W&B API dump of the invalid sweeps' validation curves) | `report/figures/make_report_figures.py` (wandb_bell) |

Waves: **0** = the Bio-MQM paragraph pipeline (2026-08-14/17, declared uninterpretable),
**1** = the 36-arm WMT sweep at 6 epochs (2026-08-18, one train file per epoch fixed),
**2** = the four-arm 60-epoch wave (2026-09-01, `frac100`, `frac000agg`, `frac000nat`,
`frac000` for both families). Labels are `ArmLabel` strings (`da-frac000`,
`qe-frac000agg-frozen`, `da-base`). Models: `CorrelationResults`, `LengthProfileResults`,
`MetaDocEvalResults` in `models.py`. Prediction caches live under `cache/` (ignored).
