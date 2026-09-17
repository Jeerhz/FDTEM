# Part 1 results — provenance

All JSON files are `RunResult` (`models.py`) unless stated; plots in `plots/` carry the
JSON's stem as prefix. Encoder keys are `enc_tag` labels (`comet:wmt22-comet-da`,
`xlmr:xlm-roberta-large`, `labse`, `e5:multilingual-e5-base`, `comet:<run>-<ckpt>`).

| file | date | produced by | backend | models | cited by |
|---|---|---|---|---|---|
| `duel.json` | 2026-07-28 | `evaluate_duel.py` (then `run_duel.py`), D ∈ {6, 5, 4}, de/es/fr/ru, k = 2..5 | heuristic | COMET-DA, XLM-R, LaBSE, E5, Bio-MQM COMET | `figures/make_figures.py` (iso_*), `report/figures/make_answer_figures.py` |
| `comet_score.json` | 2026-09-01 | `evaluate_comet_score.py` (then `run_comet_align.py`), k = 1..5 | spacy | CometKiwi (score + cosine), COMET-DA (cosine), three part-2 arms | `figures/make_figures.py` (iso_comet_score, align_*), `report/make_status_page.py` |
| `encoder_cosine_2026-07-16_heuristic.json` | 2026-07-16 | `evaluate_encoders.py` (then `run_xsim.py`), pre pool-ablation schema | heuristic | COMET-DA, Bio-MQM COMET, XLM-R, LaBSE, E5 | `report/main.tex` table `tab:blockxsim` |
| `encoder_cosine_arm_frac000.json` | 2026-08-23 | same, baseline zoo + the wave-1 `mix-frac000` arm | heuristic | + `comet:mw5cryt7-…` | `report/figures/make_answer_figures.py` (q3_detection) |
| `encoder_cosine_arm_frac100.json` | 2026-08-23 | same, the wave-1 `mix-frac100` arm alone | heuristic | `comet:4cnnyi3x-…` | same |
| `encoder_cosine_spacy_smoke.json` | 2026-08-30 | `evaluate_encoders.py`, CPU, `--max_blocks 40`, FLORES-200 raw, de/es/fr/ru/zh | spacy | `e5:multilingual-e5-small` | the only output of the current pool-ablation code path; `report/partie1_longueur.tex` (control: error 0 on a pool of correct blocks only) |
| `matched_core.json` | 2026-08-21 | matched-core probe, **dropped 2026-09-16** (no producer; plain dict) | — | 5 encoders, L ∈ {60, 120, 240, 480}, 4 fillers, 3 positions | `report/figures/make_report_figures.py`, `make_answer_figures.py` |
| `matched_core_layers.json` | 2026-07-28 | per-layer variant of the same probe, **dropped** (plain dict) | — | XLM-R large, 25 layers, de/fr | `report/main.tex` (layer-wise claim) |

The July/August files were produced with the heuristic perturber and cannot be
regenerated identically with the spaCy backend. The spacy smoke run is a subsampled
validation of the code, not a cluster result.
