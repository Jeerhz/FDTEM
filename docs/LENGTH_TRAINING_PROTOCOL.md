# Protocol — impact of training-data length on learned MT metrics (DA + QE)

> **Correction (2026-08-18).** The runs this protocol describes trained on one
> language pair's slice of their mix, not on the mix, because COMET reads a
> single file per epoch and the dataloader was never rebuilt. Arm rankings from
> those runs track training-set size, not composition, and must not be reported.
> The defect, the fix and the corrected protocol are in
> [`experiments/length_training/README.md`](../experiments/length_training/README.md).
> The WMT sweep described there (36 arms) supersedes the 28-arm Bio-MQM grid below.


Status: **running** (DA arms since 2026-08-14, QE arms submitted 2026-08-17).
Companion docs: `docs/RETRAIN_AND_BLOCK_XSIM.md` (original DA design),
cluster runbook for operations. This document is the full experimental
protocol covering **both** base metrics and all evaluation lenses.

---

## 1. Research questions

- **RQ1 (competence):** does continuing training on multi-sentence inputs
  improve a learned metric's agreement with human judgement *on long inputs*?
- **RQ2 (forgetting / restoration):** what does long-text training cost at
  the sentence level, and what is the smallest fraction of sentence-level
  data that keeps both? This is the practically decisive number.
- **RQ3 (locus):** does the length behaviour live in the encoder weights or
  in the regression head? (frozen-encoder vs unfrozen arms)
- **RQ4 (metric family):** do reference-based (COMET-DA) and reference-free
  (CometKiwi QE) metrics respond differently? QE has no reference to anchor
  on, so length-induced drift could be worse — or the src-only conditioning
  could make it more robust. No prior evidence either way.
- **RQ5 (genuine document competence):** does long-text training buy
  detection of *discourse-level* errors, or only a rescaled score?
  Segment correlation cannot answer this (discourse negatives preserve
  sentence-level fluency); MetaDocEval can.

## 2. Factors and arms

| factor | levels |
|---|---|
| base metric | `Unbabel/wmt22-comet-da` (ref-based) · `Unbabel/wmt22-cometkiwi-da` (ref-free) |
| sentence fraction f | 0, 10, 20, 40, 60, 80, 100 % (share of k=1 windows in the training set) |
| encoder regime | unfrozen (default continue schedule, `nr_frozen_epochs=0.3`) · frozen (`nr_frozen_epochs=1000`, head-only) |

**28 training arms** (2 × 7 × 2) + the two public baselines evaluated as-is.
Checkpoint layout: `~/scratch/checkpoints/retrain/[kiwi-]mix-fracNNN[-frozen]/`.

Both metric families train on **identical CSVs** (QE simply never reads the
`ref` column), with the same conservative continue-training schedule
(encoder LR 5e-7, head LR 1e-5, effective batch 32, early stopping on
validation Kendall τ with patience 20, `max_epochs` unbounded), so the only
moving part across arms is the **composition** of the training mix — and,
across families, the base metric.

## 3. Training data

Source: **Bio-MQM** (Zouhar et al. 2024) — sentence-level MQM annotations
with `(system, doc_id, seg_id)`, biomedical domain, published aggregation at
[AdleBenSalem/bio-mqm-paragraphs](https://huggingface.co/datasets/AdleBenSalem/bio-mqm-paragraphs).

The paragraph-level examples are built by **aggregating in-domain
sentence-level data**: sliding windows of k ∈ {1,2,3,4,6} *consecutive
segments of the same document*, concatenated on src/mt(/ref), scored with
the **mean per-segment MQM penalty** (z-normalised + sigmoid per language
pair, fit on train — monotone, so rank correlations are unaffected).
k=6 windows are abstract-sized (~paragraph "notes"); ≤480 XLM-R tokens.
Train windows: Bio-MQM *dev* documents. Validation windows: Bio-MQM *test*
documents — **document-disjoint by construction**.

The seven mixes `frac000…frac100` hold the **total number of windows
constant** (7,025; identical per-LP totals) and vary only the k=1 share;
long-k mass is split evenly over k ∈ {2,3,4,6}. Each (lp,k) pool is
shuffled once (seed 42) and mixes take prefixes, so mixes differ only by
prefix length. `mixes/manifest.json` records seed + md5 per CSV.
Language pairs: en↔{de,es,fr,ru}, 8 directions.

## 4. Evaluation lenses

| lens | data | status w.r.t. selection | question |
|---|---|---|---|
| **Dev per-k correlation** | val CSVs (Bio-MQM test docs, non-overlapping windows, n=46,882) | *selection-coupled* — same pool early stopping monitors | development signal; τ per k ∈ {1,2,3,4,6} |
| **Held-out test per-k correlation** | `~/scratch/paragraph_mqm_heldout/{zh-en,en-zh}_val.csv` (17,372 windows) | **fully held out**: language pairs never trained on, documents never seen, same in-domain construction | the reportable per-k numbers |
| **MetaDocEval** | external contrastive test set (Dahan, Bawden & Yvon 2026) | fully held out, different corpus | RQ5: discourse-error detection accuracy vs context window w ∈ {1,3,6,9}, per phenomenon; `sentence_splitting` is quality-preserving → its "accuracy" is a false-positive rate and is tracked as such |
| **Block-xSIM++** | FLORES+ doc-level ([AdleBenSalem/flores-plus-doc-level](https://huggingface.co/datasets/AdleBenSalem/flores-plus-doc-level)) | fully held out | encoder-level: is a single buried error still separable after fine-tuning? (DA arms; run on the most interesting f) |

Notes on the held-out test: Kendall/Spearman are invariant to the per-LP
monotone score normalisation, so unseen-LP evaluation is legitimate despite
normalisation params being LP-specific. br-en was intended as a third
held-out pair but the local Bio-MQM clone carries no br-en annotation files.

## 5. Analysis plan (pre-specified)

Per arm, per lens:

1. **Δτ(k=1)** vs the arm's own base metric — sentence-level forgetting.
2. **Restoration point f\*** — smallest f whose k=1 τ is within the
   bootstrap CI of the base metric. Report per family (DA, QE).
3. **Length profile** — τ as a function of k; slope of τ vs log₂k as a
   scalar length-sensitivity summary; compare frac000 vs frac100 envelopes.
4. **Locus decomposition (RQ3)** — frozen-arm effect ÷ unfrozen-arm effect
   per f; if frozen arms barely move, the behaviour lives in the encoder.
5. **Family contrast (RQ4)** — same quantities side-by-side DA vs QE; the
   families differ in base τ, so compare *relative* changes, not absolutes.
6. **MetaDocEval (RQ5)** — per arm: does lexical-perturbation accuracy rise
   with w? (In the paper, no reference-based metric does.) Any arm climbing
   above chance is a headline finding — re-verify before claiming (cache off,
   fresh run). Track the splitting false-positive rate across w; flag any
   arm that worsens it.
7. **Uncertainty** — bootstrap over *documents* (not windows: windows of one
   doc share errors), 1,000 resamples, 95% percentile CIs, everywhere.
   Report per-k n alongside every number; k=6 cells are smallest — never
   compare across k without CIs.

Decision table:

| observed pattern | conclusion |
|---|---|
| frac000 k=1 τ drops, f* small (≤20%) | length behaviour is data-composition; cheap to fix |
| frac000 k=1 τ drops, f* large (≥60%) | strong tension; document-level metrics need dedicated mixes |
| frozen arms flat, unfrozen move | effect lives in encoder weights (ties to matched-core/layer results) |
| MetaDocEval flat for all arms | long-text training buys score rescaling, not discourse competence |
| QE mirrors DA | conclusions generalise across metric families |
| QE diverges from DA | reference anchoring interacts with length — analyse per phenomenon |

## 6. Operations

- Kiwi arms: `MODEL=qe VARIANT=mix MIX=fracNNN FROZEN={0,1} sbatch
  scripts/slurm_retrain_comet.sh` (unfrozen arms with
  `--exclude=gpu001,gpu004,gpu011,gpu014` — ≤16 GB GPUs OOM at unfreeze).
  Config: `configs/models/comet_kiwi_paragraph_continue.yaml`
  (`layer_transformation: softmax` — the released kiwi hparams'
  `sparsemax_patch` resolves to softmax under comet ≥ 2.2.4, issue #244).
- Evaluations are cached per model — rerun cheaply as arms finish:
  `SWEEP=1 sbatch scripts/slurm_metadoceval.sh` (auto-discovers both
  `mix-*` and `kiwi-mix-*` arms);
  `scripts/eval_length_correlation.py --data_dir ~/scratch/paragraph_mqm`
  (dev lens) and `--data_dir ~/scratch/paragraph_mqm_heldout` (test lens).
- Wave discipline: frac000/frac100 × frozen/unfrozen first per family;
  remaining 10 arms only after a clean first validation cycle.
- **License**: any published kiwi fine-tune must be CC-BY-NC-SA-4.0
  (base-model license); DA fine-tunes are Apache-2.0.

## 7. Threats to validity (acknowledged)

- Dev lens is selection-coupled (early stopping monitors it) — reportable
  numbers come from the held-out lenses.
- Single domain (biomedical) and 4 (+1 held-out) language pairs; claims are
  about *this* regime until replicated on a second domain (WMT news MQM has
  doc structure and is the natural replication).
- Window scores are *means* of segment scores — aggregation itself is a
  modelling choice; MetaDocEval's contrastive design is the control that
  does not depend on it.
- One seed per arm (compute-bound). The constant-total design and 28-arm
  grid trade seed replication for composition resolution; the bootstrap CIs
  quantify sampling noise but not seed noise.
