# Part 1 — aligning FLORES+ blocks: does one buried error still get noticed?

Take FLORES+ articles, concatenate `k` consecutive sentences into a block, and ask a
model to pick the correct translation of that block among hard negatives that differ
by **one** perturbed sentence (xSIM++ categories: causality, entity, number). As `k`
grows the error is diluted; the question is how fast, and whether the metric's score
resists the dilution better than the cosine of the encoder it is built on.

Why retrieval: a learned metric is a pull-together / push-apart machine. Its encoder
must pull a translation towards its source and push a wrong one away, and a single
edit inside a long block is the smallest push-apart signal we can inject while keeping
everything else identical. Encoders trained purely for alignment (LaBSE, E5) are the
controls: they were never taught what an error is.

## Pipeline

```
load_flores.py     FLORES+ (gated) or a local FLORES-200 dump        -> data/flores_<source>_<split>.json
build_blocks.py    k consecutive sentences of one article, stride k   -> data/blocks_<source>_<split>_k<k>.json
perturb.py         gold + single-edit negatives per block             -> data/pools_<backend>_<lang>_k<k>.json
evaluate_encoders.py     cosine retrieval on four nested pools        -> results/encoder_cosine.json
evaluate_duel.py         gold vs D own negatives, closed form         -> results/duel.json
evaluate_comet_score.py  COMET score vs encoder cosine, same pools    -> results/comet_score.json
figures/make_figures.py  the nine report figures                      -> figures/
```

`data/` is not committed: `sbatch part1_block_alignment/slurm/build_pools.sh` rebuilds
it on CPU (`python -m part1_block_alignment.perturb --dry_run` prints coverage and
example negatives without writing anything). The models are in `models.py`:
`FloresCorpus` (common), `Block`, `Candidate`, `CandidatePool` (the dataset an
evaluation loads), and the `RunResult` written by every evaluation.

## Perturbation backends

`--backend spacy` (default) uses real NER, morphology and the dependency parse; it is
the only backend that yields entity negatives for uncased scripts (zh/ja) and the one
that gets "ne … pas", German V2 and Romance clitics right (`perturb_spacy.py`; needs
`python -m spacy download <lang>_core_*_sm`). `--backend heuristic` is the
self-contained fallback (casing for entities, regexes for numbers and negation). The
two generate different negatives: runs are not comparable across backends. Every
negative is deterministic: the RNG is seeded from (seed, lang, category, sentence,
variant index).

## What is in the candidate pool

`evaluate_encoders.py` scores four nested pools off one similarity matrix
(`pool_ablation` in the JSON):

| pool | candidates for query *b* | |
|---|---|---|
| `true_only` | every true block | classic xsim |
| `true+perturbed` | every true block + every negative | classic xsim++ |
| `gold+all_perturbed` | *b*'s gold + every negative | classic distractors dropped |
| `gold+own_perturbed` | *b*'s gold + *b*'s own negatives | **hard negatives only — headline** |

Only the last pool leaves the model nothing to separate but the injected edit; the two
classic pools also reward telling articles apart. Both hard-negative pools still grow
with `k`, which is what the duel pins.

## Controlling for everything but length

| confound | control |
|---|---|
| the candidate pool grows with `k` | `evaluate_duel.py` pins it to gold + **D** single-error negatives at every `k` and averages over all C(m, D) subsets in closed form |
| blocks overlap | blocks are non-overlapping (stride = `k`) |
| longer blocks come from longer articles | per-`k` block counts are reported; long `k` is rare and must not be compared without CIs |

## Aligning with the metric instead of with the encoder

`evaluate_comet_score.py` keeps the blocks and negatives and swaps only the decision
rule: `encoder-cos:<ckpt>` (argmax cosine) versus `comet-score:<ckpt>` (argmax
COMET(src = source block, mt = candidate)). Run both from the same checkpoint and the
contrast is "the regression head versus the representation it sits on". `comet-score:`
refuses a reference-based checkpoint: at alignment time the reference *is* the
translation being retrieved. Two fixed-size protocols: a **duel** (gold vs one of its
own negatives, chance 0.5, broken down by category and by position of the perturbed
sentence) and a **shortlist** (gold vs 2 of its own negatives, chance 1/3; a block that
cannot supply two is skipped, never padded, and the coverage is reported).

## Run

```bash
sbatch part1_block_alignment/slurm/build_pools.sh            # once: FLORES -> blocks -> pools
sbatch part1_block_alignment/slurm/evaluate_encoders.sh      # dilution curve, four pools
sbatch part1_block_alignment/slurm/evaluate_duel.sh          # fixed-pool control
sbatch part1_block_alignment/slurm/evaluate_comet_score.sh   # score vs cosine
python -m part1_block_alignment.figures.make_figures         # figures/iso_*, align_*
```

Every script is `python -m part1_block_alignment.<script> --help`. Encoders are named
`comet:<hub-id-or-ckpt>`, `hf-mean:<hf-id>`, `labse`, `e5`; pass a checkpoint trained
in part 2 to test a length-trained encoder (`RETRAIN_CKPT`, `QE_CKPT`, `DA_CKPT`).

## Files

| file | role |
|---|---|
| `models.py` | `Block`, `Candidate`, `CandidatePool`, the metric cells and `RunResult` |
| `load_flores.py` · `build_blocks.py` · `perturb.py` · `perturb_spacy.py` | the data pipeline |
| `evaluate_encoders.py` · `evaluate_duel.py` · `evaluate_comet_score.py` | the three evaluations |
| `slurm/` | one job per script above, plus `build_pools.sh` |
| `results/` | committed JSON + plots, see `results/README.md` for provenance |
| `figures/make_figures.py` | the report figures (French labels) |

The matched-core probe (same core sentence, filler of controlled length) was dropped
on 2026-09-16; its results `results/matched_core*.json` and plots are kept because
`report/main.tex` cites them, but they have no producer any more.
