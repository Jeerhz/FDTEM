# 2 — Isolating length

Take FLORES+ articles, concatenate `k` consecutive sentences into a block, and
ask an encoder to retrieve the correct translation of that block among hard
negatives that differ by **one** perturbed sentence (xSIM++ categories:
causality, entity, number).

As `k` grows the error is diluted — but so does everything else, which is why
the design controls the confounds instead of just growing `k`.

## What is in the candidate pool

Every run of `run_xsim.py` scores four nested pools off one similarity matrix
(`pool_ablation` in the JSON, `plots/pool_ablation.png`), so dropping the classic
xsim distractors — the *other* blocks' true targets — costs no extra compute:

| pool | candidates for query *b* | |
|---|---|---|
| `true_only` | every true block | classic xsim |
| `true+perturbed` | every true block + every negative | classic xsim++ |
| `gold+all_perturbed` | *b*'s gold + every negative | classic distractors dropped |
| `gold+own_perturbed` | *b*'s gold + *b*'s own negatives | **hard negatives only — headline** |

`gold+own_perturbed` is what is reported and plotted
(`plots/hard_negative_error.png`): it leaves the encoder nothing to separate but
the injected edit, whereas the two classic pools also reward simply telling
articles apart. The classic rows stay in the JSON as a reference, not as the
result. Both hard-negative pools still grow with `k`, which is what
`run_duel.py` pins.

## Perturbation backends

`--perturb_backend spacy` (default) uses real NER, morphology and the dependency
parse — the only backend that produces
entity negatives for uncased scripts (zh/ja), and the one that gets "ne … pas",
German V2 word order and Romance clitics right. `--perturb_backend heuristic` is
the older self-contained fallback (casing for entities, regex for numbers and
negation); the two generate different negatives, so runs are not comparable
across backends. Antonyms stay on the curated
lexicons; WordNet is English-only by default because Open Multilingual WordNet
has no German or Russian and is sense-ambiguous elsewhere. See
`nlp_perturb.py` for the evidence and `pip install spacy nltk` +
`python -m spacy download <lang>_core_*_sm` to enable it.

## Controlling for everything but length

| confound | control |
|---|---|
| the candidate pool grows with `k` | `run_duel.py` pins the pool to gold + **D** single-error negatives at every `k`, and averages over all `C(m, D)` subsets in closed form |
| blocks overlap | blocks are non-overlapping (stride = `k`) |
| longer blocks come from longer articles | per-`k` block counts are reported; long `k` is rare and must not be compared without CIs |
| "is it length or is it content?" | `run_matched_core*.py` keeps the **same** core sentence and only changes the filler around it (inert / neutral / distractor / natural) and its position |

## Aligning with the metric instead of with the encoder

Everything above retrieves by cosine similarity between encoder embeddings —
which measures the encoder, not the metric anybody reports. `run_comet_align.py`
keeps the articles, the blocks and the hard negatives exactly as they are and
swaps only the decision rule:

| rule | pick |
|---|---|
| `encoder-cos:<ckpt>` | argmax cos(embed(source block), embed(candidate)) |
| `comet-score:<ckpt>` | argmax COMET(src = source block, mt = candidate) |

Run both from the **same** checkpoint and the contrast is "the regression head
versus the representation it sits on"; run them from a base metric and from a
length-trained arm and it becomes "what did long-text training move".

`comet-score:` refuses a reference-based checkpoint: at alignment time the
reference *is* the translation being retrieved, so passing it hands the model
the answer. COMET-DA and its arms take part through `encoder-cos:`.

The candidate set is the reference block plus **its own** perturbed variants.
The other blocks' true targets — the classic xsim distractors — are excluded on
purpose: separating articles is an easy, different skill, and leaving it in lets
a model look sensitive to the injected edit when it is only recognising the
topic. (`blocks.evaluate_blocks` reports the same ablation for the cosine rule,
as `pool_ablation`.)

Two fixed-size protocols, because the candidate count must never be a by-product
of k: a **duel** (gold vs one of its own single-error negatives, chance 0.5,
broken down by perturbation category and by the position of the perturbed
sentence) and a **shortlist** (gold vs 4 of its own negatives, chance 0.2; a
block that cannot supply four is skipped rather than padded, and the skipped
count is reported). Score scales differ between a cosine and a regression head,
so margins are not comparable across rules — `frac_negatives_beaten` and the
gold's rank are, and both are reported.

## Run

```bash
sbatch experiments/length_isolation/slurm/xsim.sh          # dilution curve over k
sbatch experiments/length_isolation/slurm/duel.sh          # fixed-pool control
sbatch experiments/length_isolation/slurm/matched_core.sh  # same core, growing filler
sbatch experiments/length_isolation/slurm/comet_align.sh   # metric score vs encoder cosine
```

`comet_align.sh` prints the number of COMET forward passes it is about to cost
(a `--dry_run` pass) before spending them; `NEGATIVES` and `SHORTLIST` are the
two knobs that set it.

Encoders are named uniformly: `comet:<hub-id-or-ckpt>`, `hf-mean:<hf-id>`,
`labse`, `e5`. Pass a checkpoint from experiment 3 to test a finetuned encoder.

## Files

| file | role |
|---|---|
| `blocks.py` | blocks, perturbations, candidate pools, evaluation |
| `nlp_perturb.py` | optional spaCy/WordNet perturbation backend |
| `run_xsim.py` | the dilution curve |
| `run_comet_align.py` | same blocks, retrieval by COMET score instead of cosine |
| `run_duel.py` | fixed-pool duel — the candidate-count control |
| `matched_core.py`, `run_matched_core*.py` | matched-core probes, incl. per-layer |
