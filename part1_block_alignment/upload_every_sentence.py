"""Publish the every-sentence dataset (perturb_every_sentence.py) on the Hugging Face Hub.

    python -m part1_block_alignment.upload_every_sentence --repo_id AdleBenSalem/flores-plusplus-blocks

Uploads the files the evaluation read, data/every_sentence_<backend>_<lang>.jsonl, as
data/en-<lang>.jsonl (one config per language pair, split `test`) with a dataset card.
FLORES+ asks not to re-host its text where web crawlers can reach it, so the repository
is gated with the FLORES+ conditions (automatic approval) before it is made public;
--private keeps it private instead. Needs `hf auth login` once.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from common.auth import hf_token
from part1_block_alignment.models import CATEGORIES, PerturbedBlock
from part1_block_alignment.perturb_every_sentence import dataset_path, load_rows

CARD = """\
---
license: cc-by-sa-4.0
language:
{languages}
task_categories:
- translation
pretty_name: FLORES++ blocks
size_categories:
- 1K<n<10K
source_datasets:
- openlanguagedata/flores_plus
tags:
- machine-translation-evaluation
- flores
- xsim++
- perturbations
- long-context
configs:
{configs}
extra_gated_heading: "Protecting the integrity of FLORES+ for evaluation"
extra_gated_prompt: "This dataset contains FLORES+ text and inherits its conditions."
extra_gated_fields:
  I agree not to re-host this dataset or FLORES+ in places where it could be picked up by web crawlers: checkbox
  If I evaluate using this dataset, I will ensure that its contents are not in the training data: checkbox
---

# FLORES++ blocks

Blocks of k = 1..{k_max} consecutive segments of one FLORES+ article (dev + devtest),
English source, with the correct translation (the reference) and up to {n_max}
**distractors**: copies of the reference in which **every segment carries one
perturbation** from the three xSIM++ categories (causality, entity, number;
Chen et al., 2023). The error share is one per segment at every k, so if a model
finds the reference less often on longer blocks, the drop cannot come from the error
being diluted.

The task: given the English `source`, rank `reference` above the distractors.

## Construction

- **Windows.** Non-overlapping runs of {k_max} consecutive segments of one article; the
  row of length k holds the first k segments. Every k is measured on the same windows.
- **Distractors.** Distractor d perturbs segment j with category
  `(causality, entity, number)[(d + j) % 3]`, so neighbouring segments carry different
  types; when a segment does not admit it, the next category in that order, never an
  edit another distractor already took at that segment. A perturbation depends only
  on the segment, j and d, so **distractor d at k+1 is distractor d at k plus one
  perturbed segment**: longer blocks carry exactly the errors of the shorter ones.
- A window keeps the distractors whose first segment differs from all earlier ones
  (they then differ at every k); windows with a segment admitting no perturbation are
  dropped. Perturbations are applied to the translation (target side) by the spaCy
  generator of the FDTEM code, seed 42.
- **Single-error control.** The same distractors with only their first segment
  perturbed are `row_k1.distractors[d] + row.reference[len(row_k1.reference):]`, where
  `row_k1` is the k = 1 row of the same window and language.

## Fields

| field | |
|---|---|
| `id` | `<lang>-w<window>-k<k>` |
| `lang_pair` | `en-<lang>` |
| `window` | window id, shared by every k and every language pair |
| `k` | segments in the block |
| `split` | FLORES+ split of the article (`dev` or `devtest`) |
| `url` | the article |
| `rows` | row indices of the segments in that split, FLORES+ id order |
| `source` | the English block: the query |
| `source_n_tokens` | tokens of `source`, `xlm-roberta-large` tokenizer (the one COMET uses), special tokens excluded |
| `reference` | the correct translation block |
| `distractors` | the perturbed blocks, same order at every k |
| `categories` | `categories[d][j]`: perturbation category of segment j in distractor d |

## Statistics

| pair | windows | rows | distractors per window: windows with at least D |
|---|---|---|---|
{stats}

Categories over all distractor segments at k = {k_max}: {category_share}.

## Usage

```python
from datasets import load_dataset
rows = load_dataset("{repo_id}", "en-de", split="test")
```

Built and evaluated with `part1_block_alignment/perturb_every_sentence.py` and
`evaluate_every_sentence.py` of https://github.com/Jeerhz/FDTEM.

## License and citation

CC BY-SA 4.0, as FLORES+. Please cite FLORES+ and xSIM++:

```bibtex
@article{{nllb2022,
  title   = {{No Language Left Behind: Scaling Human-Centered Machine Translation}},
  author  = {{{{NLLB Team}}}},
  journal = {{arXiv preprint arXiv:2207.04672}},
  year    = {{2022}}
}}
@inproceedings{{chen-etal-2023-xsim,
  title     = {{xSIM++: An Improved Proxy to Bitext Mining Performance for Low-Resource Languages}},
  author    = {{Chen, Mingda and Heffernan, Kevin and {{\\c{{C}}}}elebi, Onur and Mourachko, Alexandre and Schwenk, Holger}},
  booktitle = {{Proceedings of ACL 2023}},
  year      = {{2023}}
}}
```
"""


def write_card(rows_by_lang: dict[str, list[PerturbedBlock]], repo_id: str, out: Path) -> None:
    langs = list(rows_by_lang)
    k_max = max(r.k for rows in rows_by_lang.values() for r in rows)
    n_max = max(len(r.distractors) for rows in rows_by_lang.values() for r in rows)
    stats, cats = [], Counter()
    for lang, rows in rows_by_lang.items():
        m = {r.window: len(r.distractors) for r in rows}
        cover = " ".join(f"D={D}: {sum(v >= D for v in m.values())}" for D in range(1, n_max + 1))
        stats.append(f"| en-{lang} | {len(m)} | {len(rows)} | {cover} |")
        cats.update(c for r in rows if r.k == k_max for d in r.categories for c in d)
    total = sum(cats.values())
    out.write_text(CARD.format(
        languages="\n".join(f"- {c}" for c in ["en", *langs]),
        configs="\n".join(f"- config_name: en-{lang}\n  data_files:\n  - split: test\n"
                          f"    path: data/en-{lang}.jsonl" for lang in langs),
        k_max=k_max, n_max=n_max, stats="\n".join(stats), repo_id=repo_id,
        category_share=", ".join(f"{c} {cats[c] / total:.0%}" for c in CATEGORIES)),
        encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", default="AdleBenSalem/flores-plusplus-blocks")
    ap.add_argument("--langs", nargs="+", default=["de", "es", "fr", "ru"])
    ap.add_argument("--backend", choices=["spacy", "heuristic"], default="spacy")
    ap.add_argument("--private", action="store_true", help="Keep the repository private.")
    args = ap.parse_args()

    if hf_token() is None:
        raise SystemExit("no Hugging Face token: run `hf auth login` or export HF_TOKEN")
    from huggingface_hub import HfApi

    rows_by_lang = {lang: load_rows(args.backend, lang) for lang in args.langs}
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "data").mkdir()
        for lang in args.langs:
            shutil.copy(dataset_path(args.backend, lang), folder / "data" / f"en-{lang}.jsonl")
        write_card(rows_by_lang, args.repo_id, folder / "README.md")
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(folder),
                          commit_message="FLORES+ blocks, every segment perturbed")
    if not args.private:
        api.update_repo_settings(args.repo_id, repo_type="dataset", gated="auto")
        api.update_repo_settings(args.repo_id, repo_type="dataset", private=False)
    info = api.dataset_info(args.repo_id)
    print(f"https://huggingface.co/datasets/{args.repo_id}  private={info.private} gated={info.gated}")


if __name__ == "__main__":
    main()
