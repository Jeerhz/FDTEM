# Publicly available data for Metrics:

# Direct Assessments:

Every year the WMT News Translation task organizers collect thousands of quality annotations in the form of _Direct Assessments_. Most COMET models use this data for training.

In the table below you can find that data in an easy to use format:

| year | data | paper |
|:---: | :--: | :---: |
| 2017 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2017-da.tar.gz) | [Findings of the 2017 Conference on Machine Translation (WMT17)](https://aclanthology.org/W17-4717.pdf) |
| 2018 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2018-da.tar.gz) | [Findings of the 2018 Conference on Machine Translation (WMT18)](https://aclanthology.org/W18-6401.pdf) |
| 2019 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2019-da.tar.gz) | [Findings of the 2019 Conference on Machine Translation (WMT19)](https://aclanthology.org/W19-5301.pdf) |
| 2020 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2020-da.tar.gz) | [Findings of the 2020 Conference on Machine Translation (WMT20)](https://aclanthology.org/2020.wmt-1.1.pdf) |
| 2021 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2021-da.tar.gz) | [Findings of the 2021 Conference on Machine Translation (WMT21)](https://aclanthology.org/2021.wmt-1.1.pdf) |
| 2022 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2022-da.tar.gz) | [Findings of the 2022 Conference on Machine Translation (WMT22)](https://aclanthology.org/2022.wmt-1.1.pdf) |

Another large source of DA annotations is the [MLQE-PE corpus](https://aclanthology.org/2022.lrec-1.530.pdf) that is typically used for quality estimation shared tasks [(Specia et al. 2020](https://aclanthology.org/2020.wmt-1.79.pdf)[, 2021](https://aclanthology.org/2021.wmt-1.71.pdf)[; Zerva et al. 2022)](https://aclanthology.org/2022.wmt-1.3.pdf).

You can download MLQE-PE by using the following [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/mlqe-pe.tar.gz).

### DA Relative Ranks
Before 2021 the WMT Metrics shared task used relative ranks to evaluate metrics. 

Relative ranks can be created when we have at least two DA scores for translations of the same source input, by converting those DA scores into a relative ranking judgement, if the difference in DA scores allows conclusion that one translation is better than the other (usually atleast 25 points). 

To make it easier to replicate results from previous Metrics shared tasks (2017-2020) you can find the preprocessed DA relative ranks in the table below:

| year | relative ranks | paper |
|:---: | :--: | :---: |
| 2017 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2017-daRR.csv.tar.gz) | [Results of the WMT17 Metrics Shared Task](https://statmt.org/wmt17/pdf/WMT55.pdf) |
| 2018 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2018-daRR.csv.tar.gz) | [Results of the WMT18 Metrics Shared Task](https://statmt.org/wmt18/pdf/WMT078.pdf) |
| 2019 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2019-daRR.csv.tar.gz) | [Results of the WMT19 Metrics Shared Task](https://statmt.org/wmt19/pdf/53/WMT02.pdf) |
| 2020 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2020-daRR.csv.tar.gz) | [Results of the WMT20 Metrics Shared Task](https://aclanthology.org/2020.wmt-1.77.pdf) |

# Multidimensional Quality Metrics:

Since 2021 the WMT Metrics task decided to perform they own expert-based evaluation based on _Multidimensional Quality Metrics (MQM)_ framework. In the table below you can find MQM annotations from previous years.

| year | data | paper |
|:---: | :--: | :---: |
| 2020 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2020-mqm.tar.gz) | [A Large-Scale Study of Human Evaluation for Machine Translation](https://aclanthology.org/2021.tacl-1.87.pdf) |
| 2021 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2021-mqm.tar.gz) | [Results of the WMT21 Metrics Shared Task](https://aclanthology.org/2021.wmt-1.73.pdf) |
| 2022 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2022-mqm.tar.gz) | [Results of the WMT22 Metrics Shared Task](https://aclanthology.org/2022.wmt-1.2.pdf) |
| 2023 | [🔗](https://github.com/google-research/mt-metrics-eval) | [Results of the WMT23 Metrics Shared Task](https://aclanthology.org/2023.wmt-1.51.pdf) |
| 2024 | [🔗](https://github.com/google-research/mt-metrics-eval) | [Are LLMs Breaking MT Metrics? Results of the WMT24 Metrics Shared Task](https://aclanthology.org/2024.wmt-1.2.pdf) |
| 2025 | [🔗](https://github.com/google-research/mt-metrics-eval) | [Findings of the WMT25 Shared Task on Automated Translation Evaluation Systems](https://aclanthology.org/2025.wmt-1.24.pdf) |

**Note:** You can find the original MQM data (up to 2022) [here](https://github.com/google/wmt-mqm-human-evaluation). For 2023 onwards, MQM annotations are distributed via the [mt-metrics-eval](https://github.com/google-research/mt-metrics-eval) repository.

# Direct Assessment + Scalar Quality Metric:

In 2022, several changes were made to the annotation procedure used in the WMT Translation task. In contrast to the standard DA (sliding scale from 0-100) used in previous years, in 2022 annotators performed DA+SQM (Direct Assessment + Scalar Quality Metric). In DA+SQM, the annotators still provide a raw score between 0 and 100, but also are presented with seven labeled tick marks. DA+SQM helps to stabilize scores across annotators (as compared to DA).

| year | data | paper |
|:---: | :--: | :---: |
| 2022 | [🔗](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/comet/data/2022-sqm.tar.gz) | [Findings of the 2022 Conference on Machine Translation (WMT22)](https://aclanthology.org/2022.wmt-1.1.pdf) |
| 2023 | [🔗](https://github.com/google-research/mt-metrics-eval) | [Findings of the 2023 Conference on Machine Translation (WMT23)](https://aclanthology.org/2023.wmt-1.1.pdf) |

# Error Span Annotations:

Starting with WMT24, the WMT human evaluation protocol switched from DA+SQM to _Error Span Annotations (ESA)_. ESA combines the continuous rating of DA with the high-level error severity span marking of MQM. ESA is faster than MQM, does not require expert annotators, and produces more stable assessments than standard DA. MQM continues to be used for a subset of language pairs.

| year | data | paper |
|:---: | :--: | :---: |
| 2024 | [🔗](https://github.com/wmt-conference/ErrorSpanAnnotation) | [Findings of the WMT24 General Machine Translation Shared Task](https://aclanthology.org/2024.wmt-1.1.pdf) |
| 2025 | [🔗](https://github.com/wmt-conference/ErrorSpanAnnotation) | [Findings of the WMT25 General Machine Translation Shared Task](https://aclanthology.org/2025.wmt-1.22.pdf) |

# Document-Level Evaluation:

Evaluating translations beyond the sentence level requires datasets that preserve document structure and capture discourse phenomena. The following resources are used to train and benchmark document-level metrics.

## Document-Level Parallel Corpora

Large-scale document-aligned parallel corpora for training and evaluating document-level MT systems and metrics.

| dataset | coverage | data | paper |
|:---: | :--: | :--: | :---: |
| WMT24++ | 55 languages × English, 4 domains (news, literary, social, speech) | [🔗](https://huggingface.co/datasets/google/wmt24pp) | [WMT24++: Expanding the Language Coverage of WMT24 to 55 Languages & Dialects](https://aclanthology.org/2025.findings-acl.634.pdf) |
| DocHPLT | 50 languages × English, 124M document pairs | [🔗](https://huggingface.co/datasets/HPLT/DocHPLT) | [DocHPLT: A Massively Multilingual Document-Level Translation Dataset](https://arxiv.org/pdf/2508.13079) |

## Contrastive Challenge Sets

Contrastive challenge sets inject controlled perturbations into MT outputs and test whether metrics assign higher scores to the original. The following datasets target discourse and document-level phenomena specifically.

| dataset | phenomena | language pairs | data | paper |
|:---: | :--: | :--: | :--: | :---: |
| ACES | 68 accuracy error categories | 146 pairs | [🔗](https://github.com/EdinburghNLP/ACES) | [ACES: Translation Accuracy Challenge Sets for Evaluating MT Metrics](https://aclanthology.org/2022.wmt-1.44.pdf) |
| DEMETR | 35 linguistic perturbations | 10 source languages → en | [🔗](https://github.com/marzenakrp/demetr) | [DEMETR: Diagnosing Evaluation Metrics for Translation](https://aclanthology.org/2022.emnlp-main.649.pdf) |
| BlonDe | discourse-related spans (pronouns, tense, connectives) | ZH ↔ EN | [🔗](https://github.com/EleanorJiang/BlonDe) | [BlonDe: An Automatic Evaluation Metric for Document-level MT](https://aclanthology.org/2022.naacl-main.111.pdf) |
| ContraPro | anaphoric pronoun gender agreement | EN–DE | [🔗](https://github.com/ZurichNLP/ContraPro) | [A Large-Scale Test Set for Context-Aware Pronoun Translation](https://aclanthology.org/W18-6307.pdf) |

