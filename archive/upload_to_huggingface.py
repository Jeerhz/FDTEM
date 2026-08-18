#!/usr/bin/env python3
"""
Upload a finetuned COMET checkpoint to Hugging Face Hub.

What it does
────────────
1. Loads the .ckpt with COMET's load_from_checkpoint() to verify it is valid.
2. Saves the model in HF-compatible format (config + weights) to a temp dir.
3. Uploads everything to the specified HF repo (creates it if it doesn't exist).
4. Optionally logs the HF model URL back to Weights & Biases.

Usage
─────
    python scripts/upload_to_huggingface.py \
        --checkpoint  checkpoints/bio_mqm/epoch=4-step=3000-val_kendall=0.812.ckpt \
        --repo_id     your-username/comet-bio-mqm \
        --run_name    comet-bio-mqm-20240901-1200 \
        [--private]   \
        [--wandb_run_id  <wandb_run_id>]

Requirements
────────────
    pip install huggingface_hub wandb unbabel-comet
    huggingface-cli login   # once per machine
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload a COMET checkpoint to Hugging Face Hub."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the .ckpt file produced by comet-train.",
    )
    parser.add_argument(
        "--repo_id", required=True,
        help="HF repo in the form 'username/model-name'.",
    )
    parser.add_argument(
        "--run_name", default=None,
        help="Human-readable run name added to the model card.",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create a private HF repository.",
    )
    parser.add_argument(
        "--wandb_run_id", default=None,
        help="W&B run ID to log the HF model URL to (optional).",
    )
    parser.add_argument(
        "--wandb_project", default=os.environ.get("WANDB_PROJECT", "comet-bio-mqm"),
        help="W&B project name (used only when --wandb_run_id is set).",
    )
    return parser.parse_args()


def load_and_export(ckpt_path: str, export_dir: Path) -> dict:
    """
    Load the COMET checkpoint and export it to `export_dir`.

    Layout written to export_dir:
      model.ckpt          — raw Lightning checkpoint (for load_from_checkpoint)
      hparams.yaml        — hyperparameters (required by load_from_checkpoint)
      encoder/            — HF-format XLM-R encoder + tokenizer (for from_pretrained)
    """
    import shutil
    from comet import load_from_checkpoint

    ckpt_path = Path(ckpt_path)
    print(f"Loading checkpoint: {ckpt_path}")
    model = load_from_checkpoint(str(ckpt_path))

    hparams = dict(model.hparams) if hasattr(model, "hparams") else {}

    # -- HF encoder + tokenizer (enables encoder-only use via from_pretrained) --
    encoder_dir = export_dir / "encoder"
    encoder_dir.mkdir()
    print(f"Saving HF encoder to: {encoder_dir}")
    model.encoder.model.save_pretrained(str(encoder_dir))
    model.encoder.tokenizer.save_pretrained(str(encoder_dir))

    # -- hparams.yaml next to the checkpoint (required by load_from_checkpoint) --
    hparams_src = ckpt_path.parents[1] / "hparams.yaml"
    if hparams_src.is_file():
        shutil.copy(hparams_src, export_dir / "hparams.yaml")
    else:
        import yaml
        with open(export_dir / "hparams.yaml", "w") as fh:
            yaml.dump(hparams, fh)

    return hparams


def write_model_card(export_dir: Path, repo_id: str, run_name: str | None,
                     hparams: dict) -> None:
    """Write a minimal README.md / model card."""
    base_model = hparams.get("pretrained_model", "xlm-roberta-large")
    lang_pairs = [
        "de↔en", "es↔en", "fr↔en", "ru↔en", "zh↔en",
    ]
    card = f"""\
---
language:
{chr(10).join('- ' + lp[:2] for lp in ['de','en','es','fr','ru','zh'])}
license: apache-2.0
tags:
  - comet
  - machine-translation-evaluation
  - biomedical
  - mqm
base_model: Unbabel/wmt22-comet-da
---

# {repo_id.split('/')[-1]}

A domain-adapted version of [wmt22-comet-da](https://huggingface.co/Unbabel/wmt22-comet-da)
finetuned on the [Amazon Bio-MQM](https://github.com/amazon-science/bio-mqm-dataset)
biomedical translation evaluation dataset.

## Training details

| Parameter | Value |
|---|---|
| Base model | `{base_model}` |
| Finetuning data | Amazon Bio-MQM (dev splits) |
| Language pairs | {', '.join(lang_pairs)} |
| Loss | {hparams.get('loss', 'mse')} |
| Encoder LR | {hparams.get('encoder_learning_rate', 'N/A')} |
| Head LR | {hparams.get('learning_rate', 'N/A')} |
| Batch size | {hparams.get('batch_size', 'N/A')} |
{f'| Run | {run_name} |' if run_name else ''}

## Usage

```python
from comet import download_model, load_from_checkpoint

model_path = download_model("{repo_id}")
model = load_from_checkpoint(model_path)

data = [{{
    "src": "The patient was administered 500 mg of amoxicillin.",
    "mt":  "Der Patient erhielt 500 mg Amoxicillin.",
    "ref": "Dem Patienten wurden 500 mg Amoxicillin verabreicht.",
}}]
output = model.predict(data, batch_size=8, gpus=1)
print(output.scores)
```

## Citation

If you use this model, please cite the original COMET paper and the Bio-MQM dataset:

```bibtex
@inproceedings{{rei-etal-2020-comet,
  title     = {{COMET: A Neural Framework for MT Evaluation}},
  author    = {{Rei, Ricardo and Stewart, Craig and Farinha, Ana C and Lavie, Alon}},
  booktitle = {{Proceedings of EMNLP 2020}},
}}

@inproceedings{{bio-mqm-2024,
  title     = {{Fine-Tuned Machine Translation Metrics Struggle in Unseen Domains}},
  booktitle = {{Proceedings of ACL 2024}},
}}
```
"""
    (export_dir / "README.md").write_text(card)


def upload_to_hub(export_dir: Path, repo_id: str, private: bool) -> str:
    """Push the exported directory to the HF Hub. Returns the repo URL."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"Uploading to https://huggingface.co/{repo_id} …")
    api.upload_folder(
        folder_path=str(export_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload finetuned COMET Bio-MQM model",
    )
    url = f"https://huggingface.co/{repo_id}"
    print(f"Upload complete: {url}")
    return url


def log_to_wandb(project: str, run_id: str, hf_url: str) -> None:
    """Log the HF URL as a W&B run summary entry."""
    try:
        import wandb
        api = wandb.Api()
        run = api.run(f"{project}/{run_id}")
        run.summary["hf_model_url"] = hf_url
        run.summary.update()
        print(f"Logged HF URL to W&B run {run_id}.")
    except Exception as exc:
        print(f"W&B logging skipped: {exc}")


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    with tempfile.TemporaryDirectory() as tmp:
        export_dir = Path(tmp) / "hf_export"
        export_dir.mkdir()

        hparams = load_and_export(ckpt_path, export_dir)

        # Place the .ckpt in checkpoints/ so load_from_checkpoint finds hparams.yaml
        # two levels up (export_dir/hparams.yaml, export_dir/checkpoints/model.ckpt)
        ckpt_dest = export_dir / "checkpoints"
        ckpt_dest.mkdir()
        shutil.copy(ckpt_path, ckpt_dest / "model.ckpt")

        write_model_card(export_dir, args.repo_id, args.run_name, hparams)

        hf_url = upload_to_hub(export_dir, args.repo_id, args.private)

    if args.wandb_run_id:
        log_to_wandb(args.wandb_project, args.wandb_run_id, hf_url)

    print(f"\nModel available at: {hf_url}")


if __name__ == "__main__":
    main()
