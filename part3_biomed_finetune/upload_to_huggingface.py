"""Publish a fine-tuned COMET checkpoint on the Hugging Face Hub.

Validates the checkpoint with comet.load_from_checkpoint, exports it (raw
.ckpt + hparams.yaml + HF-format encoder), writes a model card and uploads
the folder; optionally records the Hub URL on the W&B run.

    python -m part3_biomed_finetune.upload_to_huggingface --checkpoint auto \
        --ckpt_dir ~/scratch/checkpoints/bio_mqm --repo_id <user>/comet-bio-mqm \
        [--run_name comet-bio-mqm-<date>] [--private] [--wandb_run_id <id>]

Needs `hf auth login` once per machine (the token is resolved by common.auth.hf_token).
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from common.auth import hf_token
from common.paths import CHECKPOINTS


def load_and_export(ckpt_path: Path, export_dir: Path) -> dict:
    """Load the checkpoint and write hparams.yaml + encoder/ (HF format) into export_dir."""
    from comet import load_from_checkpoint

    print(f"loading {ckpt_path}")
    model = load_from_checkpoint(str(ckpt_path))
    hparams = dict(model.hparams) if hasattr(model, "hparams") else {}

    encoder_dir = export_dir / "encoder"
    encoder_dir.mkdir()
    model.encoder.model.save_pretrained(str(encoder_dir))
    model.encoder.tokenizer.save_pretrained(str(encoder_dir))

    hparams_src = ckpt_path.parents[1] / "hparams.yaml"
    if hparams_src.is_file():
        shutil.copy(hparams_src, export_dir / "hparams.yaml")
    else:
        import yaml
        (export_dir / "hparams.yaml").write_text(yaml.dump(hparams))
    return hparams


def write_model_card(export_dir: Path, repo_id: str, run_name: str | None, hparams: dict) -> None:
    base_model = hparams.get("pretrained_model", "xlm-roberta-large")
    lang_pairs = ["de<->en", "es<->en", "fr<->en", "ru<->en", "zh<->en"]
    languages = "\n".join(f"- {code}" for code in ["de", "en", "es", "fr", "ru", "zh"])
    run_row = f"| Run | {run_name} |" if run_name else ""
    card = f"""\
---
language:
{languages}
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
fine-tuned on the [Amazon Bio-MQM](https://github.com/amazon-science/bio-mqm-dataset)
biomedical translation evaluation dataset.

## Training details

| Parameter | Value |
|---|---|
| Base model | `{base_model}` |
| Fine-tuning data | Amazon Bio-MQM (dev splits) |
| Language pairs | {', '.join(lang_pairs)} |
| Loss | {hparams.get('loss', 'mse')} |
| Encoder LR | {hparams.get('encoder_learning_rate', 'N/A')} |
| Head LR | {hparams.get('learning_rate', 'N/A')} |
| Batch size | {hparams.get('batch_size', 'N/A')} |
{run_row}

## Usage

```python
from comet import download_model, load_from_checkpoint

model = load_from_checkpoint(download_model("{repo_id}"))
data = [{{
    "src": "The patient was administered 500 mg of amoxicillin.",
    "mt":  "Der Patient erhielt 500 mg Amoxicillin.",
    "ref": "Dem Patienten wurden 500 mg Amoxicillin verabreicht.",
}}]
print(model.predict(data, batch_size=8, gpus=1).scores)
```

## Citation

Please cite the COMET paper and the Bio-MQM dataset:

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
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"uploading to https://huggingface.co/{repo_id}")
    api.upload_folder(folder_path=str(export_dir), repo_id=repo_id, repo_type="model",
                      commit_message="Upload fine-tuned COMET Bio-MQM model")
    return f"https://huggingface.co/{repo_id}"


def log_to_wandb(project: str, run_id: str, hf_url: str) -> None:
    try:
        import wandb
        run = wandb.Api().run(f"{project}/{run_id}")
        run.summary["hf_model_url"] = hf_url
        run.summary.update()
        print(f"logged the Hub URL on W&B run {run_id}")
    except Exception as exc:  # noqa: BLE001
        print(f"W&B logging skipped: {exc}")


def resolve_checkpoint_arg(ref: str, ckpt_dir: Path) -> Path:
    if ref == "auto":
        from common.comet_models import best_checkpoint
        ckpt = best_checkpoint(ckpt_dir)
        if ckpt is None:
            raise SystemExit(f"no checkpoint under {ckpt_dir}")
        return ckpt
    ckpt = Path(ref).expanduser()
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    return ckpt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="auto", help=".ckpt path, or 'auto' = best checkpoint under --ckpt_dir")
    ap.add_argument("--ckpt_dir", default=str(CHECKPOINTS / "bio_mqm"))
    ap.add_argument("--repo_id", required=True, help="'username/model-name'")
    ap.add_argument("--run_name", default=None, help="added to the model card")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--wandb_run_id", default=None, help="record the Hub URL on this W&B run")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "comet-bio-mqm"))
    args = ap.parse_args()

    if hf_token() is None:
        raise SystemExit("no Hugging Face token: run `hf auth login` or export HF_TOKEN")
    ckpt_path = resolve_checkpoint_arg(args.checkpoint, Path(args.ckpt_dir).expanduser())

    with tempfile.TemporaryDirectory() as tmp:
        export_dir = Path(tmp) / "hf_export"
        export_dir.mkdir()
        hparams = load_and_export(ckpt_path, export_dir)
        # load_from_checkpoint looks for hparams.yaml two levels above the .ckpt
        (export_dir / "checkpoints").mkdir()
        shutil.copy(ckpt_path, export_dir / "checkpoints" / "model.ckpt")
        write_model_card(export_dir, args.repo_id, args.run_name, hparams)
        hf_url = upload_to_hub(export_dir, args.repo_id, args.private)

    if args.wandb_run_id:
        log_to_wandb(args.wandb_project, args.wandb_run_id, hf_url)
    print(f"model available at {hf_url}")


if __name__ == "__main__":
    main()
