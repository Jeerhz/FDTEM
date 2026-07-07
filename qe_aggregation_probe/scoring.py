#!/usr/bin/env python3
"""
scoring.py — metric wrappers with a persistent score cache.

A scoring *job* is a dict {src, mt, ref, meta}. Real metrics use (src, mt[, ref])
and ignore meta; the mock metric uses meta to synthesise scores from a KNOWN
aggregation function (paragraph = power mean of its parts at cfg true_p), which
turns the smoke test into a parameter-recovery check for the whole pipeline.

Metrics
-------
  comet:<hf-id-or-ckpt>   COMET / CometKiwi via the local `comet` package.
  metricx:<hf-id>         MetricX-24(-QE) via google-research/metricx
                          (pip install git+https://github.com/google-research/metricx).
  mock                    deterministic pseudo-metric (smoke only).
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from common import md5_short, pick_device, sanitize

logger = logging.getLogger(__name__)


def job_key(metric_name: str, job: dict) -> str:
    return md5_short(f"{metric_name}|{job['src']}|{job['mt']}|{job.get('ref') or ''}", 20)


# ════════════════════════════════════════════════════════════════════════════
# Metric implementations
# ════════════════════════════════════════════════════════════════════════════
class Metric:
    name: str
    kind: str            # "quality" | "error"
    needs_ref: bool

    def score_batch(self, jobs: List[dict], batch_size: int) -> List[float]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class CometMetric(Metric):
    def __init__(self, mcfg: dict, device: str):
        from comet import download_model, load_from_checkpoint  # noqa: PLC0415
        import os  # noqa: PLC0415
        ref = mcfg["spec"].split(":", 1)[1]
        ckpt = ref if os.path.isfile(ref) else download_model(ref)
        logger.info(f"  [comet] loading {ckpt}")
        self.model = load_from_checkpoint(ckpt).eval()
        self.name = mcfg["name"]
        self.kind = mcfg["kind"]
        self.needs_ref = mcfg["needs_ref"]
        self.gpus = 1 if device == "cuda" else 0

    def score_batch(self, jobs, batch_size):
        data = []
        for j in jobs:
            d = {"src": j["src"], "mt": j["mt"]}
            if self.needs_ref:
                d["ref"] = j["ref"]
            data.append(d)
        out = self.model.predict(data, batch_size=batch_size, gpus=self.gpus,
                                 progress_bar=True)
        return [float(s) for s in out["scores"]]

    def close(self):
        del self.model


class MetricXMetric(Metric):
    """MetricX-24 QE (error scale 0–25, higher = worse).

    Requires the upstream package:  pip install git+https://github.com/google-research/metricx
    Input format follows upstream predict.py: "source: {src} candidate: {mt}"
    (QE) — tokenised to ≤1536, final EOS removed. Verify a few scores against
    upstream predict.py before trusting a new install.
    """

    MAX_LEN = 1536

    def __init__(self, mcfg: dict, device: str):
        try:
            from metricx24 import models as mx_models  # noqa: PLC0415
        except ImportError:
            try:
                from metricx23 import models as mx_models  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "metricx not installed: pip install "
                    "git+https://github.com/google-research/metricx") from exc
        import torch  # noqa: PLC0415
        from transformers import AutoTokenizer  # noqa: PLC0415
        ref = mcfg["spec"].split(":", 1)[1]
        logger.info(f"  [metricx] loading {ref}")
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(mcfg.get("tokenizer", "google/mt5-xl"))
        self.model = mx_models.MT5ForRegression.from_pretrained(
            ref, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        self.device = device
        self.model.to(device).eval()
        self.name = mcfg["name"]
        self.kind = mcfg["kind"]
        self.needs_ref = mcfg["needs_ref"]

    def _texts(self, jobs):
        if self.needs_ref:
            return [f"source: {j['src']} candidate: {j['mt']} reference: {j['ref']}"
                    for j in jobs]
        return [f"source: {j['src']} candidate: {j['mt']}" for j in jobs]

    def score_batch(self, jobs, batch_size):
        torch = self.torch
        scores: List[float] = []
        texts = self._texts(jobs)
        for s in range(0, len(texts), batch_size):
            enc = self.tok(texts[s:s + batch_size], max_length=self.MAX_LEN,
                           truncation=True, padding=True, return_tensors="pt")
            # upstream removes the trailing EOS token
            ids = enc["input_ids"][:, :-1].to(self.device)
            am = enc["attention_mask"][:, :-1].to(self.device)
            with torch.no_grad():
                out = self.model(input_ids=ids, attention_mask=am)
            scores.extend(float(x) for x in out.predictions.float().cpu().numpy())
        return scores

    def close(self):
        del self.model


class MockMetric(Metric):
    """Deterministic pseudo-metric for the logic smoke test.

    Sentence score  = 0.85 − delta(condition if perturbed) + N(0, 0.03) [hash-seeded]
    Paragraph score = power_mean(part scores, p = true_p) + N(0, noise_sd)

    Fitting the pipeline on mock data must recover p ≈ true_p — a full
    parameter-recovery test with zero heavy dependencies.
    """

    BASE = 0.85
    SENT_SD = 0.03
    DELTA = {"minor_fluency": 0.05, "minor_accuracy": 0.08,
             "major_fluency": 0.15, "major_accuracy": 0.30}

    def __init__(self, mcfg: dict):
        self.name = mcfg["name"]
        self.kind = "quality"
        self.needs_ref = False
        self.true_p = float(mcfg.get("true_p", -4.0))
        self.noise_sd = float(mcfg.get("noise_sd", 0.01))

    def _sent_score(self, text: str, is_pert: bool, condition: Optional[str]) -> float:
        rng = random.Random(f"mocksent|{text}")
        s = self.BASE + rng.gauss(0.0, self.SENT_SD)
        if is_pert and condition:
            s -= self.DELTA.get(condition, 0.1)
        return float(np.clip(s, 0.02, 1.0))

    def score_batch(self, jobs, batch_size):
        from fit import power_mean  # noqa: PLC0415  (fit.py has no scoring import)
        out = []
        for j in jobs:
            meta = j.get("meta") or {}
            if meta.get("level") == "par":
                parts = np.array([[self._sent_score(p["text"], p["is_pert"],
                                                    p.get("condition"))
                                   for p in meta["parts"]]])
                rng = random.Random(f"mockpar|{j['mt']}")
                s = float(power_mean(parts, self.true_p)[0]) + rng.gauss(0, self.noise_sd)
                out.append(float(np.clip(s, 0.01, 1.05)))
            else:
                out.append(self._sent_score(j["mt"], meta.get("is_pert", False),
                                            meta.get("condition")))
        return out


def build_metric(mcfg: dict, device: str) -> Metric:
    spec = mcfg["spec"]
    if spec.startswith("comet:"):
        return CometMetric(mcfg, device)
    if spec.startswith("metricx:"):
        return MetricXMetric(mcfg, device)
    if spec == "mock":
        return MockMetric(mcfg)
    raise ValueError(f"unknown metric spec {spec!r}")


# ════════════════════════════════════════════════════════════════════════════
# Persistent score cache
# ════════════════════════════════════════════════════════════════════════════
class ScoreCache:
    def __init__(self, cache_dir: str, metric_name: str):
        self.path = Path(cache_dir) / f"{sanitize(metric_name)}.json"
        self.cache: Dict[str, float] = (json.loads(self.path.read_text(encoding="utf-8"))
                                        if self.path.exists() else {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache), encoding="utf-8")


def score_jobs(cfg: dict, mcfg: dict, jobs: List[dict]) -> Dict[str, float]:
    """Score all jobs with one metric, deduped + disk-cached.

    Returns {job_key: score}. Sentence-level jobs must precede paragraph jobs in
    `jobs` only for readability; dedup handles ordering.
    """
    name = mcfg["name"]
    cache = ScoreCache(cfg["paths"]["score_cache"], name)
    uniq: Dict[str, dict] = {}
    for j in jobs:
        uniq.setdefault(job_key(name, j), j)
    missing = {k: j for k, j in uniq.items() if k not in cache.cache}
    logger.info(f"  [{name}] {len(uniq)} unique jobs, {len(missing)} to score "
                f"({len(uniq) - len(missing)} cached)")
    if missing:
        device = pick_device(cfg["scoring"].get("device"))
        metric = build_metric(mcfg, device)
        keys = list(missing.keys())
        todo = [missing[k] for k in keys]
        scores = metric.score_batch(todo, cfg["scoring"]["batch_size"])
        for k, s in zip(keys, scores):
            cache.cache[k] = float(s)
        cache.save()
        metric.close()
    return {k: cache.cache[k] for k in uniq}
