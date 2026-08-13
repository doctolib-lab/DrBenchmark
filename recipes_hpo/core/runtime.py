"""Execution environment: what the machine can do, never what the task is.

Nothing here may change a result silently — a degraded precision is printed and
recorded in the run JSON.
"""

import os

import torch

# protocol default; overriding it changes results, hence the record in summary()
PRECISION = os.environ.get("DRB_PRECISION", "bf16")

PRECISION_ARGS = {"bf16": {"bf16": True}, "fp16": {"fp16": True}, "fp32": {}}


def trial_resources():
    """Ray resources per trial: one GPU when there is one, CPUs split across concurrent trials."""
    gpus = torch.cuda.device_count()
    per_trial_gpu = float(os.environ.get("DRB_TRIAL_GPUS", 1 if gpus else 0))
    concurrent = max(1, gpus) if per_trial_gpu else 1
    spare = max(1, (os.cpu_count() or 1) - 2)
    per_trial_cpu = int(os.environ.get("DRB_TRIAL_CPUS", max(1, spare // concurrent)))
    return {"cpu": per_trial_cpu, "gpu": per_trial_gpu}


def available_precisions():
    if not torch.cuda.is_available():
        return ["fp32"]
    try:
        best = ["bf16"] if torch.cuda.is_bf16_supported() else []
    except Exception:
        best = []
    return best + ["fp16", "fp32"]


def resolve_precision(requested=None):
    """Return (TrainingArguments kwargs, effective precision), degrading if unsupported."""
    requested = requested or PRECISION
    available = available_precisions()
    effective = requested if requested in available else available[0]
    if effective != requested:
        print(
            f"precision: {requested} unavailable on {device()}, falling back to {effective}. "
            "Results are not comparable with reference runs."
        )
    return PRECISION_ARGS[effective], effective


def device():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def summary(precision):
    return {
        "precision": precision,
        "precision_requested": PRECISION,
        "device": device(),
        "trial_resources": trial_resources(),
    }
