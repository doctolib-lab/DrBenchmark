import argparse
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import yaml

from core.paths import ROOT

FAMILIES = {
    "token_classification",
    "sequence_classification",
    "multilabel_classification",
    "regression",
    "multiple_choice",
}

TEXT_FAMILIES = {
    "sequence_classification",
    "multilabel_classification",
    "regression",
    "multiple_choice",
}

# hyperparameters consumed by the model config, not by TrainingArguments
MODEL_HP = {"dropout"}

DEDUP_SCOPES = {"none", "cascade", "against_test"}

DEBUG = os.environ.get("DEBUG", "0") not in ("0", "")


def _fail(message):
    raise SystemExit(f"config error: {message}")


def load(config_path, model_id, seed):
    path = Path(config_path).resolve()
    if not path.name.endswith("_hpo.yaml"):
        _fail(f"{path.name} does not match *_hpo.yaml")

    raw = yaml.safe_load(path.read_text())
    recipe_dir = path.parent

    dedup = {"scope": "none", "drop_empty": False, "drop_all_negative": False}
    dedup.update(raw.get("dedup", {}))

    cfg = SimpleNamespace(
        task_type=raw["task_type"],
        corpus=raw["corpus"],
        task=raw["task"],
        subset=raw["subset"],
        lang=raw["lang"],
        hf_loader=raw["hf_loader"],
        label_column=raw["label_column"],
        text_columns=raw.get("text_columns", []),
        text_separator=raw.get("text_separator", " "),
        max_position_embeddings=raw["max_position_embeddings"],
        fewshot=raw.get("fewshot", 1.0),
        merge_subsets=raw.get("merge_subsets", []),
        slow_tokenizer=raw.get("slow_tokenizer", False),
        dedup=dedup,
        metrics=raw["metrics"],
        direction=raw["direction"],
        threshold=raw.get("threshold"),
        search_space=raw.get("search_space", {}),
        fixed=raw.get("fixed", {}),
        scheduler=raw.get("scheduler", {}),
        n_trials=raw["n_trials"],
        hooks=raw.get("hooks", {}),
        trainer_overrides=raw.get("trainer_overrides", {}),
        config_path=str(path),
        recipe_dir=recipe_dir,
        data_dir=recipe_dir / "data",
        runs_dir=recipe_dir / "runs",
        save_dir=recipe_dir / "save_models",
        model_id=model_id,
        seed=seed,
    )
    _validate(cfg)

    cfg.offline = bool(yaml.safe_load((ROOT / "config.yaml").read_text())["offline"])
    cfg.model_path = (
        str(ROOT / "models" / model_id.lower().replace("/", "_"))
        if cfg.offline
        else model_id
    )
    cfg.output_name = f"{cfg.corpus}-{cfg.task}-{cfg.subset}-{uuid.uuid4().hex}"
    cfg.debug = DEBUG
    if DEBUG:
        cfg.n_trials = 1
        print("DEBUG: 1 trial, no hyperparameter reuse, run JSON written to runs/debug/")
    return cfg


def _validate(cfg):
    if cfg.task_type not in FAMILIES:
        _fail(f"unknown task_type {cfg.task_type!r}")
    if cfg.lang != cfg.recipe_dir.parent.name:
        _fail(f"lang {cfg.lang!r} != language directory {cfg.recipe_dir.parent.name!r}")
    if cfg.corpus != cfg.recipe_dir.name:
        _fail(f"corpus {cfg.corpus!r} != recipe directory {cfg.recipe_dir.name!r}")
    if len(cfg.direction) != 2 or cfg.direction[0] not in ("max", "min"):
        _fail(f"direction must be [max|min, maximize|minimize], got {cfg.direction!r}")
    if cfg.direction[1] != {"max": "maximize", "min": "minimize"}[cfg.direction[0]]:
        _fail(f"inconsistent direction {cfg.direction!r}")
    if cfg.dedup["scope"] not in DEDUP_SCOPES:
        _fail(f"dedup.scope must be one of {sorted(DEDUP_SCOPES)}")
    overlap = set(cfg.search_space) & set(cfg.fixed)
    if overlap:
        _fail(f"keys present in both search_space and fixed: {sorted(overlap)}")
    if not (ROOT / cfg.hf_loader).exists():
        _fail(f"loader not found: {cfg.hf_loader}")
    if cfg.task_type in TEXT_FAMILIES and not cfg.text_columns:
        _fail(f"{cfg.task_type} needs text_columns")
    if cfg.task_type == "multilabel_classification" and cfg.threshold is None:
        _fail("multilabel_classification needs threshold")
    if cfg.task_type == "regression" and len(cfg.text_columns) != 2:
        _fail(f"regression takes exactly two text_columns, got {cfg.text_columns!r}")
    if cfg.task_type == "multiple_choice" and len(cfg.text_columns) < 3:
        _fail(f"multiple_choice takes a source and its candidates, got {cfg.text_columns!r}")
    if cfg.merge_subsets and cfg.subset in cfg.merge_subsets:
        _fail(f"subset {cfg.subset!r} names the pooled corpus, not one of merge_subsets")
    if cfg.hooks:
        _fail("hooks is declared but not implemented yet")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return load(args.config, args.model, args.seed)
