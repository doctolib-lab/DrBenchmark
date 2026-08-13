import glob
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from core import runtime
from core.config import MODEL_HP


class SaveAndEvaluateLastStep(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == state.max_steps:
            control.should_evaluate = True
            control.should_save = True


def run(cfg, family):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    train, val, test, label_list = family.prepare_datasets(cfg, tokenizer)
    collator = family.build_collator(cfg, tokenizer)
    compute_metrics, report = family.build_metrics(cfg, label_list, cfg.output_name)

    best_hp = None if cfg.debug else _load_best_hp(cfg)
    do_hpo = best_hp is None
    run_dir, best_model_dir = _dirs(cfg)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    precision_args, precision = runtime.resolve_precision()

    base = {
        "output_dir": str(run_dir),
        "eval_strategy": "steps",
        "eval_steps": 0.1,
        "save_strategy": "steps",
        "save_steps": 0.1,
        **precision_args,
        "push_to_hub": False,
        "metric_for_best_model": cfg.metrics,
        "greater_is_better": cfg.direction[0] == "max",
        "seed": cfg.seed,
        "load_best_model_at_end": True,
        **cfg.fixed,
        **cfg.trainer_overrides,
    }

    if do_hpo:
        best_hp = _search(
            cfg, family, base, label_list, train, val, collator, tokenizer, compute_metrics
        )
    else:
        model_hp = {k: v for k, v in best_hp.items() if k in MODEL_HP}
        train_hp = {k: v for k, v in best_hp.items() if k not in MODEL_HP}
        trainer = Trainer(
            args=TrainingArguments(**{**base, **train_hp}),
            model=family.build_model(cfg, label_list, model_hp.get("dropout")),
            train_dataset=train,
            eval_dataset=val,
            data_collator=collator,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[SaveAndEvaluateLastStep],
        )
        trainer.train()
        trainer.save_model(str(best_model_dir))
        shutil.rmtree(run_dir, ignore_errors=True)

    logging.info("***** Evaluating the best model on test *****")
    # this Trainer only predicts: no eval loop, no checkpointing
    predict_args = {
        **base,
        "eval_strategy": "no",
        "save_strategy": "no",
        "load_best_model_at_end": False,
    }
    trainer = Trainer(
        args=TrainingArguments(**predict_args),
        model=family.build_model(cfg, label_list, path=str(best_model_dir)),
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    predictions, labels, _ = trainer.predict(test)
    metrics, preds, refs = report(predictions, labels)
    print(metrics)

    _dump(cfg, metrics, best_hp, preds, refs, test, do_hpo, best_model_dir, precision)
    shutil.rmtree(run_dir, ignore_errors=True)


def _dirs(cfg):
    return cfg.save_dir / cfg.output_name, cfg.save_dir / f"{cfg.output_name}_best_model"


def _load_best_hp(cfg):
    identity = (cfg.corpus, cfg.task, cfg.subset, cfg.lang)
    files = sorted(
        glob.glob(str(cfg.runs_dir / "*_hpo.json")), key=os.path.getmtime, reverse=True
    )
    for path in files:
        with open(path, encoding="utf-8") as handle:
            previous = json.load(handle)
        benchmark = previous.get("benchmark", {})
        keys = ("corpus", "task", "subset", "lang")
        if tuple(benchmark.get(k) for k in keys) != identity:
            continue
        if previous.get("model_id") != cfg.model_id:
            continue
        if previous.get("best_hp_trial"):
            print(f"reusing hyperparameters from {path}")
            return previous["best_hp_trial"]
    return None


def _search(cfg, family, base, label_list, train, val, collator, tokenizer, compute_metrics):
    import ray
    from ray import tune
    from ray.tune import CheckpointConfig
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.hyperopt import HyperOptSearch

    tmp_dir = os.environ.get("RAY_TMPDIR")
    ray.init(_temp_dir=tmp_dir, include_dashboard=False) if tmp_dir else ray.init(
        include_dashboard=False
    )

    metric_key = "eval_" + cfg.metrics
    mode = cfg.direction[0]
    run_dir, best_model_dir = _dirs(cfg)

    class Cleanup(tune.Callback):
        """Drop checkpoints of every trial that is not the current best."""

        def on_trial_complete(self, iteration, trials, trial, **info):
            scored = [t for t in trials if metric_key in t.metric_analysis]
            if not scored:
                return
            values = [t.metric_analysis[metric_key][mode] for t in scored]
            best = max(values) if mode == "max" else min(values)
            for candidate in scored:
                if candidate.status != "TERMINATED":
                    continue
                if candidate.metric_analysis[metric_key][mode] != best:
                    self.cleanup(candidate)

        def cleanup(self, trial):
            print(f"cleaning up trial {trial.trial_id}")
            if os.path.exists(trial.path):
                for name in os.listdir(trial.path):
                    if "checkpoint" in name:
                        shutil.rmtree(os.path.join(trial.path, name))
            staged = os.path.join(
                os.path.dirname(trial.local_experiment_path), "working_dirs", "save_models"
            )
            if os.path.exists(staged):
                staged = os.path.join(
                    staged, os.listdir(staged)[0], f"run-{trial.trial_id}"
                )
                if os.path.exists(staged):
                    shutil.rmtree(staged)

    trainer = Trainer(
        args=TrainingArguments(**base),
        model_init=lambda trial: family.build_model(
            cfg, label_list, trial["dropout"] if trial else None
        ),
        train_dataset=train,
        eval_dataset=val,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[SaveAndEvaluateLastStep],
    )

    results_dir = os.environ.get("RAY_RESULTS_DIR")
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    best = trainer.hyperparameter_search(
        direction=cfg.direction[1],
        backend="ray",
        search_alg=HyperOptSearch(metric=metric_key, mode=mode),
        scheduler=ASHAScheduler(metric=metric_key, mode=mode, **cfg.scheduler),
        hp_space=lambda trial: {
            key: eval(expr, {"tune": tune}) for key, expr in cfg.search_space.items()
        },
        resources_per_trial=runtime.trial_resources(),
        n_trials=cfg.n_trials,
        checkpoint_config=CheckpointConfig(
            checkpoint_score_attribute=metric_key,
            num_to_keep=1,
            checkpoint_score_order=mode,
        ),
        storage_path=results_dir,
        callbacks=[Cleanup()],
    )

    best_trial = best.run_summary.get_best_trial(
        mode=mode, metric=metric_key, scope="all"
    )
    scores = best.run_summary.trial_dataframes[best_trial.trial_id][metric_key]
    best_score = scores.max() if mode == "max" else scores.min()
    checkpoint = best.run_summary.get_best_checkpoint(
        best_trial, mode=mode, metric=metric_key
    )

    logging.info("***** Saving the best model *****")
    shutil.copytree(
        glob.glob(checkpoint.path + "/checkpoint-*")[0], str(best_model_dir)
    )
    shutil.rmtree(run_dir, ignore_errors=True)
    shutil.rmtree(str(Path(checkpoint.path).parents[1]), ignore_errors=True)
    ray.shutdown()

    print(f"best trial {best_trial.trial_id} with {metric_key}={best_score}")
    return best.hyperparameters


def _dump(cfg, metrics, best_hp, preds, refs, test, do_hpo, best_model_dir, precision):
    # a debug run is kept out of the glob that feeds HPO reuse and the stats
    runs_dir = cfg.runs_dir / "debug" if cfg.debug else cfg.runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    identifiers = (
        list(test["id"]) if "id" in test.column_names else list(range(len(test)))
    )
    payload = {
        "benchmark": {
            "corpus": cfg.corpus,
            "task_type": cfg.task_type,
            "task": cfg.task,
            "subset": cfg.subset,
            "lang": cfg.lang,
            "fewshot": cfg.fewshot,
        },
        "model_id": cfg.model_id,
        "model_path": str(cfg.model_path),
        "best_model_path": str(best_model_dir),
        "runtime": runtime.summary(precision),
        "metrics": metrics,
        "hpo_settings": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(cfg).items()
        },
        "best_hp_trial": best_hp,
        "predictions": {
            "identifiers": identifiers,
            "real_labels": refs,
            "system_predictions": preds,
        },
        "run_seed": cfg.seed,
    }
    suffix = "hpo" if do_hpo else "train"
    path = runs_dir / f"{cfg.output_name}_{suffix}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=4, default=_encode),
        encoding="utf-8",
    )
    print(f"saved {path}")


def _encode(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value)}")
