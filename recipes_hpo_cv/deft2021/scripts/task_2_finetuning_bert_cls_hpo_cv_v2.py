#!/usr/bin/env python3

import os, glob
import shutil
import uuid
import json
import logging
import numpy as np
from datasets import load_dataset, load_from_disk, concatenate_datasets

from utils_hpo_cls import parse_args

from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

import ray
tmp_dir = os.environ.get("RAY_TMPDIR")
ray.init(_temp_dir=tmp_dir, include_dashboard=False) if tmp_dir else ray.init(include_dashboard=False)

THRESHOLD_VALUE = 0.70

def toLogits(predictions, threshold=THRESHOLD_VALUE):
    import torch
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = (probs >= threshold).float().numpy()
    return y_pred

def multi_label_metrics(predictions, labels, threshold=THRESHOLD_VALUE):
    y_pred = toLogits(predictions, threshold)
    y_true = labels
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average="macro", zero_division=0)
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro", zero_division=0)
    f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average="weighted", zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    except Exception:
        roc_auc = 0.0
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "f1_macro": f1_macro_average,
        "f1_micro": f1_micro_average,
        "f1_weighted": f1_weighted_average,
        "f1": f1_weighted_average,  # alias to satisfy eval_f1 expectations
        "accuracy": accuracy,
        "roc": roc_auc,
    }

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids, threshold=THRESHOLD_VALUE)
    return result


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    if args.offline == True:
        dataset = load_from_disk(f"{args.data_dir.rstrip('/')}/local_hf_{args.subset}/")
    else:
        dataset = load_dataset(
            "Dr-BERT/DEFT2021",
            name="cls",
            data_dir=args.data_dir,
        )

    args.fold -= 1
    # Retrieve past best_hp_trial, if any:
    # search_pattern = "../runs/*_fold*.json"
    search_pattern = "../runs/DrBenchmark-DEFT2021-cls*_fold*.json"
    do_hpo = True
    matching_files = glob.glob(search_pattern)

    for file in matching_files:
        with open(file, "r", encoding="utf-8") as f:
            data_fold = json.load(f)
        model = data_fold["hpo_settings"]["model_name"].split("/")[-1]
        if model != args.model_name.split("/")[-1]:
            continue
        if data_fold["hpo_settings"]["fold"] != args.fold:
            continue
        best_hp_trial = data_fold["best_hp_trial"]
        for key, value in best_hp_trial.items():
            setattr(args, key, value)
        do_hpo = False

    # Concatenate splits and shuffle
    dataset = concatenate_datasets(
        [dataset["train"], dataset["validation"], dataset["test"]]
    ).shuffle(seed=42)

    # Create 5 shards (folds)
    num_folds = 5
    shards = [dataset.shard(num_shards=num_folds, index=i) for i in range(num_folds)]

    # Allocate each shard to a split
    dataset = {
        "test": shards[args.fold],
        "validation": shards[(args.fold + 1) % num_folds],
        "train": concatenate_datasets(
            [
                shards[(args.fold + 2) % num_folds],
                shards[(args.fold + 3) % num_folds],
                shards[(args.fold + 4) % num_folds],
            ]
        ),
    }
    # Keep ~10% validation: split validation, half into train
    dataset["train"] = concatenate_datasets(
        [dataset["validation"].shard(num_shards=2, index=0), dataset["train"]]
    )
    dataset["validation"] = dataset["validation"].shard(num_shards=2, index=1)

    # Get label space
    labels_list = dataset["train"].features["specialities"].feature.names

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess_function(e):
        res = tokenizer(
            e["text"],
            truncation=True,
            max_length=args.max_position_embeddings,
            padding="max_length",
        )
        res["labels"] = e["specialities_one_hot"]
        return res

    dataset_train = dataset["train"].map(preprocess_function, batched=False).shuffle(seed=42).shuffle(seed=42).shuffle(seed=42)
    if args.fewshot != 1.0:
        dataset_train = dataset_train.select(range(int(len(dataset_train) * args.fewshot)))
    dataset_train = dataset_train.remove_columns(["text", "id", "specialities"])
    dataset_train.set_format("torch")

    dataset_val = dataset["validation"].map(preprocess_function, batched=False)
    dataset_val = dataset_val.remove_columns(["text", "id", "specialities"])
    dataset_val.set_format("torch")

    dataset_test = dataset["test"].map(preprocess_function, batched=False)
    dataset_test_ids = list(dataset["test"]["id"])
    dataset_test = dataset_test.remove_columns(["text", "id", "specialities"])
    dataset_test.set_format("torch")

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = f"DrBenchmark-DEFT2021-cls-{uuid.uuid4()}_fold={args.fold}"

    training_args = {
        k: v
        for k, v in vars(args).items()
        if k
        in [
            "per_device_train_batch_size",
            "learning_rate",
            "num_train_epochs",
            "weight_decay",
            "warmup_ratio",
            "gradient_accumulation_steps",
        ]
    }
    training_args_base = {
        "output_dir": f"{args.output_dir}/{output_name}",
        "eval_strategy": "steps",
        "eval_steps": 0.1,
        # "save_strategy": "no",
        "save_strategy": "steps",
        "save_steps": 0.1,
        "bf16": True,
        "push_to_hub": False,
        "metric_for_best_model": args.metrics,
        "greater_is_better": True if args.direction[0] == "max" else False,
    }
    training_args = {**training_args_base, **training_args}
    training_args = TrainingArguments(
        **training_args_base if do_hpo else training_args,
    )

    absolute_path_to_model = os.path.abspath(args.model_name)
    if do_hpo:
        def model_init(trial):
            model = AutoModelForSequenceClassification.from_pretrained(
                absolute_path_to_model, num_labels=len(labels_list), problem_type="multi_label_classification"
            )
            if trial is not None:
                model.config.update(
                    {
                        "attention_probs_dropout_prob": trial["dropout"],
                        "classifier_dropout": trial["dropout"],
                        "hidden_dropout_prob": trial["dropout"],
                    }
                )
            return model

        def hp_space(trial):
            from ray import tune
            return {
                # "per_device_train_batch_size": eval(args.per_device_train_batch_size),
                "learning_rate": eval(args.learning_rate),
                "num_train_epochs": eval(args.num_train_epochs),
                "weight_decay": eval(args.weight_decay),
                "warmup_ratio": eval(args.warmup_ratio),
                "dropout": eval(args.dropout),
                "gradient_accumulation_steps": eval(args.gradient_accumulation_steps),
            }
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            absolute_path_to_model, num_labels=len(labels_list), problem_type="multi_label_classification"
        )
        model.config.update(
            {
                "attention_probs_dropout_prob": args.dropout,
                "classifier_dropout": args.dropout,
                "hidden_dropout_prob": args.dropout,
            }
        )

    class SaveAndEvaluateLastStepCallback(TrainerCallback):
        """Ensure final step triggers eval/save so Ray always has a checkpoint."""

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            if state.global_step == state.max_steps:
                control.should_evaluate = True
                control.should_save = True

    trainer = Trainer(
        args=training_args,
        model_init=model_init if do_hpo else None,
        model=None if do_hpo else model,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[SaveAndEvaluateLastStepCallback],
    )

    if do_hpo:
        from ray.tune.search.hyperopt import HyperOptSearch
        from ray.tune.schedulers import ASHAScheduler
        from ray.tune import CheckpointConfig
        search_alg = HyperOptSearch(
            metric="eval_" + args.metrics, mode=args.direction[0]
        )
        scheduler = ASHAScheduler(
            metric="eval_" + args.metrics,
            mode=args.direction[0],
            reduction_factor=args.reduction_factor,
            grace_period=args.grace_period,
            max_t=args.max_t,
        )
        from ray import tune

        ray_results_dir = os.environ.get("RAY_RESULTS_DIR")
        if ray_results_dir:
            os.makedirs(ray_results_dir, exist_ok=True)

        class CleanupCallback(tune.Callback):
            def on_trial_complete(self, iteration, trials, trial, **info):
                trials_with_metric = [
                    t
                    for t in trials
                    if "eval_" + args.metrics in t.metric_analysis.keys()
                ]
                if not trials_with_metric:
                    return
                if args.direction[0] == "max":
                    trials_current_best = max(
                        t.metric_analysis["eval_" + args.metrics][args.direction[0]]
                        for t in trials_with_metric
                    )
                else:
                    trials_current_best = min(
                        t.metric_analysis["eval_" + args.metrics][args.direction[0]]
                        for t in trials_with_metric
                    )
                for t in trials:
                    if t.status == "TERMINATED":
                        if (
                            trials_current_best
                            < t.metric_analysis["eval_" + args.metrics][
                                args.direction[0]
                            ]
                            if args.direction[0] == "min"
                            else trials_current_best
                            > t.metric_analysis["eval_" + args.metrics][
                                args.direction[0]
                            ]
                        ):
                            self.cleanup_trial(t)
                        else:
                            print(
                                f"Current best: {t.trial_id} with eval_{args.metrics}: {t.metric_analysis['eval_' + args.metrics][args.direction[0]]} at iteration {t.last_result.get('training_iteration')}"
                            )

            def cleanup_trial(self, trial):
                if os.path.exists(trial.path):
                    checkpoint_dir = [
                        d for d in os.listdir(trial.path) if "checkpoint" in d
                    ]
                    for d in checkpoint_dir:
                        shutil.rmtree(trial.path + "/" + d)

        best_trial = trainer.hyperparameter_search(
            direction=args.direction[1],
            backend="ray",
            search_alg=search_alg,
            scheduler=scheduler,
            hp_space=hp_space,
            resources_per_trial={"cpu": 22, "gpu": 1},
            n_trials=args.n_trials,
            checkpoint_config=CheckpointConfig(
                checkpoint_score_attribute="eval_" + args.metrics,
                num_to_keep=1,
                checkpoint_score_order=args.direction[0],
            ),
            storage_path=ray_results_dir,
            callbacks=[CleanupCallback()],
        )
        best_trial_number = best_trial.run_summary.get_best_trial(
            mode=args.direction[0], metric="eval_" + args.metrics, scope="all"
        )
        best_result_series = best_trial.run_summary.trial_dataframes[
            best_trial_number.trial_id
        ]["eval_" + args.metrics]
        best_result = (
            best_result_series.max()
            if args.direction[0] == "max"
            else best_result_series.min()
        )
        best_result_id = (
            best_trial.run_summary.trial_dataframes[best_trial_number.trial_id][
                "eval_" + args.metrics
            ]
            == best_result
        )
        best_iteration = best_trial.run_summary.trial_dataframes[
            best_trial_number.trial_id
        ]["training_iteration"][best_result_id].values[0]
        best_checkpoint = best_trial.run_summary.get_best_checkpoint(
            best_trial_number, mode=args.direction[0], metric="eval_" + args.metrics
        )
        logging.info("***** Save the best model *****")
        if best_checkpoint is None or best_checkpoint.path is None:
            raise RuntimeError(
                "Ray Tune did not return a best checkpoint. Ensure checkpoints are produced (final-step save callback) and the metric matches scheduler."
            )
        best_checkpoint_path = glob.glob(
            best_checkpoint.path + "/checkpoint-*", recursive=True
        )[0]
        shutil.copytree(
            best_checkpoint_path, f"{args.output_dir}/{output_name}_best_model"
        )
        shutil.rmtree(f"{args.output_dir}/{output_name}")
        shutil.rmtree("/".join(best_checkpoint.path.split("/")[:-2]))
        ray.shutdown()
        print(
            f"Current best: {best_trial_number.trial_id} with eval_{args.metrics}: {best_result} at iteration {best_iteration}"
        )
    else:
        trainer.train()
        trainer.save_model(f"{args.output_dir}/{output_name}_best_model")
        shutil.rmtree(f"{args.output_dir}/{output_name}")

    logging.info("***** Load the best model *****")
    trainer.model = AutoModelForSequenceClassification.from_pretrained(
        f"{args.output_dir}/{output_name}_best_model"
    ).to(trainer.args.device)

    logging.info("***** Starting Evaluation *****")
    predictions, labels, _ = trainer.predict(dataset_test)
    y_pred = toLogits(predictions, THRESHOLD_VALUE)
    report = classification_report(labels, y_pred, target_names=labels_list, output_dict=True, zero_division=0)
    print(report)

    def np_encoder(object):
        if isinstance(object, np.generic):
            return object.item()

    with open(f"../runs/{output_name}_hpo.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": f"{args.output_dir}/{output_name}_best_model",
                "metrics": report,
                "hpo_settings": vars(args),
                "best_hp_trial": best_hp_trial
                if not do_hpo
                else best_trial.hyperparameters,
                "predictions": {
                    "identifiers": dataset_test_ids,
                "real_labels": labels.tolist(),
                "system_predictions": y_pred.tolist(),
                },
            },
            f,
            ensure_ascii=False,
            indent=4,
            default=np_encoder,
        )


if __name__ == "__main__":
    main()


