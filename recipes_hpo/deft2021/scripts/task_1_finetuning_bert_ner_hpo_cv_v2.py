#!/usr/bin/env python3

import os, glob
import shutil
import uuid
import json
import logging

import evaluate
import numpy as np

from utils_hpo import parse_args

from datasets import load_dataset, load_from_disk, concatenate_datasets

from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

import ray
tmp_dir = os.environ.get("RAY_TMPDIR")
ray.init(_temp_dir=tmp_dir, include_dashboard=False) if tmp_dir else ray.init(include_dashboard=False)

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
            name=str(args.subset),
            data_dir=args.data_dir,
        )

    args.fold -= 1
    # Retrieve past best_hp_trial, if any:
    # search_pattern = "../runs/*_fold*.json"
    search_pattern = "../runs/DrBenchmark-DEFT2021-ner*_fold*.json"
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

    # Concatenate all the splits and shuffle
    dataset = concatenate_datasets(
        [dataset["train"], dataset["validation"], dataset["test"]]
    )
    dataset = dataset.shuffle(seed=42)

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
    # In order to get 10% of validation set, allocate half of validation to training
    dataset["train"] = concatenate_datasets(
        [dataset["validation"].shard(num_shards=2, index=0), dataset["train"]]
    )
    dataset["validation"] = dataset["validation"].shard(num_shards=2, index=1)

    label_list = dataset["train"].features["ner_tags"].feature.names
    label2id = {name: str(i) for i, name in enumerate(label_list)}
    id2label = {str(i): name for i, name in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_and_align_labels(examples):
        label_all_tokens = False
        tokenized_inputs = tokenizer(
            list(examples["tokens"]),
            truncation=True,
            max_length=args.max_position_embeddings,
            padding="max_length",
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            label_ids = []
            previous_word_idx = None
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    train_tokenized_datasets = (
        dataset["train"]
        .map(tokenize_and_align_labels, batched=True)
        .shuffle(seed=42)
        .shuffle(seed=42)
        .shuffle(seed=42)
    )
    if args.fewshot != 1.0:
        train_tokenized_datasets = train_tokenized_datasets.select(
            range(int(len(train_tokenized_datasets) * args.fewshot))
        )
    validation_tokenized_datasets = dataset["validation"].map(
        tokenize_and_align_labels, batched=True
    )
    test_tokenized_datasets = dataset["test"].map(
        tokenize_and_align_labels, batched=True
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = f"DrBenchmark-DEFT2021-ner-{uuid.uuid4()}_fold={args.fold}"

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
            model = AutoModelForTokenClassification.from_pretrained(
                absolute_path_to_model, num_labels=len(label_list)
            )
            model.config.label2id = label2id
            model.config.id2label = id2label
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
                "gradient_accumulation_steps": eval(args.gradient_accumulation_steps),
                "weight_decay": eval(args.weight_decay),
                "warmup_ratio": eval(args.warmup_ratio),
                "dropout": eval(args.dropout),
            }
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            absolute_path_to_model, num_labels=len(label_list)
        )
        model.config.label2id = label2id
        model.config.id2label = id2label
        model.config.update(
            {
                "attention_probs_dropout_prob": args.dropout,
                "classifier_dropout": args.dropout,
                "hidden_dropout_prob": args.dropout,
            }
        )

    metric = evaluate.load("../../../metrics/seqeval.py", experiment_id=output_name)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    def remove_dummy_label(predictions, labels):
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_predictions, true_labels

    def compute_metrics(p):
        predictions, labels = p
        true_predictions, true_labels = remove_dummy_label(predictions, labels)
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    from transformers import TrainerCallback, TrainerControl, TrainerState

    class SaveAndEvaluateLastStepCallback(TrainerCallback):
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

    if do_hpo:
        trainer = Trainer(
            args=training_args,
            model_init=model_init,
            train_dataset=train_tokenized_datasets,
            eval_dataset=validation_tokenized_datasets,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[SaveAndEvaluateLastStepCallback],
        )
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
                        metric_val = t.metric_analysis["eval_" + args.metrics][
                            args.direction[0]
                        ]
                        if (
                            (args.direction[0] == "min" and trials_current_best < metric_val)
                            or (args.direction[0] == "max" and trials_current_best > metric_val)
                        ):
                            self.cleanup_trial(t)
                        else:
                            print(
                                f"Current best: {t.trial_id} with eval_{args.metrics}: {metric_val} at iteration {t.last_result.get('training_iteration')}"
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
                "Ray Tune did not return a best checkpoint. Ensure checkpoints are produced and metric name matches scheduler."
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
        model = AutoModelForTokenClassification.from_pretrained(
            absolute_path_to_model, num_labels=len(label_list)
        )
        model.config.label2id = label2id
        model.config.id2label = id2label
        model.config.update(
            {
                "attention_probs_dropout_prob": args.dropout,
                "classifier_dropout": args.dropout,
                "hidden_dropout_prob": args.dropout,
            }
        )
        trainer = Trainer(
            args=training_args,
            model=model,
            train_dataset=train_tokenized_datasets,
            eval_dataset=validation_tokenized_datasets,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[SaveAndEvaluateLastStepCallback],
        )
        trainer.train()
        trainer.save_model(f"{args.output_dir}/{output_name}_best_model")
        shutil.rmtree(f"{args.output_dir}/{output_name}")

    logging.info("***** Load the best model *****")
    trainer.model = AutoModelForTokenClassification.from_pretrained(
        f"{args.output_dir}/{output_name}_best_model"
    ).to(trainer.args.device)

    logging.info("***** Starting Evaluation *****")
    predictions, labels, _ = trainer.predict(test_tokenized_datasets)
    true_predictions, true_labels = remove_dummy_label(predictions, labels)

    cr_metric = metric.compute(predictions=true_predictions, references=true_labels)
    print(cr_metric)

    def np_encoder(object):
        if isinstance(object, np.generic):
            return object.item()

    with open(f"../runs/{output_name}_hpo.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": f"{args.output_dir}/{output_name}_best_model",
                "metrics": cr_metric,
                "hpo_settings": vars(args),
                "best_hp_trial": best_hp_trial
                if not do_hpo
                else best_trial.hyperparameters,
                "predictions": {
                    "identifiers": list(dataset["test"]["id"]),
                    "real_labels": true_labels,
                    "system_predictions": true_predictions,
                },
            },
            f,
            ensure_ascii=False,
            indent=4,
            default=np_encoder,
        )


if __name__ == "__main__":
    main()


