import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification

from core import data


def prepare_datasets(cfg, tokenizer):
    source, *candidates = cfg.text_columns
    key = lambda example: (
        example[source],
        tuple(sorted(example[column] for column in candidates)),
    )
    splits = data.load_splits(cfg, key)
    label_list = data.label_names(splits["train"].features[cfg.label_column])
    if len(label_list) != len(candidates):
        raise SystemExit(
            f"{len(candidates)} candidate columns for {len(label_list)} labels: "
            "text_columns must name the source then one column per label, in label order"
        )
    train, val, test = data.dedup(splits, cfg, key, cfg.label_column, None)

    def tokenize(example):
        encode = lambda text: tokenizer.encode(
            text, truncation=True, add_special_tokens=False
        )
        return {
            # source first, then the candidates in label order; the collator shuffles them
            "input_ids": [encode(example[source])]
            + [encode(example[column]) for column in candidates],
            "labels": example[cfg.label_column],
        }

    train = train.map(tokenize, keep_in_memory=True)
    for _ in range(3):
        train = train.shuffle(seed=cfg.seed)
    if cfg.fewshot != 1.0:
        train = train.select(range(int(len(train) * cfg.fewshot)))
    val = val.map(tokenize, keep_in_memory=True)
    test = test.map(tokenize, keep_in_memory=True)
    return train, val, test, label_list


def build_collator(cfg, tokenizer):
    limit = cfg.max_position_embeddings
    eos_id = (
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else tokenizer.sep_token_id
    )

    def collate(features):
        batch = []
        for feature in features:
            source, *candidates = feature["input_ids"]
            specials = len(candidates) + 2  # cls, one sep per candidate, eos
            if len(source) + sum(len(c) for c in candidates) + specials > limit:
                # an equal share each, so the assembled sequence cannot exceed the limit
                budget = (limit - specials) // (len(candidates) + 1)
                source = source[:budget]
                candidates = [c[:budget] for c in candidates]
            # the label is a position in this presentation, not a candidate identity
            order = np.random.permutation(len(candidates))
            ids = [tokenizer.cls_token_id] + source
            for position in order:
                ids += [tokenizer.sep_token_id] + candidates[position]
            ids.append(eos_id)
            batch.append(
                {
                    "input_ids": ids,
                    "attention_mask": [1] * len(ids),
                    "labels": list(order).index(feature["labels"]),
                }
            )
        return tokenizer.pad(batch, padding=True, return_tensors="pt")

    return collate


def build_model(cfg, label_list, dropout=None, path=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        path or cfg.model_path, num_labels=len(label_list)
    )
    if dropout is not None:
        model.config.update(
            {
                "attention_probs_dropout_prob": dropout,
                "classifier_dropout": dropout,
                "hidden_dropout_prob": dropout,
            }
        )
    return model


def build_metrics(cfg, label_list, experiment_id):
    def compute_metrics(eval_prediction):
        labels = eval_prediction.label_ids
        preds = eval_prediction.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def report(predictions, labels):
        preds = np.argmax(predictions, axis=1)
        metrics = classification_report(labels, preds, output_dict=True, zero_division=0)
        # exactly one candidate is correct, so both reduce to the accuracy §12 reads here
        metrics["exact_match"] = accuracy_score(labels, preds)
        metrics["hamming_score"] = metrics["exact_match"]
        # positions, left as such: the presentation order they index is not recorded
        return metrics, preds.tolist(), labels.tolist()

    return compute_metrics, report
