import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, default_data_collator

from core import data
from core.labels import label_names


def prepare_datasets(cfg, tokenizer):
    text = _text(cfg, tokenizer)
    splits = data.load_splits(cfg, text)
    label_list = label_names(splits["train"].features[cfg.label_column])
    train, val, test = data.dedup(splits, cfg, text, cfg.label_column, None)

    def tokenize(example):
        encoded = tokenizer(
            text(example),
            truncation=True,
            max_length=cfg.max_position_embeddings,
            padding="max_length",
        )
        encoded["label"] = example[cfg.label_column]
        return encoded

    train = train.map(tokenize, keep_in_memory=True)
    for _ in range(3):
        train = train.shuffle(seed=cfg.seed)
    if cfg.fewshot != 1.0:
        train = train.select(range(int(len(train) * cfg.fewshot)))
    val = val.map(tokenize, keep_in_memory=True)
    test = test.map(tokenize, keep_in_memory=True)
    return train, val, test, label_list


def build_collator(cfg, tokenizer):
    return default_data_collator


def build_model(cfg, label_list, dropout=None, path=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        path or cfg.model_path, num_labels=len(label_list)
    )
    model.config.label2id = {name: str(i) for i, name in enumerate(label_list)}
    model.config.id2label = {str(i): name for i, name in enumerate(label_list)}
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
        # no target_names: naming the classes would add the absent ones and move macro avg
        metrics = classification_report(labels, preds, output_dict=True, zero_division=0)
        decode = lambda ids: [label_list[i] for i in ids]
        return metrics, decode(preds), decode(labels)

    return compute_metrics, report


def _text(cfg, tokenizer):
    """The sentence fed to the model, and what makes two examples duplicates."""
    separator = (
        f" {tokenizer.sep_token} "  # spaced, or a word-piece tokenizer glues it to the text
        if cfg.text_separator == "sep_token"
        else cfg.text_separator
    )

    def build(example):
        parts = [example[column] for column in cfg.text_columns]
        return separator.join(
            " ".join(part) if isinstance(part, list) else part for part in parts
        )

    return build
