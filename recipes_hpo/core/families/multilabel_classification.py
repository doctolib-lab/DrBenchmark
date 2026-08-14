from datasets import Sequence, Value
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding

from core import data
from core.labels import label_names


def prepare_datasets(cfg, tokenizer):
    text = data.text_builder(cfg, tokenizer)
    splits = data.load_splits(cfg, text)
    label_list = label_names(splits["train"].features[cfg.label_column])
    train, val, test = data.dedup(splits, cfg, text, cfg.label_column, None)

    def tokenize(example):
        encoded = tokenizer(
            text(example),
            truncation=True,
            max_length=cfg.max_position_embeddings,
        )
        
        multi_hot = [0.0] * len(label_list)
        for label_id in example[cfg.label_column]:
            multi_hot[label_id] = 1.0
        encoded["labels"] = multi_hot
        return encoded

    def encode(split):
        # float32: BCEWithLogitsLoss rejects the float64 that map() infers here
        return split.map(tokenize, keep_in_memory=True).cast_column(
            "labels", Sequence(Value("float32"))
        )

    train = encode(train)
    for _ in range(3):
        train = train.shuffle(seed=cfg.seed)
    if cfg.fewshot != 1.0:
        train = train.select(range(int(len(train) * cfg.fewshot)))
    return train, encode(val), encode(test), label_list


def build_collator(cfg, tokenizer):
    return DataCollatorWithPadding(tokenizer)


def build_model(cfg, label_list, dropout=None, path=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        path or cfg.model_path,
        num_labels=len(label_list),
        problem_type="multi_label_classification",
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
    def binarize(logits):
        return (expit(logits) >= cfg.threshold).astype(float)

    def compute_metrics(eval_prediction):
        labels = eval_prediction.label_ids
        preds = binarize(eval_prediction.predictions)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
            "roc_auc": _roc_auc(labels, preds),
        }

    def report(predictions, labels):
        preds = binarize(predictions)
        # target_names is safe here: an indicator matrix carries every label as a column
        metrics = classification_report(
            labels, preds, target_names=label_list, output_dict=True, zero_division=0
        )
        decode = lambda matrix: [
            [label_list[i] for i, flag in enumerate(row) if flag] for row in matrix
        ]
        return metrics, decode(preds), decode(labels)

    return compute_metrics, report


def _roc_auc(labels, preds):
    """A label nothing predicts leaves its column constant, which roc_auc rejects."""
    try:
        return roc_auc_score(labels, preds, average="micro")
    except ValueError:
        return 0.0
