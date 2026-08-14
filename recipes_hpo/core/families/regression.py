import numpy as np
from scipy import stats
from sklearn.metrics import root_mean_squared_error
from transformers import AutoModelForSequenceClassification

from core import data

SCORE_MIN, SCORE_MAX = 0.0, 5.0

def prepare_datasets(cfg, tokenizer):
    first, second = cfg.text_columns
    # unordered: the collator feeds the pair in either order, so (a, b) is (b, a)
    key = lambda example: tuple(sorted((example[first], example[second])))
    splits = data.load_splits(cfg, key)
    train, val, test = data.dedup(splits, cfg, key, cfg.label_column, None)

    def tokenize(example):
        return {
            # the pair rides inside input_ids so that Trainer's column pruning keeps it
            "input_ids": [
                tokenizer.encode(example[first], truncation=True, add_special_tokens=False),
                tokenizer.encode(example[second], truncation=True, add_special_tokens=False),
            ],
            "labels": float(example[cfg.label_column]),
        }

    train = train.map(tokenize, keep_in_memory=True)
    for _ in range(3):
        train = train.shuffle(seed=cfg.seed)
    if cfg.fewshot != 1.0:
        train = train.select(range(int(len(train) * cfg.fewshot)))
    val = val.map(tokenize, keep_in_memory=True)
    test = test.map(tokenize, keep_in_memory=True)
    # one continuous target, named for the record: the head has a single output
    return train, val, test, [cfg.label_column]


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
            first, second = feature["input_ids"]
            if len(first) + len(second) + 3 > limit:
                first = first[: limit // 2 - 2]
                second = second[: limit // 2 - 1]
            # order swapped at random on every split
            if np.random.rand() >= 0.5:
                first, second = second, first
            ids = (
                [tokenizer.cls_token_id]
                + first
                + [tokenizer.sep_token_id]
                + second
                + [eos_id]
            )
            batch.append(
                {
                    "input_ids": ids,
                    "attention_mask": [1] * len(ids),
                    "labels": feature["labels"],
                }
            )
        return tokenizer.pad(batch, padding=True, return_tensors="pt")

    return collate


def build_model(cfg, label_list, dropout=None, path=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        path or cfg.model_path, num_labels=1
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
        return {
            "rmse": root_mean_squared_error(
                eval_prediction.label_ids, eval_prediction.predictions
            )
        }

    def report(predictions, labels):
        preds = np.asarray(predictions, dtype=float).flatten()
        refs = np.asarray(labels, dtype=float).flatten()
        coefficient, p_value = stats.spearmanr(refs, preds)
        metrics = {
            "EDRM": _edrm(refs, preds),
            "RMSE": float(root_mean_squared_error(refs, preds)),
            "spearman_correlation_coef": float(coefficient),
            "spearman_correlation_p": float(p_value),
        }
        return metrics, preds.tolist(), refs.tolist()

    return compute_metrics, report


def _edrm(refs, preds):
    """Distance to the reference, normalised per example by the worst one it allows."""
    bounded = np.clip(preds, SCORE_MIN, SCORE_MAX)
    worst = np.maximum(np.abs(SCORE_MIN - refs), np.abs(SCORE_MAX - refs))
    return float(np.mean(1 - np.abs(refs - bounded) / worst))
