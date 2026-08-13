import evaluate
import numpy as np
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

from core import data
from core.labels import label_names
from core.paths import ROOT

TOKEN_COLUMN = "tokens"


def prepare_datasets(cfg, tokenizer):
    splits = data.load_splits(cfg, TOKEN_COLUMN)
    label_list = label_names(splits["train"].features[cfg.label_column])
    negative_id = label_list.index("O") if "O" in label_list else None
    train, val, test = data.dedup(
        splits, cfg, TOKEN_COLUMN, cfg.label_column, negative_id
    )

    align = _align_slow if cfg.slow_tokenizer else _align_fast

    def tokenize(examples):
        return align(examples, tokenizer, cfg)

    train = train.map(tokenize, batched=True, keep_in_memory=True)
    for _ in range(3):
        train = train.shuffle(seed=cfg.seed)
    if cfg.fewshot != 1.0:
        train = train.select(range(int(len(train) * cfg.fewshot)))
    val = val.map(tokenize, batched=True, keep_in_memory=True)
    test = test.map(tokenize, batched=True, keep_in_memory=True)
    return train, val, test, label_list


def build_collator(cfg, tokenizer):
    return DataCollatorForTokenClassification(tokenizer)


def build_model(cfg, label_list, dropout=None, path=None):
    model = AutoModelForTokenClassification.from_pretrained(
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
    metric = evaluate.load(
        str(ROOT / "metrics" / "seqeval.py"), experiment_id=experiment_id
    )

    def decode(predictions, labels):
        predictions = np.argmax(predictions, axis=2)
        preds = [
            [label_list[p] for p, l in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        refs = [
            [label_list[l] for p, l in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        return preds, refs

    def compute_metrics(eval_prediction):
        preds, refs = decode(eval_prediction[0], eval_prediction[1])
        results = metric.compute(predictions=preds, references=refs)
        per_entity = [
            results[k]["f1"] for k in results if not k.startswith("overall_")
        ]
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
            "macro_f1": sum(per_entity) / len(per_entity),
        }

    def report(predictions, labels):
        preds, refs = decode(predictions, labels)
        return metric.compute(predictions=preds, references=refs), preds, refs

    return compute_metrics, report


def _align_fast(examples, tokenizer, cfg):
    tokenized = tokenizer(
        list(examples[TOKEN_COLUMN]),
        truncation=True,
        max_length=cfg.max_position_embeddings,
        padding="max_length",
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[cfg.label_column]):
        ids, previous = [], None
        for word_idx in tokenized.word_ids(batch_index=i):
            if word_idx is None or word_idx == previous:
                ids.append(-100)
            else:
                ids.append(label[word_idx])
            previous = word_idx
        labels.append(ids)
    tokenized["labels"] = labels
    return tokenized


def _align_slow(examples, tokenizer, cfg):
    """Manual alignment for tokenizers without word_ids() support (e.g. FlauBERT)."""
    limit = cfg.max_position_embeddings
    bos = tokenizer("<s>")["input_ids"][1]
    eos = tokenizer("</s>")["input_ids"][1]
    pad = tokenizer("<pad>")["input_ids"][1]

    all_ids, all_labels = [], []
    for tokens, label in zip(examples[TOKEN_COLUMN], examples[cfg.label_column]):
        ids, labels = [bos], [-100]
        for token, tag in zip(tokens, label):
            pieces = tokenizer(token)["input_ids"][1:-1]
            ids.extend(pieces)
            labels.extend([tag] * len(pieces))
        ids, labels = ids[: limit - 1], labels[: limit - 1]
        ids.append(eos)
        labels.append(-100)
        padding = limit - len(ids)
        if padding > 0:
            ids.extend([pad] * padding)
            labels.extend([-100] * padding)
        all_ids.append(ids)
        all_labels.append(labels)
    return {"input_ids": all_ids, "labels": all_labels}
