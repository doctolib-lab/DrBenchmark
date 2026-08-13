from datasets import (
    ClassLabel,
    DatasetDict,
    Features,
    Sequence,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

from core.labels import label_names
from core.paths import ROOT

MERGE_SHUFFLE_SEED = 42
MERGE_SHARDS = 5


def load_splits(cfg, token_column=None):
    if cfg.merge_subsets:
        return _merge(cfg, token_column)
    return _load(cfg, cfg.subset)


def _load(cfg, subset):
    if cfg.offline:
        local = cfg.data_dir / f"local_hf_{subset}"
        if not local.exists():
            raise SystemExit(
                f"missing {local}\n"
                f"run: python {ROOT}/recipes_hpo/core/prepare_data.py --config {cfg.config_path}"
            )
        return load_from_disk(str(local))
    return load_dataset(
        str(ROOT / cfg.hf_loader),
        name=subset,
        data_dir=str(cfg.data_dir),
        trust_remote_code=True,
    )


def _merge(cfg, token_column):
    """Pool subsets declaring divergent label orders, remapped onto their union."""
    pooled, label_list = [], []
    for subset in cfg.merge_subsets:
        splits = _load(cfg, subset)
        rows = concatenate_datasets(
            [splits[name] for name in ("train", "validation", "test")]
        )
        names = label_names(rows.features[cfg.label_column])
        negative_id = names.index("O") if "O" in names else None
        # filtered before the shard: this is what decides each shard's content
        rows = rows.filter(
            _keep(cfg, token_column, cfg.label_column, negative_id, None),
            load_from_cache_file=False,
        )
        for name in names:
            if name not in label_list:
                label_list.append(name)
        pooled.append((rows, names))

    features = Features(
        {
            **pooled[0][0].features,
            cfg.label_column: Sequence(ClassLabel(names=label_list)),
        }
    )
    merged = []
    for rows, names in pooled:
        remap = {i: label_list.index(name) for i, name in enumerate(names)}
        merged.append(
            rows.map(
                lambda example: {
                    cfg.label_column: [remap[tag] for tag in example[cfg.label_column]]
                },
                features=features,
                load_from_cache_file=False,
            )
        )

    rows = concatenate_datasets(merged).shuffle(seed=MERGE_SHUFFLE_SEED)
    shards = [rows.shard(num_shards=MERGE_SHARDS, index=i) for i in range(MERGE_SHARDS)]
    # official splits are dropped: per-subset, too small to survive the merge
    return DatasetDict(
        test=shards[0],
        validation=shards[1].shard(num_shards=2, index=1),
        train=concatenate_datasets(
            [shards[1].shard(num_shards=2, index=0)] + shards[2:]  # its other half
        ),
    )


def _keep(cfg, token_column, label_column, negative_id, seen):
    """seen=None keeps duplicates; the filters themselves always apply."""
    rules = cfg.dedup

    def _filter(example):
        if rules["drop_empty"] and len(example[token_column]) == 0:
            return False
        if seen is not None:
            key = tuple(example[token_column])
            if key in seen:
                return False
            seen.add(key)
        if rules["drop_all_negative"] and negative_id is not None:
            if all(tag == negative_id for tag in example[label_column]):
                return False
        return True

    return _filter


def dedup(splits, cfg, token_column, label_column, negative_id):
    """Filters always apply; dedup.scope controls duplicate removal across splits."""
    rules = cfg.dedup

    def apply(split, seen):
        return splits[split].filter(
            _keep(cfg, token_column, label_column, negative_id, seen),
            load_from_cache_file=False,
        )

    # test is always filtered first: the other splits are deduplicated against it
    if rules["scope"] == "cascade":
        shared = set()
        test = apply("test", shared)
        val = apply("validation", shared)
        train = apply("train", shared)
    elif rules["scope"] == "against_test":
        from_test = set()
        test = apply("test", from_test)
        train = apply("train", set(from_test))
        val = apply("validation", set(from_test))
    else:
        test = apply("test", None)
        val = apply("validation", None)
        train = apply("train", None)
    return train, val, test
