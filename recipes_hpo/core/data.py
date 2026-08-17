import importlib.util

from datasets import (
    ClassLabel,
    DatasetDict,
    DownloadManager,
    Features,
    Sequence,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

from core.paths import ROOT

MERGE_SHUFFLE_SEED = 42
MERGE_SHARDS = 5


def label_names(feature):
    """Loaders declare labels as [ClassLabel], Sequence(ClassLabel) or ClassLabel."""
    if isinstance(feature, list):
        return list(feature[0].names)
    if hasattr(feature, "feature"):
        return list(feature.feature.names)
    return list(feature.names)


def text_builder(cfg, tokenizer):
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


def load_splits(cfg, key=None):
    if cfg.merge_subsets:
        return _merge(cfg, key)
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
    return build_dataset(cfg, subset)


def build_dataset(cfg, subset):
    """A loader exposing build() is called directly; the others go through load_dataset()."""
    path = ROOT / cfg.hf_loader
    module = _import(path)

    if not hasattr(module, "build"):
        return load_dataset(
            str(path), name=subset, data_dir=str(cfg.data_dir), trust_remote_code=True
        )

    # invariant 4, checked where the module is already imported so legacy loaders pay nothing
    if subset not in module.SUBSETS:
        raise SystemExit(
            f"config error: subset {subset!r} not declared by {path.name}; "
            f"SUBSETS = {module.SUBSETS}"
        )

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return module.build(subset, cfg.data_dir, DownloadManager(dataset_name=path.stem))


def _import(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _merge(cfg, key):
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
            _keep(cfg, key, cfg.label_column, negative_id, None),
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


def _keep(cfg, key, label_column, negative_id, seen):
    """seen=None keeps duplicates; the filters themselves always apply."""
    rules = cfg.dedup

    def _filter(example):
        value = key(example)
        if rules["drop_empty"] and not value:
            return False
        if seen is not None:
            if value in seen:
                return False
            seen.add(value)
        if rules["drop_all_negative"] and negative_id is not None:
            if all(tag == negative_id for tag in example[label_column]):
                return False
        return True

    return _filter


def dedup(splits, cfg, key, label_column, negative_id):
    """`key` maps an example to what makes it a duplicate; dedup.scope controls the scope."""
    rules = cfg.dedup

    def apply(split, seen):
        return splits[split].filter(
            _keep(cfg, key, label_column, negative_id, seen),
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
