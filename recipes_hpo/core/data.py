from datasets import load_dataset, load_from_disk

from core.paths import ROOT


def load_splits(cfg):
    if cfg.offline:
        local = cfg.data_dir / f"local_hf_{cfg.subset}"
        if not local.exists():
            raise SystemExit(
                f"missing {local}\n"
                f"run: python {ROOT}/recipes_hpo/core/prepare_data.py --config {cfg.config_path}"
            )
        return load_from_disk(str(local))
    return load_dataset(
        str(ROOT / cfg.hf_loader),
        name=cfg.subset,
        data_dir=str(cfg.data_dir),
        trust_remote_code=True,
    )


def dedup(splits, cfg, token_column, label_column, negative_id):
    """Filters always apply; dedup.scope controls duplicate removal across splits."""
    rules = cfg.dedup

    def keep(seen):
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

    def apply(split, seen):
        return splits[split].filter(keep(seen), load_from_cache_file=False)

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
