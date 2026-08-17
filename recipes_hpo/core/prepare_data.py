"""Materialise a recipe's dataset into its own data/ directory, for offline runs."""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


from core import data
from core.config import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load(args.config, "none", 42)

    for subset in cfg.merge_subsets or [cfg.subset]:
        target = cfg.data_dir / f"local_hf_{subset}"
        dataset = data.build_dataset(cfg, subset)
        dataset.save_to_disk(str(target))
        print({split: len(rows) for split, rows in dataset.items()})
        print(f"saved {target}")


if __name__ == "__main__":
    main()
