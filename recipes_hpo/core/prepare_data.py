"""Materialise a recipe's dataset into its own data/ directory, for offline runs."""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


from datasets import load_dataset

from core.config import load
from core.paths import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load(args.config, "none", 42)
    target = cfg.data_dir / f"local_hf_{cfg.subset}"
    dataset = load_dataset(
        str(ROOT / cfg.hf_loader),
        name=cfg.subset,
        data_dir=str(cfg.data_dir),
        trust_remote_code=True,
    )
    dataset.save_to_disk(str(target))
    print({split: len(rows) for split, rows in dataset.items()})
    print(f"saved {target}")


if __name__ == "__main__":
    main()
