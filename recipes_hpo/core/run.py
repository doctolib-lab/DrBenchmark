import importlib
import os
import pathlib
import sys

# PYTHONPATH too: Ray workers are separate processes and do not inherit sys.path
_PACKAGE_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, _PACKAGE_ROOT)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [_PACKAGE_ROOT] + [p for p in [os.environ.get("PYTHONPATH")] if p]
)

from core import pipeline
from core.config import parse_args

# families migrated to core/ so far
IMPLEMENTED = {"token_classification": "core.families.token_classification"}


def main():
    cfg = parse_args()
    if cfg.task_type not in IMPLEMENTED:
        raise SystemExit(f"{cfg.task_type} is not migrated to core/ yet")
    pipeline.run(cfg, importlib.import_module(IMPLEMENTED[cfg.task_type]))


if __name__ == "__main__":
    main()
