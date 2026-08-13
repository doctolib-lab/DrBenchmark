import os
import subprocess
from pathlib import Path


def _root():
    env = os.environ.get("DRB_ROOT")
    if env:
        return Path(env).resolve()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(out.stdout.strip())
    except Exception:
        return Path(__file__).resolve().parents[2]


ROOT = _root()
