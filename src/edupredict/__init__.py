"""
EduPredict package.

Provides a CLI entry-point ``edupredict`` (configured in pyproject.toml)
and re-exports the main sub-module functions for programmatic use.

Quick-start (CLI):
    edupredict preprocess   # clean raw data → data/processed/
    edupredict train        # train models   → model/
    edupredict              # print this help

Programmatic use:
    from edupredict.preprocessing import run as preprocess
    from edupredict.training import run as train_models
"""

from __future__ import annotations

import sys


def main() -> None:
    """CLI entry-point: dispatch sub-commands."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "preprocess":
        from edupredict.preprocessing import run as _run
        _run()

    elif cmd == "train":
        from edupredict.training import run as _run
        _run()

    elif cmd in ("help", "--help", "-h"):
        print(__doc__)

    else:
        print(f"Unknown command: '{cmd}'. Use 'preprocess' or 'train'.")
        sys.exit(1)
