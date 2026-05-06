#!/usr/bin/env python3
"""CLI entrypoint for the natural-patch expanded CDI benchmark harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.studies.cdi_natural_patch_benchmark import (
    DEFAULT_SEED,
    NATURAL_PATCH_ROW_ROSTER,
    run_natural_patch_benchmark,
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the natural-patch expanded CDI benchmark.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--item-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["dry-run", "benchmark", "recollate"], required=True)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--rows",
        type=str,
        default=",".join(NATURAL_PATCH_ROW_ROSTER),
        help="Comma-separated natural-patch row roster.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    rows = tuple(row.strip() for row in args.rows.split(",") if row.strip())
    result = run_natural_patch_benchmark(
        dataset_root=args.dataset_root,
        item_root=args.item_root,
        mode=args.mode,
        rows=rows,
        seed=args.seed,
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
