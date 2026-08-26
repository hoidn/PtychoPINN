#!/usr/bin/env python
"""Collate per-arm reconstruction metrics of a study into a TSV table.

Usage:
    python scripts/studies/collate_study_metrics.py <study_root> \\
        [--metrics reconstruction/metrics.json]

Walks <study_root>/*/<metrics>, keeps numeric scalars (nested dicts flattened
to dotted keys), prints one TSV row per arm.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _scalars(data: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[prefix + key] = value
        elif isinstance(value, dict):
            out.update(_scalars(value, prefix + key + "."))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_root", type=Path)
    parser.add_argument("--metrics", default="reconstruction/metrics.json")
    args = parser.parse_args(argv)

    rows: list[tuple[str, dict[str, float]]] = []
    for metrics_path in sorted(args.study_root.glob(f"*/{args.metrics}")):
        arm = metrics_path.relative_to(args.study_root).parts[0]
        rows.append((arm, _scalars(json.loads(metrics_path.read_text()))))
    if not rows:
        print(f"no {args.metrics} under {args.study_root}", file=sys.stderr)
        return 1

    keys = sorted({key for _, values in rows for key in values})
    print("\t".join(["arm", *keys]))
    for arm, values in rows:
        cells = [f"{values[k]:.6g}" if k in values else "" for k in keys]
        print("\t".join([arm, *cells]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
