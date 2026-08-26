"""CUDA-hidden child process for deterministic synthetic simulation."""

from __future__ import annotations

import os

# This assignment must precede every TensorFlow-reachable project import.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
from pathlib import Path
from typing import Any


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate canonical flat synthetic acquisitions on CPU."
    )
    parser.add_argument(
        "--request-json",
        type=Path,
        required=True,
        help=(
            "JSON request containing profile plus optional file_values and "
            "cli_values accepted by resolve_synthetic_workflow"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def _load_request(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load simulation request {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("simulation request must be a JSON object")
    unknown = set(payload) - {"profile", "file_values", "cli_values"}
    if unknown:
        raise ValueError(f"unknown simulation request fields: {sorted(unknown)!r}")
    return payload


def run_worker(request_path: str | Path, output_root: str | Path):
    """Resolve a Task 1 workflow request and publish its dataset artifacts."""

    from ptycho.simulation.flat_acquisition import generate_flat_acquisitions
    from ptycho.workflows.synthetic_config import resolve_synthetic_workflow

    request = _load_request(Path(request_path))
    resolved = resolve_synthetic_workflow(
        profile=request.get("profile", "hybrid-resnet-lines"),
        file_values=request.get("file_values"),
        cli_values=request.get("cli_values"),
    )
    return generate_flat_acquisitions(resolved, Path(output_root) / "datasets")


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    result = run_worker(args.request_json, args.output_root)
    print(
        json.dumps(
            {
                "manifest": str(result.manifest_path),
                "train": str(result.train_path),
                "test": str(result.test_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
