#!/usr/bin/env python
"""Deprecated argument adapter for :mod:`scripts.simulation.synthetic_pipeline`."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_OWNED_OPTIONS = {"--stages", "--output-root"}
_AMBIGUOUS_OPTIONS = {"--n-images"}
_UNSUPPORTED_OPTIONS = {"--visualize"}
_RENAMED_OPTIONS = {
    "--n-photons": "--photons-per-pattern",
    "--buffer": "--scan-buffer",
}


def _option_name(token: str) -> str:
    return token.split("=", 1)[0]


def translate_legacy_arguments(
    argv: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    parser = argparse.ArgumentParser(
        description=("Deprecated: use ptycho_synthetic --stages simulate directly."),
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-size", type=int)
    parser.add_argument("--simulation-config", type=Path)
    args, passthrough = parser.parse_known_args(argv)

    if hasattr(args, "simulation_config"):
        raise ValueError(
            "legacy --simulation-config is not the generic workflow schema; "
            "migrate it to ptycho_synthetic --config"
        )
    translated_passthrough = []
    for token in passthrough:
        option = _option_name(token)
        if option in _AMBIGUOUS_OPTIONS:
            raise ValueError(
                "legacy --n-images is ambiguous between train/test raw patterns; "
                "use --train-patterns and --test-patterns"
            )
        if option in _OWNED_OPTIONS:
            raise ValueError(
                f"{option} is owned by the compatibility adapter; invoke "
                "ptycho_synthetic directly for custom stage/output selection"
            )
        if option in _UNSUPPORTED_OPTIONS:
            raise ValueError(
                f"legacy {option} has no equivalent in ptycho_synthetic; "
                "render diagnostics after simulation instead"
            )
        renamed = _RENAMED_OPTIONS.get(option)
        if renamed is not None:
            suffix = token[len(option) :]
            translated_passthrough.append(renamed + suffix)
        else:
            translated_passthrough.append(token)

    translated = [
        "--stages",
        "simulate",
        "--output-root",
        str(args.output_dir),
        "--N",
        str(getattr(args, "probe_size", 64)),
    ]
    translated.extend(translated_passthrough)
    return translated


def _run_synthetic_main(argv: list[str]) -> int:
    from scripts.simulation.synthetic_pipeline import main

    return main(argv)


def main(argv: list[str] | tuple[str, ...] | None = None) -> int:
    warnings.warn(
        "run_with_synthetic_lines.py is deprecated; use "
        "ptycho_synthetic --stages simulate",
        DeprecationWarning,
        stacklevel=2,
    )
    return _run_synthetic_main(translate_legacy_arguments(argv))


if __name__ == "__main__":
    raise SystemExit(main())
