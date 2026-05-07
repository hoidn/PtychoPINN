"""Fast smoke check for BRDT sinogram-input adapters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from scripts.studies.born_rytov_dt import train as train_mod
from scripts.studies.born_rytov_dt.run_sinogram_input_40ep import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
)


SCRIPT_PATH = "scripts/studies/born_rytov_dt/run_sinogram_input_smoke.py"


def run_sinogram_input_smoke(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_root: Path = DEFAULT_OUTPUT_ROOT / "smoke",
    device_choice: str = "auto",
) -> Dict[str, object]:
    output_root = Path(output_root).resolve()
    rows: Dict[str, object] = {}
    for row_id in ("ffno", "sru_net"):
        rows[row_id] = train_mod.run_training(
            architecture=row_id,
            manifest_path=Path(manifest_path),
            output_root=output_root / row_id,
            epochs=1,
            batch_size=2,
            learning_rate=2e-4,
            device_choice=device_choice,
            fast_dev_run=True,
            in_channels=2,
            hybrid_label="sru_net",
            input_mode="sinogram",
        )
    return {
        "schema_version": "brdt_sinogram_input_smoke_v1",
        "manifest_path": str(Path(manifest_path).resolve()),
        "output_root": str(output_root),
        "rows": rows,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brdt_sinogram_input_smoke")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "smoke")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = run_sinogram_input_smoke(
        manifest_path=args.manifest,
        output_root=args.output_root,
        device_choice=str(args.device),
    )
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
