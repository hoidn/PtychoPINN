"""Fast smoke check for BRDT sinogram-input adapters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.studies.born_rytov_dt import train as train_mod
from scripts.studies.born_rytov_dt.run_sinogram_input_40ep import DEFAULT_MANIFEST
from scripts.studies.invocation_logging import write_invocation_artifacts


SCRIPT_PATH = "scripts/studies/born_rytov_dt/run_sinogram_input_smoke.py"
BACKLOG_ITEM = "2026-05-07-brdt-sinogram-input-adapter-contract"
DEFAULT_OUTPUT_ROOT = Path(
    ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog"
) / BACKLOG_ITEM / "smoke"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary_path(output_root: Path) -> Path:
    return output_root.parent / "smoke_summary.json" if output_root.name == "smoke" else output_root / "smoke_summary.json"


def _build_row_args(
    *,
    row_id: str,
    manifest_path: Path,
    output_root: Path,
    device_choice: str,
) -> List[str]:
    return [
        "--architecture",
        row_id,
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(output_root),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--learning-rate",
        "2e-4",
        "--device",
        str(device_choice),
        "--fast-dev-run",
        "--in-channels",
        "2",
        "--hybrid-label",
        "sru_net",
        "--input-mode",
        "sinogram",
    ]


def run_sinogram_input_smoke(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_root: Path = DEFAULT_OUTPUT_ROOT / "smoke",
    device_choice: str = "auto",
) -> Dict[str, object]:
    output_root = Path(output_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    rows: Dict[str, object] = {}
    row_statuses: Dict[str, str] = {}
    row_artifacts: Dict[str, Dict[str, str]] = {}
    for row_id in ("ffno", "sru_net"):
        row_output_root = output_root / row_id
        row_args = _build_row_args(
            row_id=row_id,
            manifest_path=manifest_path,
            output_root=row_output_root,
            device_choice=device_choice,
        )
        write_invocation_artifacts(
            row_output_root,
            train_mod.SCRIPT_PATH,
            row_args,
            {
                "architecture": row_id,
                "manifest": str(manifest_path),
                "output_root": str(row_output_root),
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 2e-4,
                "device": str(device_choice),
                "fast_dev_run": True,
                "in_channels": 2,
                "hybrid_label": "sru_net",
                "input_mode": "sinogram",
            },
            extra={"backlog_item": BACKLOG_ITEM, "execution_role": "smoke_row"},
        )
        rows[row_id] = train_mod.run_training(
            architecture=row_id,
            manifest_path=manifest_path,
            output_root=row_output_root,
            epochs=1,
            batch_size=2,
            learning_rate=2e-4,
            device_choice=device_choice,
            fast_dev_run=True,
            in_channels=2,
            hybrid_label="sru_net",
            input_mode="sinogram",
        )
        row_statuses[row_id] = str(rows[row_id]["row_status"])
        adapter_contract_path = Path(
            str(rows[row_id].get("adapter_contract_path", row_output_root / "adapter_contract.json"))
        )
        invocation_path = row_output_root / "invocation.json"
        invocation_sh_path = row_output_root / "invocation.sh"
        contract_payload = json.loads(adapter_contract_path.read_text(encoding="utf-8"))
        matching_row = next(
            (row for row in contract_payload.get("rows", []) if row.get("row_id") == row_id),
            None,
        )
        if matching_row is None:
            raise ValueError(f"adapter contract for {row_id} missing selected row payload")
        if matching_row.get("input_mode") != "sinogram":
            raise ValueError(
                f"adapter contract for {row_id} recorded input_mode={matching_row.get('input_mode')!r}"
            )
        row_artifacts[row_id] = {
            "adapter_contract_path": str(adapter_contract_path),
            "invocation_path": str(invocation_path),
            "invocation_sh_path": str(invocation_sh_path),
        }

    summary_path = _summary_path(output_root)
    summary_payload = {
        "schema_version": "brdt_sinogram_input_smoke_v2",
        "backlog_item": BACKLOG_ITEM,
        "dataset_manifest_path": str(manifest_path),
        "input_mode": "sinogram",
        "model_input_source": "measured complex sinogram",
        "born_consistency_target_source": "measured complex sinogram",
        "born_inverse_role": "non_learned_reference_only",
        "learned_rows": ["ffno", "sru_net"],
        "row_statuses": row_statuses,
        "row_artifacts": row_artifacts,
        "smoke_output_root": str(output_root),
    }
    _write_json(summary_path, summary_payload)
    return {
        "schema_version": "brdt_sinogram_input_smoke_v1",
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "rows": rows,
        "smoke_summary_path": str(summary_path),
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
