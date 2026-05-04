"""One-shot backfill of U-NO row provenance for the lines128 table-extension bundle.

This script retroactively populates the per-row `invocation.json` and
`randomness_contract.json` files for the two fresh U-NO rows
(`pinn_neuralop_uno`, `supervised_neuralop_uno`) inside the existing extension
bundle so they record the full provenance contract required by the design
(`docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md`).

The script runs inside `ptycho311`. It imports the same provenance helpers that
the runner now uses, so the backfilled values come from the same canonical
source as future fresh runs. Fields that are not retroactively recoverable
(working-tree dirty state at the original launch time) are recorded as `null`
together with a `backfilled_at_utc` marker.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.studies.invocation_logging import (
    capture_neuralop_provenance,
    capture_runtime_provenance,
    update_invocation_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_ROOT = (
    REPO_ROOT
    / ".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog"
      "/2026-04-30-cdi-lines128-uno-table-extension/runs"
      "/complete_table_plus_uno_20260504T100347Z/runs"
)
ROW_DIRS = ["pinn_neuralop_uno", "supervised_neuralop_uno"]


def _backfill_invocation(row_dir: Path, runtime: dict, neuralop: dict) -> None:
    invocation_path = row_dir / "invocation.json"
    if not invocation_path.exists():
        raise SystemExit(f"missing invocation file: {invocation_path}")
    payload = json.loads(invocation_path.read_text())
    extra = payload.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}

    existing_runtime = extra.get("runtime_provenance", {}) or {}
    merged_runtime = dict(existing_runtime)
    for key, value in runtime.items():
        merged_runtime.setdefault(key, value)
    if "torch" not in merged_runtime:
        merged_runtime["torch"] = runtime.get("torch")

    backfilled_at = datetime.now(timezone.utc).isoformat()

    update_invocation_artifacts(
        invocation_path,
        extra={
            "runtime_provenance": merged_runtime,
            "neuralop_provenance": neuralop,
            "git_dirty": None,
            "provenance_backfilled_at_utc": backfilled_at,
            "provenance_backfill_reason": (
                "Initial run did not record python version, torch/CUDA/GPU, "
                "neuraloperator package version, neuralop.__version__, or UNO "
                "signature. These fields were backfilled by "
                "scripts/studies/lines128_uno_provenance_backfill.py using the "
                "same ptycho311 environment that produced the run; git dirty "
                "state at original launch time is not recoverable and is "
                "recorded as null."
            ),
        },
    )


def _backfill_randomness(row_dir: Path) -> None:
    contract_path = row_dir / "randomness_contract.json"
    if not contract_path.exists():
        raise SystemExit(f"missing randomness contract file: {contract_path}")
    payload = json.loads(contract_path.read_text())
    payload.setdefault("requested_seed", 3)
    payload.setdefault("effective_subsample_seed", 3)
    payload.setdefault("effective_lightning_seed", 3)
    payload["deterministic_mode"] = "warn"
    payload["deterministic_carve_out"] = (
        "neuralop_uno uses upsample_bicubic2d which lacks a deterministic "
        "CUDA backward; Lightning runs with deterministic='warn' for this "
        "architecture only."
    )
    payload["backfilled_at_utc"] = datetime.now(timezone.utc).isoformat()
    contract_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    runtime = capture_runtime_provenance()
    neuralop = capture_neuralop_provenance()
    for row in ROW_DIRS:
        row_dir = BUNDLE_ROOT / row
        if not row_dir.exists():
            raise SystemExit(f"missing row dir: {row_dir}")
        _backfill_invocation(row_dir, runtime, neuralop)
        _backfill_randomness(row_dir)
        print(f"backfilled {row_dir}")


if __name__ == "__main__":
    main()
