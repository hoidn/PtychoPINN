"""CPU-only Task 27 floor-candidate derivation from sealed reference evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .runtime_atomic import atomic_write_bytes_no_clobber
from .runtime_errors import RuntimeExecutionError, sha256_file
from .verdicts import IntegrationBridgeEvidence

FLOOR_CANDIDATE_SCHEMA_VERSION = "grid_lines_reference_floor_candidate_v1"
FLOOR_DERIVATION_METHOD = "task27_reference_tolerance_policy_v1"
AMP_SSIM_TOLERANCE = 0.035
PHASE_SSIM_TOLERANCE = 0.015
AMP_MAE_GUARD = 0.015
PHASE_MAE_GUARD = 0.025


def derive_floor_candidate(
    spec_path: Path,
    reference_id: str,
    evidence_path: Path,
    output_path: Path,
    *,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Adjudicate sealed evidence, then derive and publish candidate floors."""
    from .runtime_reference import (
        adjudicate_reference,
        arm_requirement,
        load_reference_spec,
    )
    from .verdicts import Verdict

    reference_spec = load_reference_spec(spec_path, base_dir=base_dir)
    arm = reference_spec.arm(reference_id)
    requirement = arm_requirement(arm)
    requirement_bytes = json.dumps(
        requirement.to_mapping(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    evidence_path = Path(evidence_path)
    evidence_bytes = evidence_path.read_bytes()
    evidence = IntegrationBridgeEvidence.from_sealed_artifact_bytes(evidence_bytes)
    adjudication = adjudicate_reference(arm, evidence)
    if adjudication.verdict is not Verdict.PASS:
        raise RuntimeExecutionError(
            "floor_candidate",
            "reference evidence did not adjudicate PASS "
            f"(verdict={adjudication.verdict.value if adjudication.verdict else None}, "
            f"reason={adjudication.reason})",
        )
    checkpoint = Path(evidence.selected_checkpoint)
    if not checkpoint.is_file():
        raise RuntimeExecutionError(
            "floor_candidate", f"selected checkpoint does not exist at {checkpoint}"
        )
    if sha256_file(checkpoint) != evidence.checkpoint_sha256:
        raise RuntimeExecutionError(
            "floor_candidate",
            f"selected checkpoint {checkpoint} does not match sealed evidence SHA",
        )

    candidate = {
        "schema": FLOOR_CANDIDATE_SCHEMA_VERSION,
        "amp_ssim_min": evidence.fixture_amp_ssim - AMP_SSIM_TOLERANCE,
        "phase_ssim_min": evidence.fixture_phase_ssim - PHASE_SSIM_TOLERANCE,
        "amp_mae_max": evidence.fixture_amp_mae + AMP_MAE_GUARD,
        "phase_mae_max": evidence.fixture_phase_mae + PHASE_MAE_GUARD,
        "provenance": {
            "source_spec": str(reference_spec.spec_path),
            "source_spec_sha256": sha256_file(reference_spec.spec_path),
            "reference_id": reference_id,
            "integration_bridge_requirement_schema": requirement.schema_version,
            "integration_bridge_requirement_sha256": hashlib.sha256(
                requirement_bytes
            ).hexdigest(),
            "source_evidence": str(evidence_path),
            "source_evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
            "selected_checkpoint": evidence.selected_checkpoint,
            "checkpoint_sha256": evidence.checkpoint_sha256,
            "train_npz_sha256": evidence.train_npz_sha256,
            "test_npz_sha256": evidence.test_npz_sha256,
            "amplitude_physics_gain": evidence.amplitude_physics_gain,
            "evaluator": evidence.gauge_handling,
            "method": FLOOR_DERIVATION_METHOD,
            "tolerances": {
                "amp_ssim_subtract": AMP_SSIM_TOLERANCE,
                "phase_ssim_subtract": PHASE_SSIM_TOLERANCE,
                "amp_mae_add": AMP_MAE_GUARD,
                "phase_mae_add": PHASE_MAE_GUARD,
            },
        },
    }
    encoded = (json.dumps(candidate, indent=2, sort_keys=True) + "\n").encode()
    atomic_write_bytes_no_clobber(Path(output_path), encoded, stage="floor_candidate")
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grid_lines_reference_floor_candidate",
        description="Derive a Task 27 floor candidate from sealed reference evidence.",
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    try:
        args = parser.parse_args(argv)
        candidate = derive_floor_candidate(
            args.spec, args.reference, args.evidence, args.output
        )
    except (OSError, RuntimeExecutionError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(candidate, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
