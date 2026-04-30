"""Tests for paper-grade provenance helpers."""

import json
from pathlib import Path

from scripts.studies.paper_provenance import write_exit_code_proof


def _write_text(path: Path, contents: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def test_write_exit_code_proof_requires_recorded_zero_exit_code(tmp_path):
    invocation_json = tmp_path / "runs" / "pinn_hybrid_resnet" / "invocation.json"
    _write_text(
        invocation_json,
        json.dumps(
            {
                "status": "completed",
                "finished_at_utc": "2026-04-30T00:00:00+00:00",
                "exit_code": 1,
            }
        ),
    )
    stdout_log = tmp_path / "runs" / "pinn_hybrid_resnet" / "stdout.log"
    stderr_log = tmp_path / "runs" / "pinn_hybrid_resnet" / "stderr.log"
    _write_text(stdout_log, "row stdout\n")
    _write_text(stderr_log, "")

    proof_path = write_exit_code_proof(
        tmp_path,
        model_id="pinn_hybrid_resnet",
        invocation_json=invocation_json,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        proof_source="test",
    )

    assert proof_path is None
    assert not (tmp_path / "runs" / "pinn_hybrid_resnet" / "exit_code_proof.json").exists()
