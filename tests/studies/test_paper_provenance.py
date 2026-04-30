"""Tests for paper-grade provenance helpers."""

import json
import os
from datetime import datetime
from pathlib import Path

from scripts.studies.paper_provenance import (
    write_exit_code_proof,
    write_launcher_completion_evidence,
)


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


def test_write_launcher_completion_evidence_requires_row_completion_markers(tmp_path):
    wrapper_invocation = tmp_path / "invocation.json"
    _write_text(
        wrapper_invocation,
        json.dumps(
            {
                "status": "completed",
                "exit_code": 0,
                "timestamp_utc": "2026-04-29T23:59:59+00:00",
                "finished_at_utc": "2026-04-30T00:00:00+00:00",
                "parsed_args": {"reuse_existing_recons": True},
            }
        ),
    )
    launcher_stderr = tmp_path / "launcher_stderr.log"
    _write_text(launcher_stderr, "launcher log without row completion markers\n")
    _write_text(tmp_path / "runs" / "pinn_hybrid_resnet" / "metrics.json", "{}")
    _write_text(tmp_path / "runs" / "pinn_hybrid_resnet" / "history.json", "{}")
    _write_text(tmp_path / "recons" / "pinn_hybrid_resnet" / "recon.npz")

    evidence_path = write_launcher_completion_evidence(
        tmp_path,
        model_id="pinn_hybrid_resnet",
        wrapper_invocation_json=wrapper_invocation,
        launcher_stderr_log=launcher_stderr,
    )

    assert evidence_path is None
    assert not (tmp_path / "runs" / "pinn_hybrid_resnet" / "launcher_completion.json").exists()


def test_write_launcher_completion_evidence_accepts_stdout_eval_markers_for_reused_row(tmp_path):
    wrapper_invocation = tmp_path / "invocation.json"
    _write_text(
        wrapper_invocation,
        json.dumps(
            {
                "status": "completed",
                "exit_code": 0,
                "timestamp_utc": "2026-04-29T23:59:59+00:00",
                "finished_at_utc": "2026-04-30T00:00:00+00:00",
                "parsed_args": {"reuse_existing_recons": True},
            }
        ),
    )
    launcher_stderr = tmp_path / "launcher_stderr.log"
    launcher_stdout = tmp_path / "launcher_stdout.log"
    _write_text(launcher_stderr, "")
    _write_text(
        launcher_stdout,
        "\n".join(
            [
                "DEBUG eval_reconstruction [pinn_ffno]: amp_target stats: mean=1.0",
                "DEBUG eval_reconstruction [pinn_ffno]: amp_pred stats: mean=1.0",
            ]
        ),
    )
    _write_text(tmp_path / "runs" / "pinn_ffno" / "metrics.json", "{}")
    _write_text(tmp_path / "runs" / "pinn_ffno" / "history.json", "{}")
    _write_text(tmp_path / "recons" / "pinn_ffno" / "recon.npz")

    evidence_path = write_launcher_completion_evidence(
        tmp_path,
        model_id="pinn_ffno",
        wrapper_invocation_json=wrapper_invocation,
        launcher_stderr_log=launcher_stderr,
        launcher_stdout_log=launcher_stdout,
    )

    assert evidence_path == tmp_path / "runs" / "pinn_ffno" / "launcher_completion.json"
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert payload["evidence_source"] == "wrapper_launcher_stdout_eval_markers"
    assert payload["launcher_stdout_log"] == "launcher_stdout.log"


def test_write_launcher_completion_evidence_rejects_stale_current_root_stdout_log(tmp_path):
    wrapper_invocation = tmp_path / "invocation.json"
    invocation_started = "2026-04-30T01:00:00+00:00"
    _write_text(
        wrapper_invocation,
        json.dumps(
            {
                "status": "completed",
                "exit_code": 0,
                "timestamp_utc": invocation_started,
                "finished_at_utc": "2026-04-30T01:00:05+00:00",
                "parsed_args": {"reuse_existing_recons": True},
            }
        ),
    )
    launcher_stderr = tmp_path / "launcher_stderr.log"
    launcher_stdout = tmp_path / "launcher_stdout.log"
    _write_text(launcher_stderr, "")
    _write_text(
        launcher_stdout,
        "\n".join(
            [
                "DEBUG eval_reconstruction [pinn_ffno]: amp_target stats: mean=1.0",
                "DEBUG eval_reconstruction [pinn_ffno]: amp_pred stats: mean=9.9",
            ]
        ),
    )
    stale_epoch = datetime.fromisoformat("2026-04-30T00:00:00+00:00").timestamp()
    os.utime(launcher_stdout, (stale_epoch, stale_epoch))
    _write_text(tmp_path / "runs" / "pinn_ffno" / "metrics.json", "{}")
    _write_text(tmp_path / "runs" / "pinn_ffno" / "history.json", "{}")
    _write_text(tmp_path / "recons" / "pinn_ffno" / "recon.npz")

    evidence_path = write_launcher_completion_evidence(
        tmp_path,
        model_id="pinn_ffno",
        wrapper_invocation_json=wrapper_invocation,
        launcher_stderr_log=launcher_stderr,
        launcher_stdout_log=launcher_stdout,
    )

    assert evidence_path is None
    assert not (tmp_path / "runs" / "pinn_ffno" / "launcher_completion.json").exists()
