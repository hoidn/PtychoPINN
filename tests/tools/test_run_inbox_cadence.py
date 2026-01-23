"""
tests/tools/test_run_inbox_cadence.py - Tests for run_inbox_cadence.py CLI

Tests the unified cadence CLI that orchestrates check_inbox_for_ack.py
and update_maintainer_status.py in a single maintainer follow-up run.

Ref: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_inbox_cadence.py
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest


# Path to CLI script
CLI_SCRIPT = Path(__file__).parent.parent.parent / "plans" / "active" / "DEBUG-SIM-LINES-DOSE-001" / "bin" / "run_inbox_cadence.py"


def create_inbox_file(inbox_dir: Path, filename: str, content: str, mtime_offset_hours: float = 0):
    """
    Create a file in the inbox with optional modified time offset.

    Args:
        inbox_dir: Directory to create file in
        filename: Name of the file
        content: File content
        mtime_offset_hours: Hours to offset from now (negative = in the past)
    """
    filepath = inbox_dir / filename
    filepath.write_text(content)

    # Set mtime if offset specified
    if mtime_offset_hours != 0:
        now = datetime.now(timezone.utc)
        target_time = now + timedelta(hours=mtime_offset_hours)
        ts = target_time.timestamp()
        os.utime(filepath, (ts, ts))

    return filepath


def test_cadence_sequence_creates_artifacts(tmp_path):
    """
    Test that cadence CLI creates all expected artifacts when ack is NOT detected.

    This test:
    1. Creates a temp inbox with inbound/outbound messages (no ack keywords)
    2. Creates a response document with header
    3. Runs the cadence CLI with deterministic timestamp
    4. Asserts:
       - Exit code is 0
       - Directory structure exists with logs, scan JSON, status snippet, etc.
       - Response doc now has a status block appended
       - Follow-up note was created at expected path
       - Metadata JSON shows ack_detected=false, followup_written=true
    """
    # Setup: Create inbox with messages lacking ack keywords
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()

    # Inbound message from Maintainer <2> (1.5 hours ago, no ack)
    inbound_content = """# Request: dose_experiments_ground_truth

**From:** Maintainer <2>
**To:** Maintainer <1>
**Date:** 2026-01-23T14:00:00Z

Please provide the dose experiments ground truth bundle with SHA256 manifests.
"""
    create_inbox_file(inbox_dir, "request_dose_experiments_ground_truth_2026-01-22T014445Z.md", inbound_content, -1.5)

    # Outbound response from Maintainer <1> (0.5 hours ago)
    outbound_content = """# Response: dose_experiments_ground_truth

**From:** Maintainer <1>
**To:** Maintainer <2>
**Date:** 2026-01-23T15:00:00Z

Bundle delivered at reports/dose_experiments_ground_truth/
"""
    create_inbox_file(inbox_dir, "response_dose_experiments_ground_truth.md", outbound_content, -0.5)

    # Output directories
    output_root = tmp_path / "reports"
    followup_dir = tmp_path / "followups"

    # Deterministic timestamp
    timestamp = "2026-01-23T163500Z"

    # Run the cadence CLI
    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--keywords", "acknowledged",
            "--keywords", "confirm",
            "--keywords", "received",
            "--keywords", "thanks",
            "--ack-actor", "Maintainer <2>",
            "--ack-actor", "Maintainer <3>",
            "--ack-actor-sla", "Maintainer <2>=2.0",
            "--ack-actor-sla", "Maintainer <3>=6.0",
            "--sla-hours", "2.5",
            "--fail-when-breached",
            "--output-root", str(output_root),
            "--timestamp", timestamp,
            "--response-path", str(inbox_dir / "response_dose_experiments_ground_truth.md"),
            "--followup-dir", str(followup_dir),
            "--followup-prefix", "followup_dose_experiments_ground_truth",
            "--status-title", "Maintainer Status Automation",
            "--to", "Maintainer <2>",
            "--cc", "Maintainer <3>",
            "--escalation-brief-recipient", "Maintainer <3>",
            "--escalation-brief-target", "Maintainer <2>",
        ],
        capture_output=True,
        text=True
    )

    # Check exit code
    assert result.returncode == 0, f"Expected exit 0, got {result.returncode}. stderr: {result.stderr}\nstdout: {result.stdout}"

    # Check directory structure
    output_dir = output_root / timestamp
    assert output_dir.exists(), f"Output dir not created: {output_dir}"

    logs_dir = output_dir / "logs"
    assert logs_dir.exists(), f"Logs dir not created: {logs_dir}"
    assert (logs_dir / "check_inbox_for_ack.log").exists(), "check_inbox log not created"
    assert (logs_dir / "update_maintainer_status.log").exists(), "update_maintainer_status log not created"

    # Check inbox_sla_watch outputs
    sla_watch_dir = output_dir / "inbox_sla_watch"
    assert sla_watch_dir.exists(), f"SLA watch dir not created: {sla_watch_dir}"
    assert (sla_watch_dir / "inbox_scan_summary.json").exists(), "Scan JSON not created"
    assert (sla_watch_dir / "inbox_scan_summary.md").exists(), "Scan MD not created"

    # Check inbox_history outputs
    history_dir = output_dir / "inbox_history"
    assert history_dir.exists(), f"History dir not created: {history_dir}"
    assert (history_dir / "inbox_sla_watch.jsonl").exists(), "History JSONL not created"
    assert (history_dir / "inbox_history_dashboard.md").exists(), "History dashboard not created"

    # Check inbox_status outputs
    status_dir = output_dir / "inbox_status"
    assert status_dir.exists(), f"Status dir not created: {status_dir}"
    assert (status_dir / "status_snippet.md").exists(), "Status snippet not created"
    assert (status_dir / "escalation_note.md").exists(), "Escalation note not created"

    # Check cadence metadata
    metadata_path = output_dir / "cadence_metadata.json"
    assert metadata_path.exists(), "Cadence metadata JSON not created"
    with open(metadata_path) as f:
        metadata = json.load(f)

    assert metadata["timestamp"] == timestamp, f"Timestamp mismatch: {metadata['timestamp']}"
    assert metadata["ack_detected"] is False, "Should not detect ack (no ack keywords)"
    assert metadata["followup_written"] is True, "Follow-up should be written when no ack"
    assert metadata["status_appended"] is True, "Status should be appended"

    # Check cadence summary
    summary_path = output_dir / "cadence_summary.md"
    assert summary_path.exists(), "Cadence summary MD not created"
    summary_content = summary_path.read_text()
    assert "Acknowledgement Detected:** No" in summary_content
    assert "Follow-up Written:** Yes" in summary_content

    # Check follow-up was created
    followup_path = followup_dir / f"followup_dose_experiments_ground_truth_{timestamp}.md"
    assert followup_path.exists(), f"Follow-up not created at: {followup_path}"
    followup_content = followup_path.read_text()
    assert "dose_experiments_ground_truth" in followup_content
    assert "Maintainer <2>" in followup_content

    # Check response doc was updated with status block
    response_content = (inbox_dir / "response_dose_experiments_ground_truth.md").read_text()
    assert "### Status as of" in response_content, "Status block not appended to response doc"
    assert "Maintainer Status Automation" in response_content


def test_cadence_skips_followup_on_ack(tmp_path):
    """
    Test that cadence CLI skips follow-up when ack is detected and --skip-followup-on-ack is set.

    This test:
    1. Creates a temp inbox with a message containing ack keyword from Maintainer <2>
    2. Creates a response document
    3. Runs the cadence CLI with --skip-followup-on-ack
    4. Asserts:
       - Exit code is 3 (ack detected + follow-up skipped)
       - Metadata shows ack_detected=true, followup_written=false
       - No follow-up file was created
       - Logs and scan JSON still exist (cadence ran but skipped follow-up)
    """
    # Setup: Create inbox with ack message
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()

    # Inbound message from Maintainer <2> WITH ack keyword (1 hour ago)
    inbound_content = """# Acknowledgement: dose_experiments_ground_truth

**From:** Maintainer <2>
**To:** Maintainer <1>
**Date:** 2026-01-23T15:00:00Z

I acknowledge receipt of the dose experiments ground truth bundle.
Thanks for sending it, looks good!
"""
    create_inbox_file(inbox_dir, "ack_dose_experiments_ground_truth.md", inbound_content, -1.0)

    # Response doc
    response_content = """# Response: dose_experiments_ground_truth

**From:** Maintainer <1>
**To:** Maintainer <2>
**Date:** 2026-01-23T14:00:00Z

Bundle delivered.
"""
    create_inbox_file(inbox_dir, "response_dose_experiments_ground_truth.md", response_content, -2.0)

    # Output directories
    output_root = tmp_path / "reports"
    followup_dir = tmp_path / "followups"

    # Deterministic timestamp
    timestamp = "2026-01-23T170000Z"

    # Run the cadence CLI with --skip-followup-on-ack
    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--keywords", "acknowledged",
            "--keywords", "acknowledge",
            "--keywords", "confirm",
            "--keywords", "thanks",
            "--ack-actor", "Maintainer <2>",
            "--sla-hours", "2.5",
            "--output-root", str(output_root),
            "--timestamp", timestamp,
            "--response-path", str(inbox_dir / "response_dose_experiments_ground_truth.md"),
            "--followup-dir", str(followup_dir),
            "--followup-prefix", "followup_dose_experiments_ground_truth",
            "--status-title", "Maintainer Status Automation",
            "--to", "Maintainer <2>",
            "--skip-followup-on-ack",
        ],
        capture_output=True,
        text=True
    )

    # Check exit code is 3 (ack + skip follow-up)
    assert result.returncode == 3, f"Expected exit 3, got {result.returncode}. stderr: {result.stderr}\nstdout: {result.stdout}"

    # Check directory structure still created
    output_dir = output_root / timestamp
    assert output_dir.exists(), f"Output dir not created: {output_dir}"

    # Check logs exist
    logs_dir = output_dir / "logs"
    assert logs_dir.exists(), f"Logs dir not created: {logs_dir}"
    assert (logs_dir / "check_inbox_for_ack.log").exists(), "check_inbox log not created"
    # update_maintainer_status.log should NOT exist (skipped)
    assert not (logs_dir / "update_maintainer_status.log").exists(), "update_maintainer_status should not run when skipping"

    # Check scan JSON exists
    sla_watch_dir = output_dir / "inbox_sla_watch"
    assert (sla_watch_dir / "inbox_scan_summary.json").exists(), "Scan JSON should exist even when skipping follow-up"

    # Check metadata
    metadata_path = output_dir / "cadence_metadata.json"
    assert metadata_path.exists(), "Cadence metadata JSON not created"
    with open(metadata_path) as f:
        metadata = json.load(f)

    assert metadata["timestamp"] == timestamp
    assert metadata["ack_detected"] is True, "Should detect ack (ack keyword present)"
    assert metadata["followup_written"] is False, "Follow-up should NOT be written when ack + skip"
    assert metadata["status_appended"] is False, "Status should NOT be appended when skipping"

    # Check NO follow-up was created
    followup_path = followup_dir / f"followup_dose_experiments_ground_truth_{timestamp}.md"
    assert not followup_path.exists(), f"Follow-up should NOT be created when skipping: {followup_path}"

    # Check summary reflects skip
    summary_path = output_dir / "cadence_summary.md"
    summary_content = summary_path.read_text()
    assert "Acknowledgement Detected:** Yes" in summary_content
    assert "Follow-up Written:** No" in summary_content
    assert "Exit Code:** 3" in summary_content
