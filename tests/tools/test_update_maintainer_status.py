"""
tests/tools/test_update_maintainer_status.py - Tests for update_maintainer_status.py CLI

Tests the automation of maintainer status updates including response block
generation and follow-up note creation.

Ref: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/update_maintainer_status.py
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


# Path to CLI script
CLI_SCRIPT = Path(__file__).parent.parent.parent / "plans" / "active" / "DEBUG-SIM-LINES-DOSE-001" / "bin" / "update_maintainer_status.py"


def create_fixture_scan_json(tmp_path: Path) -> Path:
    """
    Create a fixture inbox_scan_summary.json mirroring the production schema.

    Returns:
        Path to the created JSON file
    """
    scan_data = {
        "scanned": 5,
        "matches": [
            {
                "file": "request_test_pattern.md",
                "path": "inbox/request_test_pattern.md",
                "actor": "maintainer_2",
                "direction": "inbound",
                "ack_detected": False
            },
            {
                "file": "response_test_pattern.md",
                "path": "inbox/response_test_pattern.md",
                "actor": "maintainer_1",
                "direction": "outbound",
                "ack_detected": False
            }
        ],
        "ack_detected": False,
        "ack_files": [],
        "generated_utc": "2026-01-23T15:35:00.000000+00:00",
        "parameters": {
            "inbox": "inbox",
            "request_pattern": "test_pattern",
            "keywords": ["acknowledged", "confirm", "received"],
            "ack_actors": ["maintainer_2", "maintainer_3"],
            "ack_actor_sla_hours": {
                "maintainer_2": 2.0,
                "maintainer_3": 6.0
            }
        },
        "timeline": [
            {
                "timestamp_utc": "2026-01-22T10:00:00.000000+00:00",
                "file": "request_test_pattern.md",
                "actor": "maintainer_2",
                "direction": "inbound",
                "ack": False
            },
            {
                "timestamp_utc": "2026-01-23T12:00:00.000000+00:00",
                "file": "response_test_pattern.md",
                "actor": "maintainer_1",
                "direction": "outbound",
                "ack": False
            }
        ],
        "waiting_clock": {
            "last_inbound_utc": "2026-01-22T10:00:00.000000+00:00",
            "last_outbound_utc": "2026-01-23T12:00:00.000000+00:00",
            "hours_since_last_inbound": 5.58,
            "hours_since_last_outbound": 3.58,
            "total_inbound_count": 1,
            "total_outbound_count": 3
        },
        "ack_actor_stats": {
            "maintainer_2": {
                "last_inbound_utc": "2026-01-22T10:00:00.000000+00:00",
                "hours_since_last_inbound": 5.58,
                "inbound_count": 1,
                "ack_files": [],
                "ack_detected": False,
                "last_outbound_utc": "2026-01-23T12:00:00.000000+00:00",
                "hours_since_last_outbound": 3.58,
                "outbound_count": 3,
                "sla_threshold_hours": 2.0,
                "sla_deadline_utc": "2026-01-22T12:00:00.000000+00:00",
                "sla_breached": True,
                "sla_breach_duration_hours": 3.58,
                "sla_severity": "critical",
                "sla_notes": "SLA breach: 5.58 hours since last inbound exceeds 2.00 hour threshold"
            },
            "maintainer_3": {
                "last_inbound_utc": None,
                "hours_since_last_inbound": None,
                "inbound_count": 0,
                "ack_files": [],
                "ack_detected": False,
                "last_outbound_utc": "2026-01-23T12:00:00.000000+00:00",
                "hours_since_last_outbound": 3.58,
                "outbound_count": 2,
                "sla_threshold_hours": 6.0,
                "sla_deadline_utc": None,
                "sla_breached": False,
                "sla_breach_duration_hours": None,
                "sla_severity": "unknown",
                "sla_notes": "No inbound messages from Maintainer 3"
            }
        },
        "sla_watch": {
            "threshold_hours": 2.5,
            "hours_since_last_inbound": 5.58,
            "breached": True,
            "notes": "SLA breach: 5.58 hours since last inbound exceeds 2.50 hour threshold",
            "deadline_utc": "2026-01-22T12:30:00.000000+00:00",
            "breach_duration_hours": 3.08,
            "severity": "critical"
        }
    }

    json_path = tmp_path / "inbox_scan_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scan_data, f, indent=2)

    return json_path


def create_seed_response_file(tmp_path: Path) -> Path:
    """
    Create a seed response file to append status blocks to.

    Returns:
        Path to the created response file
    """
    response_path = tmp_path / "response_test_pattern.md"
    response_content = """# Response - Test Pattern Request

**From:** Maintainer <1>
**To:** Maintainer <2>
**Re:** Test request

---

## Initial Delivery

Bundle delivered successfully.

---
"""
    response_path.write_text(response_content)
    return response_path


def test_cli_generates_followup(tmp_path):
    """
    Test that update_maintainer_status.py generates correct response block and follow-up note.

    Creates a fixture inbox_scan_summary.json and seed response file,
    invokes the CLI via subprocess, and asserts:
    1. Response file ends with status block containing actor table
    2. Follow-up note contains timestamp, action items, and ack actor stats
    3. CLI exits with code 0
    """
    # Create fixtures
    scan_json = create_fixture_scan_json(tmp_path)
    response_path = create_seed_response_file(tmp_path)
    followup_path = tmp_path / "followup_test_pattern.md"

    # Create dummy artifact paths
    artifact1 = tmp_path / "artifact1.md"
    artifact1.write_text("# Artifact 1\n")
    artifact2 = tmp_path / "artifact2.md"
    artifact2.write_text("# Artifact 2\n")

    # Invoke CLI
    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--scan-json", str(scan_json),
            "--status-title", "Test Status Block",
            "--artifact", str(artifact1),
            "--artifact", str(artifact2),
            "--response-path", str(response_path),
            "--followup-path", str(followup_path),
            "--to", "Maintainer <2>",
            "--cc", "Maintainer <3>"
        ],
        capture_output=True,
        text=True
    )

    # Assertions
    assert result.returncode == 0, f"CLI failed with exit code {result.returncode}. stderr: {result.stderr}"

    # Check response file was updated
    response_content = response_path.read_text()

    # Response should contain status block with timestamp and title
    assert "### Status as of" in response_content, "Status block heading not found in response"
    assert "(Test Status Block)" in response_content, "Status title not found in response"

    # Response should contain ack actor table with rows for both actors
    assert "Maintainer 2" in response_content, "Maintainer 2 not found in response"
    assert "Maintainer 3" in response_content, "Maintainer 3 not found in response"
    assert "CRITICAL" in response_content, "CRITICAL severity not found in response"
    assert "UNKNOWN" in response_content, "UNKNOWN severity not found in response"

    # Response should list artifacts
    assert str(artifact1) in response_content, "Artifact 1 path not found in response"
    assert str(artifact2) in response_content, "Artifact 2 path not found in response"

    # Check follow-up note was created
    assert followup_path.exists(), "Follow-up note was not created"
    followup_content = followup_path.read_text()

    # Follow-up should contain header with request pattern
    assert "# Follow-up: test_pattern" in followup_content, "Follow-up header not found"

    # Follow-up should have To/CC recipients
    assert "**To:** Maintainer <2>" in followup_content, "To recipient not found"
    assert "**CC:** Maintainer <3>" in followup_content, "CC recipient not found"

    # Follow-up should contain SLA metrics
    assert "Global SLA Watch" in followup_content, "Global SLA Watch section not found"
    assert "Per-Actor SLA Status" in followup_content, "Per-Actor SLA Status section not found"
    assert "Ack Actor Follow-Up Activity" in followup_content, "Follow-up activity section not found"

    # Follow-up should contain action items
    assert "Action Items" in followup_content, "Action Items section not found"
    assert "Confirm receipt" in followup_content, "Action item not found"

    # Follow-up should list artifacts
    assert str(artifact1) in followup_content, "Artifact 1 path not found in follow-up"
    assert str(artifact2) in followup_content, "Artifact 2 path not found in follow-up"


def test_cli_missing_scan_json_exits_nonzero(tmp_path):
    """
    Test that CLI exits with non-zero code when scan JSON is missing.
    """
    response_path = create_seed_response_file(tmp_path)
    followup_path = tmp_path / "followup_test.md"

    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--scan-json", str(tmp_path / "nonexistent.json"),
            "--response-path", str(response_path),
            "--followup-path", str(followup_path)
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode != 0, "CLI should exit non-zero when scan JSON is missing"
    assert "not found" in result.stderr.lower(), "Error message should mention file not found"


def test_cli_missing_response_doc_exits_nonzero(tmp_path):
    """
    Test that CLI exits with non-zero code when response document is missing.
    """
    scan_json = create_fixture_scan_json(tmp_path)
    followup_path = tmp_path / "followup_test.md"

    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--scan-json", str(scan_json),
            "--response-path", str(tmp_path / "nonexistent_response.md"),
            "--followup-path", str(followup_path)
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode != 0, "CLI should exit non-zero when response doc is missing"
    assert "not found" in result.stderr.lower(), "Error message should mention file not found"


def test_cli_handles_none_timestamps(tmp_path):
    """
    Test that CLI handles None/null timestamps gracefully (shows '---' instead of crashing).
    """
    # Create minimal scan JSON with None timestamps
    scan_data = {
        "scanned": 0,
        "matches": [],
        "ack_detected": False,
        "generated_utc": "2026-01-23T15:35:00.000000+00:00",
        "parameters": {
            "inbox": "inbox",
            "request_pattern": "test_pattern",
            "keywords": [],
            "ack_actors": ["maintainer_2"],
            "ack_actor_sla_hours": {"maintainer_2": 2.0}
        },
        "timeline": [],
        "waiting_clock": {
            "last_inbound_utc": None,
            "last_outbound_utc": None,
            "hours_since_last_inbound": None,
            "hours_since_last_outbound": None,
            "total_inbound_count": 0,
            "total_outbound_count": 0
        },
        "ack_actor_stats": {
            "maintainer_2": {
                "last_inbound_utc": None,
                "hours_since_last_inbound": None,
                "inbound_count": 0,
                "ack_files": [],
                "ack_detected": False,
                "last_outbound_utc": None,
                "hours_since_last_outbound": None,
                "outbound_count": 0,
                "sla_threshold_hours": 2.0,
                "sla_deadline_utc": None,
                "sla_breached": False,
                "sla_breach_duration_hours": None,
                "sla_severity": "unknown",
                "sla_notes": "No inbound messages"
            }
        },
        "sla_watch": {
            "threshold_hours": 2.5,
            "hours_since_last_inbound": None,
            "breached": False,
            "notes": "No inbound messages",
            "deadline_utc": None,
            "breach_duration_hours": None,
            "severity": "unknown"
        }
    }

    json_path = tmp_path / "inbox_scan_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scan_data, f, indent=2)

    response_path = create_seed_response_file(tmp_path)
    followup_path = tmp_path / "followup_test.md"

    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--scan-json", str(json_path),
            "--response-path", str(response_path),
            "--followup-path", str(followup_path)
        ],
        capture_output=True,
        text=True
    )

    # Should not crash
    assert result.returncode == 0, f"CLI should handle None timestamps. stderr: {result.stderr}"

    # Check that "---" is used for missing values
    response_content = response_path.read_text()
    assert "---" in response_content, "Should use '---' for missing values"
