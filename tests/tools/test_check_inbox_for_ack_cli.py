"""
tests/tools/test_check_inbox_for_ack_cli.py - Tests for check_inbox_for_ack.py CLI

Tests the SLA watch functionality including --sla-hours and --fail-when-breached flags.

Ref: plans/active/DEBUG-SIM-LINES-DOSE-001/bin/check_inbox_for_ack.py
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest


# Path to CLI script
CLI_SCRIPT = Path(__file__).parent.parent.parent / "plans" / "active" / "DEBUG-SIM-LINES-DOSE-001" / "bin" / "check_inbox_for_ack.py"


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


def test_sla_watch_flags_breach(tmp_path):
    """
    Test SLA watch with --sla-hours and --fail-when-breached flags.

    Creates a synthetic inbox with:
    - One inbound message from Maintainer <2> (3 hours old, no ack keywords)
    - One outbound message from Maintainer <1> (1 hour old)

    Tests:
    1. With 2.0 hour SLA threshold: should detect breach (3 > 2)
    2. With 4.0 hour SLA threshold: should not breach (3 < 4)
    3. With --fail-when-breached: should exit with code 2 on breach
    4. Without --fail-when-breached: should exit with code 0 even on breach
    """
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create inbound message from Maintainer <2> (3 hours ago, no ack)
    inbound_content = """# Request: dose_experiments_ground_truth

**From:** Maintainer <2>
**To:** Maintainer <1>
**Date:** 2026-01-22T14:00:00Z

Please provide the dose experiments ground truth bundle.
"""
    create_inbox_file(inbox_dir, "request_dose_experiments_ground_truth.md", inbound_content, -3.0)

    # Create outbound response from Maintainer <1> (1 hour ago)
    outbound_content = """# Response: dose_experiments_ground_truth

**From:** Maintainer <1>
**To:** Maintainer <2>
**Date:** 2026-01-22T16:00:00Z

Bundle delivered at reports/dose_experiments_ground_truth/
"""
    create_inbox_file(inbox_dir, "response_dose_experiments_ground_truth.md", outbound_content, -1.0)

    # Test 1: SLA breached with 2.0 hour threshold, no --fail-when-breached
    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--sla-hours", "2.0",
            "--output", str(output_dir / "test1")
        ],
        capture_output=True,
        text=True
    )

    # Should exit 0 (no --fail-when-breached)
    assert result.returncode == 0, f"Expected exit 0, got {result.returncode}. stderr: {result.stderr}"

    # Check JSON output for breach
    json_path = output_dir / "test1" / "inbox_scan_summary.json"
    assert json_path.exists(), "JSON summary not created"
    with open(json_path) as f:
        data = json.load(f)

    assert "sla_watch" in data, "sla_watch not in JSON output"
    assert data["sla_watch"]["threshold_hours"] == 2.0
    assert data["sla_watch"]["breached"] is True, "Expected breach with 2.0 hour threshold"
    assert data["ack_detected"] is False, "Should not detect ack (no ack keywords)"

    # Test 2: SLA not breached with 4.0 hour threshold
    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--sla-hours", "4.0",
            "--output", str(output_dir / "test2")
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    json_path = output_dir / "test2" / "inbox_scan_summary.json"
    with open(json_path) as f:
        data = json.load(f)

    assert data["sla_watch"]["threshold_hours"] == 4.0
    assert data["sla_watch"]["breached"] is False, "Should not breach with 4.0 hour threshold"

    # Test 3: SLA breached with --fail-when-breached
    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--sla-hours", "2.0",
            "--fail-when-breached",
            "--output", str(output_dir / "test3")
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode == 2, f"Expected exit 2 on breach with --fail-when-breached, got {result.returncode}"

    json_path = output_dir / "test3" / "inbox_scan_summary.json"
    with open(json_path) as f:
        data = json.load(f)
    assert data["sla_watch"]["breached"] is True

    # Test 4: Not breached with --fail-when-breached should still exit 0
    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--sla-hours", "10.0",
            "--fail-when-breached",
            "--output", str(output_dir / "test4")
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"Expected exit 0 when not breached, got {result.returncode}"


def test_sla_watch_with_ack_detected(tmp_path):
    """
    Test that SLA breach does NOT trigger when ack_detected is True.

    Even if hours since inbound exceeds threshold, having an acknowledgement
    means the SLA is satisfied.
    """
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create inbound message from Maintainer <2> with ACK keyword (5 hours ago)
    inbound_content = """# Acknowledgement: dose_experiments_ground_truth

**From:** Maintainer <2>
**To:** Maintainer <1>
**Date:** 2026-01-22T10:00:00Z

I have received and acknowledge the dose experiments ground truth bundle. Confirmed working.
"""
    create_inbox_file(inbox_dir, "ack_dose_experiments_ground_truth.md", inbound_content, -5.0)

    # Test: SLA threshold of 2.0 hours should NOT breach because ack was detected
    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--sla-hours", "2.0",
            "--fail-when-breached",
            "--output", str(output_dir)
        ],
        capture_output=True,
        text=True
    )

    # Should exit 0 because ack_detected is True
    assert result.returncode == 0, f"Expected exit 0 when ack detected, got {result.returncode}"

    json_path = output_dir / "inbox_scan_summary.json"
    with open(json_path) as f:
        data = json.load(f)

    assert data["ack_detected"] is True, "Should detect acknowledgement"
    assert data["sla_watch"]["breached"] is False, "SLA should not breach when ack is detected"
    assert "not applicable" in data["sla_watch"]["notes"].lower()


def test_sla_watch_no_inbound_messages(tmp_path):
    """
    Test SLA watch when there are no inbound messages from Maintainer <2>.

    When no inbound exists, hours_since_last_inbound is None and breach is False.
    """
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create only outbound message from Maintainer <1>
    outbound_content = """# Request: dose_experiments_ground_truth

**From:** Maintainer <1>
**To:** Maintainer <2>
**Date:** 2026-01-22T14:00:00Z

Initiating dose experiments ground truth discussion.
"""
    create_inbox_file(inbox_dir, "outbound_dose_experiments_ground_truth.md", outbound_content, -1.0)

    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--sla-hours", "2.0",
            "--fail-when-breached",
            "--output", str(output_dir)
        ],
        capture_output=True,
        text=True
    )

    # Should exit 0 - cannot breach without inbound messages
    assert result.returncode == 0

    json_path = output_dir / "inbox_scan_summary.json"
    with open(json_path) as f:
        data = json.load(f)

    assert data["sla_watch"]["hours_since_last_inbound"] is None
    assert data["sla_watch"]["breached"] is False
    assert "no inbound" in data["sla_watch"]["notes"].lower()
