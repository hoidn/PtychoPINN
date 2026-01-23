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


def test_history_logging_appends_entries(tmp_path):
    """
    Test --history-jsonl and --history-markdown flags append entries correctly.

    Creates a temp inbox, runs the CLI twice (before/after injecting a maintainer ack),
    and asserts:
    1. JSONL file has 2 entries
    2. Markdown file has 2 data rows
    3. Second entry flips ack_detected to True
    """
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    history_jsonl = tmp_path / "history.jsonl"
    history_md = tmp_path / "history.md"

    # Create inbound message from Maintainer <2> (no ack keywords)
    inbound_content = """# Request: dose_experiments_ground_truth

**From:** Maintainer <2>
**To:** Maintainer <1>
**Date:** 2026-01-22T14:00:00Z

Please provide the dose experiments ground truth bundle.
"""
    create_inbox_file(inbox_dir, "request_dose_experiments_ground_truth.md", inbound_content, -3.0)

    # Run 1: No ack detected
    result1 = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--sla-hours", "2.0",
            "--history-jsonl", str(history_jsonl),
            "--history-markdown", str(history_md),
            "--output", str(output_dir / "run1")
        ],
        capture_output=True,
        text=True
    )
    assert result1.returncode == 0, f"Run 1 failed: {result1.stderr}"

    # Verify JSONL has 1 entry
    with open(history_jsonl) as f:
        lines = [json.loads(line) for line in f.readlines()]
    assert len(lines) == 1, f"Expected 1 JSONL entry after run 1, got {len(lines)}"
    assert lines[0]["ack_detected"] is False

    # Verify Markdown has header + 1 data row
    with open(history_md) as f:
        md_content = f.read()
    # Count table data rows (lines starting with |, excluding header separator)
    md_lines = [line for line in md_content.split("\n") if line.startswith("|") and not line.startswith("|-")]
    assert len(md_lines) == 2, f"Expected 2 Markdown table lines (header + 1 data), got {len(md_lines)}"
    assert "No" in md_lines[1]  # ack_detected should be No

    # Now inject an acknowledgement from Maintainer <2>
    ack_content = """# Acknowledgement: dose_experiments_ground_truth

**From:** Maintainer <2>
**To:** Maintainer <1>
**Date:** 2026-01-22T17:00:00Z

I have received and acknowledge the bundle. Thank you!
"""
    create_inbox_file(inbox_dir, "ack_dose_experiments_ground_truth.md", ack_content, -0.5)

    # Run 2: Ack should be detected
    result2 = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--sla-hours", "2.0",
            "--history-jsonl", str(history_jsonl),
            "--history-markdown", str(history_md),
            "--output", str(output_dir / "run2")
        ],
        capture_output=True,
        text=True
    )
    assert result2.returncode == 0, f"Run 2 failed: {result2.stderr}"

    # Verify JSONL has 2 entries
    with open(history_jsonl) as f:
        lines = [json.loads(line) for line in f.readlines()]
    assert len(lines) == 2, f"Expected 2 JSONL entries after run 2, got {len(lines)}"
    assert lines[0]["ack_detected"] is False, "First entry should have ack_detected=False"
    assert lines[1]["ack_detected"] is True, "Second entry should have ack_detected=True"

    # Verify Markdown has header + 2 data rows
    with open(history_md) as f:
        md_content = f.read()
    md_lines = [line for line in md_content.split("\n") if line.startswith("|") and not line.startswith("|-")]
    assert len(md_lines) == 3, f"Expected 3 Markdown table lines (header + 2 data), got {len(md_lines)}"

    # Verify second data row has ack=Yes
    assert "Yes" in md_lines[2], f"Second data row should have ack=Yes: {md_lines[2]}"

    # Verify Markdown header was only written once (check for exactly one "# Inbox Scan History")
    assert md_content.count("# Inbox Scan History") == 1, "Header should be written exactly once"


def test_status_snippet_emits_wait_summary(tmp_path):
    """
    Test --status-snippet flag generates a Markdown snippet with wait state.

    Creates a synthetic inbox with:
    - One inbound message from Maintainer <2> (3 hours old, no ack keywords)
    - One outbound message from Maintainer <1> (1 hour old)

    Asserts:
    1. Snippet file exists and contains "Maintainer Status Snapshot" heading
    2. Snippet contains "Ack Detected: No" since no ack was received
    3. Snippet contains SLA breach note when threshold is exceeded
    4. Snippet contains a timeline row for Maintainer <2>
    """
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    status_snippet_path = tmp_path / "status.md"

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

    # Run CLI with --status-snippet and --sla-hours
    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--sla-hours", "2.0",
            "--status-snippet", str(status_snippet_path),
            "--output", str(output_dir)
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Assert snippet file exists
    assert status_snippet_path.exists(), "Status snippet file not created"

    # Read and validate snippet content
    snippet_content = status_snippet_path.read_text()

    # Check for expected heading
    assert "# Maintainer Status Snapshot" in snippet_content, \
        "Snippet missing 'Maintainer Status Snapshot' heading"

    # Check for ack status (should be "No" since no ack keywords from M2)
    assert "## Ack Detected: No" in snippet_content, \
        "Snippet missing 'Ack Detected: No' section"

    # Check for SLA breach note (3 hours > 2.0 threshold)
    assert "Breached | Yes" in snippet_content or "Breached" in snippet_content and "Yes" in snippet_content, \
        "Snippet missing SLA breach indicator"

    # Check for timeline section with Maintainer 2 row
    assert "## Timeline" in snippet_content, \
        "Snippet missing Timeline section"
    assert "Maintainer 2" in snippet_content, \
        "Snippet missing timeline row for Maintainer <2>"

    # Verify snippet is idempotent (running again should overwrite, not append)
    result2 = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--sla-hours", "2.0",
            "--status-snippet", str(status_snippet_path),
            "--output", str(output_dir / "run2")
        ],
        capture_output=True,
        text=True
    )
    assert result2.returncode == 0

    snippet_content2 = status_snippet_path.read_text()
    # Should have exactly one header (not duplicated)
    assert snippet_content2.count("# Maintainer Status Snapshot") == 1, \
        "Snippet header should appear exactly once (idempotent)"


def test_escalation_note_emits_call_to_action(tmp_path):
    """
    Test --escalation-note flag generates a Markdown escalation draft.

    Creates a synthetic inbox with:
    - One inbound message from Maintainer <2> (3 hours old, no ack keywords)
    - One outbound message from Maintainer <1> (1 hour old)

    Asserts:
    1. Escalation note file exists and contains "Escalation Note" heading
    2. Note contains "Ack Detected" status in Summary Metrics
    3. Note contains "SLA Breach" text when threshold is exceeded
    4. Note contains blockquote call-to-action referencing the recipient and request pattern
    5. Note contains a timeline row for the messages
    """
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    escalation_note_path = tmp_path / "escalation_note.md"

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

    # Run CLI with --escalation-note, --escalation-recipient, and --sla-hours
    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--sla-hours", "2.0",
            "--escalation-note", str(escalation_note_path),
            "--escalation-recipient", "Maintainer <2>",
            "--output", str(output_dir)
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Assert escalation note file exists
    assert escalation_note_path.exists(), "Escalation note file not created"

    # Read and validate escalation note content
    note_content = escalation_note_path.read_text()

    # Check for expected heading
    assert "# Escalation Note" in note_content, \
        "Note missing 'Escalation Note' heading"

    # Check for ack status in Summary Metrics
    assert "## Summary Metrics" in note_content, \
        "Note missing 'Summary Metrics' section"
    assert "| Ack Detected | No |" in note_content, \
        "Note missing ack status in Summary Metrics"

    # Check for SLA breach text (3 hours > 2.0 threshold)
    assert "## SLA Watch" in note_content, \
        "Note missing 'SLA Watch' section"
    assert "| Breached | Yes |" in note_content, \
        "Note missing SLA breach indicator"
    assert "SLA BREACH" in note_content, \
        "Note missing 'SLA BREACH' warning text"

    # Check for blockquote call-to-action with recipient and request pattern
    assert "## Proposed Message" in note_content, \
        "Note missing 'Proposed Message' section"
    assert "> Hello Maintainer <2>" in note_content, \
        "Note missing recipient greeting in blockquote"
    assert "dose_experiments_ground_truth" in note_content, \
        "Note missing request pattern reference"
    assert "> Could you please confirm receipt" in note_content, \
        "Note missing confirmation request in blockquote"

    # Check for timeline section
    assert "## Timeline" in note_content, \
        "Note missing Timeline section"
    assert "Maintainer 2" in note_content, \
        "Note missing timeline row for Maintainer <2>"

    # Verify escalation note is idempotent (running again should overwrite, not append)
    result2 = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--sla-hours", "2.0",
            "--escalation-note", str(escalation_note_path),
            "--escalation-recipient", "Maintainer <2>",
            "--output", str(output_dir / "run2")
        ],
        capture_output=True,
        text=True
    )
    assert result2.returncode == 0

    note_content2 = escalation_note_path.read_text()
    # Should have exactly one header (not duplicated)
    assert note_content2.count("# Escalation Note") == 1, \
        "Escalation note header should appear exactly once (idempotent)"


def test_escalation_note_no_breach(tmp_path):
    """
    Test --escalation-note when SLA is not breached shows "No Escalation Required".

    Creates a synthetic inbox with one recent inbound message (1 hour old)
    and a 4-hour SLA threshold. Should indicate no escalation needed.
    """
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    escalation_note_path = tmp_path / "escalation_note.md"

    # Create inbound message from Maintainer <2> (1 hour ago, no ack)
    inbound_content = """# Request: dose_experiments_ground_truth

**From:** Maintainer <2>
**To:** Maintainer <1>
**Date:** 2026-01-22T16:00:00Z

Please provide the dose experiments ground truth bundle.
"""
    create_inbox_file(inbox_dir, "request_dose_experiments_ground_truth.md", inbound_content, -1.0)

    # Run CLI with --escalation-note and longer SLA threshold
    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--sla-hours", "4.0",
            "--escalation-note", str(escalation_note_path),
            "--output", str(output_dir)
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert escalation_note_path.exists(), "Escalation note file not created"

    note_content = escalation_note_path.read_text()

    # Should indicate no escalation required
    assert "No Escalation Required" in note_content, \
        "Note should indicate 'No Escalation Required' when SLA not breached"

    # Should NOT have the Proposed Message section
    assert "## Proposed Message" not in note_content, \
        "Note should not have Proposed Message when no breach"


def test_history_dashboard_summarizes_runs(tmp_path):
    """
    Test --history-dashboard generates a Markdown dashboard from JSONL history.

    Creates a fake JSONL history file with two entries (one breach, one with ack),
    invokes write_history_dashboard via CLI, and asserts the Markdown contains:
    - Total Scans count
    - Breach Count
    - Longest Wait metric
    - Recent timeline rows with both timestamps
    """
    # Import the CLI module for direct access to write_history_dashboard
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_inbox_for_ack", CLI_SCRIPT)
    cli_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli_module)

    # Create fake JSONL history file with 2 entries
    history_jsonl = tmp_path / "history.jsonl"
    dashboard_path = tmp_path / "dashboard.md"

    entry1 = {
        "generated_utc": "2026-01-23T01:00:00+00:00",
        "ack_detected": False,
        "hours_since_inbound": 2.5,
        "hours_since_outbound": 0.5,
        "sla_breached": True,
        "sla_threshold_hours": 2.0,
        "ack_files": [],
        "total_matches": 3,
        "total_inbound": 1,
        "total_outbound": 2
    }
    entry2 = {
        "generated_utc": "2026-01-23T02:00:00+00:00",
        "ack_detected": True,
        "hours_since_inbound": 3.5,
        "hours_since_outbound": 1.5,
        "sla_breached": False,
        "sla_threshold_hours": 2.0,
        "ack_files": ["ack_dose.md"],
        "total_matches": 4,
        "total_inbound": 2,
        "total_outbound": 2
    }

    with open(history_jsonl, "w") as f:
        f.write(json.dumps(entry1) + "\n")
        f.write(json.dumps(entry2) + "\n")

    # Call write_history_dashboard directly
    cli_module.write_history_dashboard(history_jsonl, dashboard_path, max_entries=10)

    # Assert dashboard file exists
    assert dashboard_path.exists(), "Dashboard file not created"

    dashboard_content = dashboard_path.read_text()

    # Check for expected heading
    assert "# Inbox History Dashboard" in dashboard_content, \
        "Dashboard missing heading"

    # Check for Total Scans count (2 entries)
    assert "| Total Scans | 2 |" in dashboard_content, \
        "Dashboard missing Total Scans count"

    # Check for Breach Count (1 breach)
    assert "| Breach Count | 1 |" in dashboard_content, \
        "Dashboard missing Breach Count"

    # Check for Longest Wait (3.5 hours from entry2)
    assert "| Longest Wait | 3.50 hours |" in dashboard_content, \
        "Dashboard missing Longest Wait metric"

    # Check for both timestamps in Recent Scans table
    assert "2026-01-23T01:00:00" in dashboard_content, \
        "Dashboard missing first entry timestamp"
    assert "2026-01-23T02:00:00" in dashboard_content, \
        "Dashboard missing second entry timestamp"

    # Check for Ack Count (1 ack)
    assert "| Ack Count | 1 |" in dashboard_content, \
        "Dashboard missing Ack Count"


def test_history_dashboard_requires_jsonl(tmp_path):
    """
    Test that --history-dashboard without --history-jsonl returns exit code 1.
    """
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    dashboard_path = tmp_path / "dashboard.md"

    # Create minimal inbox file
    inbound_content = """# Request: dose_experiments_ground_truth

**From:** Maintainer <2>
**To:** Maintainer <1>
**Date:** 2026-01-22T14:00:00Z

Please provide the dose experiments.
"""
    create_inbox_file(inbox_dir, "request_dose.md", inbound_content)

    # Run CLI with --history-dashboard but WITHOUT --history-jsonl
    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--history-dashboard", str(dashboard_path),
            "--output", str(output_dir)
        ],
        capture_output=True,
        text=True
    )

    # Should exit with code 1 due to validation error
    assert result.returncode == 1, \
        f"Expected exit 1 when --history-dashboard without --history-jsonl, got {result.returncode}"
    assert "requires --history-jsonl" in result.stdout, \
        "Expected error message about requiring --history-jsonl"


def test_ack_actor_supports_multiple_inbound_maintainers(tmp_path):
    """
    Test --ack-actor flag enables acknowledgement detection from multiple maintainers.

    Creates a synthetic inbox with:
    - One message from Maintainer <2> (no ack keywords) - should NOT trigger ack
    - One message from Maintainer <3> with ack keywords - should trigger ack
      ONLY when --ack-actor includes "Maintainer <3>"

    Asserts:
    1. With only --ack-actor "Maintainer <2>" (default): ack_detected=False (M3 not an ack source)
    2. With --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>": ack_detected=True
    3. JSON output includes ack_actors in parameters
    """
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create message from Maintainer <2> (no ack keywords)
    m2_content = """# Request: dose_experiments_ground_truth

**From:** Maintainer <2>
**To:** Maintainer <1>
**Date:** 2026-01-22T14:00:00Z

Please provide the dose experiments ground truth bundle.
"""
    create_inbox_file(inbox_dir, "request_dose_experiments_ground_truth.md", m2_content, -3.0)

    # Create message from Maintainer <3> WITH ack keywords
    m3_content = """# Acknowledgement: dose_experiments_ground_truth

**From:** Maintainer <3>
**To:** Maintainer <1>
**Date:** 2026-01-22T15:00:00Z

I have received and confirmed the dose experiments ground truth bundle on behalf of the team.
"""
    create_inbox_file(inbox_dir, "ack_m3_dose_experiments_ground_truth.md", m3_content, -2.0)

    # Test 1: Default behavior (only M2 as ack actor) - M3 message should NOT trigger ack
    result1 = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--keywords", "received",
            "--keywords", "confirmed",
            "--output", str(output_dir / "test1")
        ],
        capture_output=True,
        text=True
    )

    assert result1.returncode == 0, f"Test 1 failed: {result1.stderr}"
    json_path1 = output_dir / "test1" / "inbox_scan_summary.json"
    with open(json_path1) as f:
        data1 = json.load(f)

    # M3 has ack keywords but is not in ack_actors (default is M2 only)
    assert data1["ack_detected"] is False, \
        "Should NOT detect ack when M3 is not in ack_actors (default)"
    assert data1["parameters"]["ack_actors"] == ["maintainer_2"], \
        "Default ack_actors should be ['maintainer_2']"

    # Test 2: With both M2 and M3 as ack actors - M3 message SHOULD trigger ack
    result2 = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--keywords", "received",
            "--keywords", "confirmed",
            "--ack-actor", "Maintainer <2>",
            "--ack-actor", "Maintainer <3>",
            "--output", str(output_dir / "test2")
        ],
        capture_output=True,
        text=True
    )

    assert result2.returncode == 0, f"Test 2 failed: {result2.stderr}"
    json_path2 = output_dir / "test2" / "inbox_scan_summary.json"
    with open(json_path2) as f:
        data2 = json.load(f)

    # Now M3 is in ack_actors, so their message with ack keywords should trigger
    assert data2["ack_detected"] is True, \
        "Should detect ack when M3 is in ack_actors and has ack keywords"
    assert "maintainer_3" in data2["parameters"]["ack_actors"], \
        "ack_actors should include maintainer_3"
    assert "ack_m3_dose_experiments_ground_truth.md" in data2["ack_files"], \
        "M3's ack file should be in ack_files"


def test_custom_keywords_enable_ack_detection(tmp_path):
    """
    Test that user-provided --keywords are honored exactly (no hidden hard-coded list).

    Creates a synthetic inbox with a message from Maintainer <2> containing
    "thanks" but NOT "acknowledged"/"confirmed"/"received".

    Asserts:
    1. With --keywords "acknowledged" --keywords "confirmed": ack_detected=False
       (message doesn't contain these exact keywords)
    2. With --keywords "thanks": ack_detected=True
       (message contains "thanks")
    3. Without any --keywords: uses defaults which include "thanks" -> ack_detected=True
    """
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create message from Maintainer <2> with "thanks" but NOT standard ack keywords
    m2_content = """# Response: dose_experiments_ground_truth

**From:** Maintainer <2>
**To:** Maintainer <1>
**Date:** 2026-01-22T14:00:00Z

Thanks for the bundle! I'll review it soon.
"""
    create_inbox_file(inbox_dir, "response_dose_experiments_ground_truth.md", m2_content, -1.0)

    # Test 1: With keywords that DON'T match - should NOT detect ack
    result1 = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--keywords", "acknowledged",
            "--keywords", "confirmed",
            "--output", str(output_dir / "test1")
        ],
        capture_output=True,
        text=True
    )

    assert result1.returncode == 0, f"Test 1 failed: {result1.stderr}"
    json_path1 = output_dir / "test1" / "inbox_scan_summary.json"
    with open(json_path1) as f:
        data1 = json.load(f)

    assert data1["ack_detected"] is False, \
        "Should NOT detect ack when message doesn't contain specified keywords"

    # Test 2: With keyword that DOES match - should detect ack
    result2 = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--keywords", "thanks",
            "--output", str(output_dir / "test2")
        ],
        capture_output=True,
        text=True
    )

    assert result2.returncode == 0, f"Test 2 failed: {result2.stderr}"
    json_path2 = output_dir / "test2" / "inbox_scan_summary.json"
    with open(json_path2) as f:
        data2 = json.load(f)

    assert data2["ack_detected"] is True, \
        "Should detect ack when message contains specified keyword 'thanks'"

    # Test 3: Without any --keywords (defaults) - should detect ack since defaults include "thanks"
    result3 = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--output", str(output_dir / "test3")
        ],
        capture_output=True,
        text=True
    )

    assert result3.returncode == 0, f"Test 3 failed: {result3.stderr}"
    json_path3 = output_dir / "test3" / "inbox_scan_summary.json"
    with open(json_path3) as f:
        data3 = json.load(f)

    assert data3["ack_detected"] is True, \
        "Should detect ack with default keywords (includes 'thanks')"
    assert "thanks" in data3["parameters"]["keywords"], \
        "Default keywords should include 'thanks'"


def test_ack_actor_wait_metrics_cover_each_actor(tmp_path):
    """
    Test that ack_actor_stats provides per-actor wait metrics for each configured actor.

    Creates a synthetic inbox with:
    - One message from Maintainer <2> (4 hours old, no ack keywords)
    - One message from Maintainer <3> (2 hours old, no ack keywords)

    Runs the CLI with --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" and asserts:
    1. JSON output includes ack_actor_stats with both maintainer_2 and maintainer_3 entries
    2. Each entry has distinct hours_since_last_inbound values matching their mtime offsets
    3. Inbound counts are correct for each actor
    4. No ack is detected since neither message has ack keywords
    5. Markdown summary includes the "Ack Actor Coverage" table with both rows
    """
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create message from Maintainer <2> (4 hours ago, no ack keywords)
    m2_content = """# Request: dose_experiments_ground_truth

**From:** Maintainer <2>
**To:** Maintainer <1>
**Date:** 2026-01-22T10:00:00Z

Please provide the dose experiments ground truth bundle.
"""
    create_inbox_file(inbox_dir, "request_m2_dose_experiments_ground_truth.md", m2_content, -4.0)

    # Create message from Maintainer <3> (2 hours ago, no ack keywords)
    m3_content = """# Update: dose_experiments_ground_truth

**From:** Maintainer <3>
**To:** Maintainer <1>
**Date:** 2026-01-22T12:00:00Z

I'm following up on this request from Maintainer <2>.
"""
    create_inbox_file(inbox_dir, "update_m3_dose_experiments_ground_truth.md", m3_content, -2.0)

    # Run CLI with both M2 and M3 as ack actors
    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--keywords", "acknowledged",
            "--keywords", "confirm",
            "--ack-actor", "Maintainer <2>",
            "--ack-actor", "Maintainer <3>",
            "--sla-hours", "3.0",
            "--output", str(output_dir)
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    # Load JSON output
    json_path = output_dir / "inbox_scan_summary.json"
    assert json_path.exists(), "JSON summary not created"
    with open(json_path) as f:
        data = json.load(f)

    # Assert ack_actor_stats is present
    assert "ack_actor_stats" in data, "JSON output missing ack_actor_stats"
    ack_stats = data["ack_actor_stats"]

    # Assert both actors are present in ack_actor_stats
    assert "maintainer_2" in ack_stats, "Missing maintainer_2 in ack_actor_stats"
    assert "maintainer_3" in ack_stats, "Missing maintainer_3 in ack_actor_stats"

    # Check M2 metrics (4 hours ago)
    m2_stats = ack_stats["maintainer_2"]
    assert m2_stats["last_inbound_utc"] is not None, "M2 should have last_inbound_utc"
    assert m2_stats["hours_since_last_inbound"] is not None, "M2 should have hours_since_last_inbound"
    # Allow some tolerance for test execution time
    assert 3.5 < m2_stats["hours_since_last_inbound"] < 4.5, \
        f"M2 hours_since_last_inbound should be ~4, got {m2_stats['hours_since_last_inbound']}"
    assert m2_stats["inbound_count"] == 1, f"M2 should have 1 inbound, got {m2_stats['inbound_count']}"
    assert m2_stats["ack_detected"] is False, "M2 should not have ack (no keywords)"
    assert m2_stats["ack_files"] == [], "M2 should have no ack files"

    # Check M3 metrics (2 hours ago)
    m3_stats = ack_stats["maintainer_3"]
    assert m3_stats["last_inbound_utc"] is not None, "M3 should have last_inbound_utc"
    assert m3_stats["hours_since_last_inbound"] is not None, "M3 should have hours_since_last_inbound"
    # Allow some tolerance for test execution time
    assert 1.5 < m3_stats["hours_since_last_inbound"] < 2.5, \
        f"M3 hours_since_last_inbound should be ~2, got {m3_stats['hours_since_last_inbound']}"
    assert m3_stats["inbound_count"] == 1, f"M3 should have 1 inbound, got {m3_stats['inbound_count']}"
    assert m3_stats["ack_detected"] is False, "M3 should not have ack (no keywords)"
    assert m3_stats["ack_files"] == [], "M3 should have no ack files"

    # Verify the two actors have different wait times (M2 > M3)
    assert m2_stats["hours_since_last_inbound"] > m3_stats["hours_since_last_inbound"], \
        "M2 should have longer wait than M3"

    # No ack should be detected overall (no ack keywords in either message)
    assert data["ack_detected"] is False, "Should not detect ack (no keywords match)"

    # Verify Markdown summary includes Ack Actor Coverage table
    md_path = output_dir / "inbox_scan_summary.md"
    assert md_path.exists(), "Markdown summary not created"
    md_content = md_path.read_text()

    assert "## Ack Actor Coverage" in md_content, \
        "Markdown missing 'Ack Actor Coverage' section"
    assert "Maintainer 2" in md_content, \
        "Markdown missing Maintainer 2 row in Ack Actor Coverage table"
    assert "Maintainer 3" in md_content, \
        "Markdown missing Maintainer 3 row in Ack Actor Coverage table"
