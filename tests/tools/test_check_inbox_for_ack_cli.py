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


def test_sla_watch_reports_deadline_and_severity(tmp_path):
    """
    Test SLA watch reports deadline_utc, breach_duration_hours, and severity.

    Creates a synthetic inbox with:
    - One message from Maintainer <2> (3.5 hours old, no ack keywords)

    Runs the CLI twice:
    1. With threshold 2.0h: should breach with severity 'critical' (1.5h late >= 1h)
    2. With threshold 10.0h: should NOT breach and severity should be 'ok'

    Asserts:
    - JSON sla_watch block contains deadline_utc, breach_duration_hours, severity
    - Markdown summary includes "Deadline", "Breach Duration", "Severity" lines
    - Severity flips from breach (critical) to 'ok' when threshold exceeds wait
    """
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create message from Maintainer <2> (3.5 hours ago, no ack keywords)
    m2_content = """# Request: dose_experiments_ground_truth

**From:** Maintainer <2>
**To:** Maintainer <1>
**Date:** 2026-01-22T10:00:00Z

Please provide the dose experiments ground truth bundle.
"""
    create_inbox_file(inbox_dir, "request_m2_dose.md", m2_content, -3.5)

    # Test 1: SLA threshold 2.0h - should breach (3.5 > 2.0, at least 1.5h late)
    result1 = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--keywords", "acknowledged",
            "--keywords", "confirm",
            "--sla-hours", "2.0",
            "--output", str(output_dir / "test1")
        ],
        capture_output=True,
        text=True
    )

    assert result1.returncode == 0, f"Test 1 failed: {result1.stderr}"

    # Load JSON output
    json_path1 = output_dir / "test1" / "inbox_scan_summary.json"
    assert json_path1.exists(), "JSON summary not created for test 1"
    with open(json_path1) as f:
        data1 = json.load(f)

    # Assert sla_watch contains new fields
    assert "sla_watch" in data1, "JSON missing sla_watch"
    sla1 = data1["sla_watch"]

    assert "deadline_utc" in sla1, "sla_watch missing deadline_utc"
    assert sla1["deadline_utc"] is not None, "deadline_utc should not be None when inbound exists"

    assert "breach_duration_hours" in sla1, "sla_watch missing breach_duration_hours"
    assert sla1["breach_duration_hours"] is not None, "breach_duration_hours should be set"
    # 3.5 - 2.0 = 1.5 hours breach (allow some tolerance for test timing)
    assert 1.0 < sla1["breach_duration_hours"] < 2.0, \
        f"breach_duration_hours should be ~1.5, got {sla1['breach_duration_hours']}"

    assert "severity" in sla1, "sla_watch missing severity"
    # 1.5 hours late >= 1.0 means critical
    assert sla1["severity"] == "critical", \
        f"Severity should be 'critical' for breach >= 1 hour, got {sla1['severity']}"

    assert sla1["breached"] is True, "Should be breached with 2.0 hour threshold"

    # Check Markdown includes the new fields
    md_path1 = output_dir / "test1" / "inbox_scan_summary.md"
    assert md_path1.exists(), "Markdown summary not created for test 1"
    md_content1 = md_path1.read_text()

    assert "**Deadline (UTC):**" in md_content1, "Markdown missing Deadline line"
    assert "**Breach Duration:**" in md_content1, "Markdown missing Breach Duration line"
    assert "**Severity:**" in md_content1, "Markdown missing Severity line"
    assert "critical" in md_content1.lower(), "Markdown should show critical severity"

    # Test 2: SLA threshold 10.0h - should NOT breach, severity 'ok'
    result2 = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--keywords", "acknowledged",
            "--keywords", "confirm",
            "--sla-hours", "10.0",
            "--output", str(output_dir / "test2")
        ],
        capture_output=True,
        text=True
    )

    assert result2.returncode == 0, f"Test 2 failed: {result2.stderr}"

    json_path2 = output_dir / "test2" / "inbox_scan_summary.json"
    with open(json_path2) as f:
        data2 = json.load(f)

    sla2 = data2["sla_watch"]
    assert sla2["breached"] is False, "Should NOT be breached with 10.0 hour threshold"
    assert sla2["severity"] == "ok", \
        f"Severity should be 'ok' when not breached, got {sla2['severity']}"
    assert sla2["breach_duration_hours"] == 0.0, \
        f"breach_duration_hours should be 0 when not breached, got {sla2['breach_duration_hours']}"

    # Verify deadline is still computed (even when not breached)
    assert sla2["deadline_utc"] is not None, "deadline_utc should be set even when not breached"


def test_ack_actor_sla_metrics_include_deadline(tmp_path):
    """
    Test that ack_actor_stats includes per-actor SLA fields when --sla-hours is set.

    Creates a synthetic inbox with:
    - One message from Maintainer <2> (3.5 hours old, no ack keywords)
    - No message from Maintainer <3> (actor configured but no inbound)

    Runs the CLI with --ack-actor flags for M2 and M3, plus --sla-hours 2.0, and asserts:
    1. JSON ack_actor_stats['maintainer_2'] includes sla_deadline_utc, sla_breach_duration_hours > 1, sla_severity == "critical"
    2. JSON ack_actor_stats['maintainer_3'] reports sla_severity == "unknown" (no inbound)
    3. Markdown inbox_scan_summary.md contains expanded table headers (Deadline/Breached/Severity/Notes)
    4. Actors without inbound show appropriate notes ("No inbound messages...")
    """
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create message from Maintainer <2> (3.5 hours ago, no ack keywords)
    m2_content = """# Request: dose_experiments_ground_truth

**From:** Maintainer <2>
**To:** Maintainer <1>
**Date:** 2026-01-22T10:00:00Z

Please provide the dose experiments ground truth bundle.
"""
    create_inbox_file(inbox_dir, "request_m2_dose_experiments_ground_truth.md", m2_content, -3.5)

    # NO message from Maintainer <3> - they are configured but have no inbound

    # Run CLI with both M2 and M3 as ack actors, with --sla-hours 2.0
    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--keywords", "acknowledged",
            "--keywords", "confirm",
            "--ack-actor", "Maintainer <2>",
            "--ack-actor", "Maintainer <3>",
            "--sla-hours", "2.0",
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

    # Check M2 has per-actor SLA fields (breached - 3.5 > 2.0, so 1.5 hours late = critical)
    m2_stats = ack_stats["maintainer_2"]
    assert "sla_deadline_utc" in m2_stats, "M2 missing sla_deadline_utc"
    assert m2_stats["sla_deadline_utc"] is not None, "M2 sla_deadline_utc should not be None"
    assert "sla_breached" in m2_stats, "M2 missing sla_breached"
    assert m2_stats["sla_breached"] is True, f"M2 should be breached (3.5h > 2.0h), got {m2_stats['sla_breached']}"
    assert "sla_breach_duration_hours" in m2_stats, "M2 missing sla_breach_duration_hours"
    # 3.5 - 2.0 = 1.5 hours breach duration (allow some tolerance for test timing)
    assert 1.0 < m2_stats["sla_breach_duration_hours"] < 2.0, \
        f"M2 breach_duration should be ~1.5, got {m2_stats['sla_breach_duration_hours']}"
    assert "sla_severity" in m2_stats, "M2 missing sla_severity"
    assert m2_stats["sla_severity"] == "critical", \
        f"M2 severity should be 'critical' (breach >= 1 hour), got {m2_stats['sla_severity']}"
    assert "sla_notes" in m2_stats, "M2 missing sla_notes"
    assert "SLA breach" in m2_stats["sla_notes"], f"M2 notes should mention breach, got {m2_stats['sla_notes']}"

    # Check M3 has per-actor SLA fields with severity "unknown" (no inbound messages)
    m3_stats = ack_stats["maintainer_3"]
    assert "sla_deadline_utc" in m3_stats, "M3 missing sla_deadline_utc"
    assert m3_stats["sla_deadline_utc"] is None, "M3 sla_deadline_utc should be None (no inbound)"
    assert "sla_breached" in m3_stats, "M3 missing sla_breached"
    assert m3_stats["sla_breached"] is False, "M3 should not be breached (no inbound)"
    assert "sla_severity" in m3_stats, "M3 missing sla_severity"
    assert m3_stats["sla_severity"] == "unknown", \
        f"M3 severity should be 'unknown' (no inbound), got {m3_stats['sla_severity']}"
    assert "sla_notes" in m3_stats, "M3 missing sla_notes"
    assert "No inbound" in m3_stats["sla_notes"], \
        f"M3 notes should mention 'No inbound', got {m3_stats['sla_notes']}"

    # Verify Markdown summary includes expanded Ack Actor Coverage table
    md_path = output_dir / "inbox_scan_summary.md"
    assert md_path.exists(), "Markdown summary not created"
    md_content = md_path.read_text()

    assert "## Ack Actor Coverage" in md_content, \
        "Markdown missing 'Ack Actor Coverage' section"
    # Check for expanded table headers when --sla-hours is used
    assert "Deadline" in md_content, \
        "Markdown Ack Actor Coverage table missing 'Deadline' column"
    assert "Breached" in md_content, \
        "Markdown Ack Actor Coverage table missing 'Breached' column"
    assert "Severity" in md_content, \
        "Markdown Ack Actor Coverage table missing 'Severity' column"
    assert "Notes" in md_content, \
        "Markdown Ack Actor Coverage table missing 'Notes' column"
    # Check both actors appear in table
    assert "Maintainer 2" in md_content, "Markdown missing Maintainer 2 row"
    assert "Maintainer 3" in md_content, "Markdown missing Maintainer 3 row"
    # Check severity values appear
    assert "critical" in md_content.lower(), "Markdown should show 'critical' severity for M2"
    assert "unknown" in md_content.lower(), "Markdown should show 'unknown' severity for M3"


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


def test_ack_actor_sla_overrides_thresholds(tmp_path):
    """
    Test --ack-actor-sla flag allows per-actor SLA threshold overrides.

    Creates a synthetic inbox with:
    - One message from Maintainer <2> (3.5 hours old, no ack keywords)
    - One message from Maintainer <3> (1.0 hour old, no ack keywords)

    Runs the CLI with:
    - --sla-hours 2.5 (global threshold)
    - --ack-actor-sla "Maintainer <2>=2.0" (M2 has stricter threshold)
    - --ack-actor-sla "Maintainer <3>=4.0" (M3 has looser threshold)

    Asserts:
    1. JSON ack_actor_stats['maintainer_2'] shows sla_threshold_hours=2.0, sla_breached=True (3.5 > 2.0)
    2. JSON ack_actor_stats['maintainer_3'] shows sla_threshold_hours=4.0, sla_breached=False (1.0 < 4.0)
    3. JSON parameters['ack_actor_sla_hours'] includes the override map
    4. Markdown Ack Actor Coverage table includes "Threshold (hrs)" column
    5. CLI stdout shows "Per-actor SLA overrides" and threshold values per actor
    """
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create message from Maintainer <2> (3.5 hours ago, no ack keywords)
    m2_content = """# Request: dose_experiments_ground_truth

**From:** Maintainer <2>
**To:** Maintainer <1>
**Date:** 2026-01-22T10:00:00Z

Please provide the dose experiments ground truth bundle.
"""
    create_inbox_file(inbox_dir, "request_m2_dose_experiments_ground_truth.md", m2_content, -3.5)

    # Create message from Maintainer <3> (1.0 hour ago, no ack keywords)
    m3_content = """# Update: dose_experiments_ground_truth

**From:** Maintainer <3>
**To:** Maintainer <1>
**Date:** 2026-01-22T12:30:00Z

I'm following up on this request from Maintainer <2>.
"""
    create_inbox_file(inbox_dir, "update_m3_dose_experiments_ground_truth.md", m3_content, -1.0)

    # Run CLI with per-actor SLA overrides
    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--keywords", "acknowledged",
            "--keywords", "confirm",
            "--ack-actor", "Maintainer <2>",
            "--ack-actor", "Maintainer <3>",
            "--sla-hours", "2.5",
            "--ack-actor-sla", "Maintainer <2>=2.0",
            "--ack-actor-sla", "Maintainer <3>=4.0",
            "--output", str(output_dir)
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}\nstdout: {result.stdout}"

    # Load JSON output
    json_path = output_dir / "inbox_scan_summary.json"
    assert json_path.exists(), "JSON summary not created"
    with open(json_path) as f:
        data = json.load(f)

    # Assert parameters include ack_actor_sla_hours
    assert "parameters" in data, "JSON missing parameters"
    params = data["parameters"]
    assert "ack_actor_sla_hours" in params, "JSON parameters missing ack_actor_sla_hours"
    sla_overrides = params["ack_actor_sla_hours"]
    assert sla_overrides.get("maintainer_2") == 2.0, \
        f"Expected maintainer_2 override 2.0, got {sla_overrides.get('maintainer_2')}"
    assert sla_overrides.get("maintainer_3") == 4.0, \
        f"Expected maintainer_3 override 4.0, got {sla_overrides.get('maintainer_3')}"

    # Assert ack_actor_stats is present
    assert "ack_actor_stats" in data, "JSON output missing ack_actor_stats"
    ack_stats = data["ack_actor_stats"]

    # Assert both actors are present
    assert "maintainer_2" in ack_stats, "Missing maintainer_2 in ack_actor_stats"
    assert "maintainer_3" in ack_stats, "Missing maintainer_3 in ack_actor_stats"

    # Check M2 has threshold 2.0 and is BREACHED (3.5 > 2.0)
    m2_stats = ack_stats["maintainer_2"]
    assert "sla_threshold_hours" in m2_stats, "M2 missing sla_threshold_hours"
    assert m2_stats["sla_threshold_hours"] == 2.0, \
        f"M2 threshold should be 2.0 (override), got {m2_stats['sla_threshold_hours']}"
    assert m2_stats["sla_breached"] is True, \
        f"M2 should be breached (3.5 > 2.0 override), got {m2_stats['sla_breached']}"
    assert m2_stats["sla_severity"] in ("warning", "critical"), \
        f"M2 severity should be warning/critical when breached, got {m2_stats['sla_severity']}"

    # Check M3 has threshold 4.0 and is NOT BREACHED (1.0 < 4.0)
    m3_stats = ack_stats["maintainer_3"]
    assert "sla_threshold_hours" in m3_stats, "M3 missing sla_threshold_hours"
    assert m3_stats["sla_threshold_hours"] == 4.0, \
        f"M3 threshold should be 4.0 (override), got {m3_stats['sla_threshold_hours']}"
    assert m3_stats["sla_breached"] is False, \
        f"M3 should NOT be breached (1.0 < 4.0 override), got {m3_stats['sla_breached']}"
    assert m3_stats["sla_severity"] == "ok", \
        f"M3 severity should be 'ok' when not breached, got {m3_stats['sla_severity']}"

    # Verify Markdown summary includes Threshold column
    md_path = output_dir / "inbox_scan_summary.md"
    assert md_path.exists(), "Markdown summary not created"
    md_content = md_path.read_text()

    assert "## Ack Actor Coverage" in md_content, \
        "Markdown missing 'Ack Actor Coverage' section"
    assert "Threshold (hrs)" in md_content, \
        "Markdown Ack Actor Coverage table missing 'Threshold (hrs)' column"
    # Check threshold values appear in table
    assert "2.00" in md_content, \
        "Markdown should show threshold 2.00 for M2"
    assert "4.00" in md_content, \
        "Markdown should show threshold 4.00 for M3"

    # Verify CLI stdout shows per-actor SLA overrides
    assert "Per-actor SLA overrides" in result.stdout, \
        "CLI stdout should show 'Per-actor SLA overrides'"
    assert "maintainer_2" in result.stdout.lower() or "Maintainer 2" in result.stdout, \
        "CLI stdout should mention Maintainer 2"
    assert "SLA Threshold: 2.00 hours" in result.stdout, \
        "CLI stdout should show M2's threshold value"
    assert "SLA Threshold: 4.00 hours" in result.stdout, \
        "CLI stdout should show M3's threshold value"


def test_ack_actor_sla_summary_flags_breach(tmp_path):
    """
    Test --ack-actor-sla summary groups actors by severity (critical/warning/ok/unknown).

    Creates a synthetic inbox where:
    - Maintainer <2> has a message 3.5 hours old with SLA threshold 2.0 hours → breach (critical)
    - Maintainer <3> has a message 1.0 hour old with SLA threshold 6.0 hours → within SLA (ok)

    Runs CLI with both --sla-hours and --ack-actor-sla flags.

    Note: The --fail-when-breached flag checks the *global* SLA watch, not per-actor breaches.
    The global SLA uses the latest inbound across all actors (1.0h from M3 < 2.5h threshold),
    so the global SLA is NOT breached even though M2's per-actor SLA IS breached.

    Asserts:
    1. JSON includes ack_actor_summary with correct buckets (critical/warning/ok/unknown)
    2. JSON ack_actor_summary.critical contains maintainer_2 entry with correct metadata
    3. JSON ack_actor_summary.ok contains maintainer_3 entry with correct metadata
    4. Markdown includes "## Ack Actor SLA Summary" section with severity subsections
    5. CLI stdout includes "Ack Actor SLA Summary:" with severity buckets listed
    6. CLI exits with code 0 (global SLA not breached, even though M2 is per-actor breached)
    """
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create message from Maintainer <2> (3.5 hours ago, no ack keywords)
    m2_content = """# Request: dose_experiments_ground_truth

**From:** Maintainer <2>
**To:** Maintainer <1>
**Date:** 2026-01-22T10:00:00Z

Please provide the dose experiments ground truth bundle.
"""
    create_inbox_file(inbox_dir, "request_m2_dose_experiments_ground_truth.md", m2_content, -3.5)

    # Create message from Maintainer <3> (1.0 hour ago, no ack keywords)
    m3_content = """# Update: dose_experiments_ground_truth

**From:** Maintainer <3>
**To:** Maintainer <1>
**Date:** 2026-01-22T12:30:00Z

I'm following up on this request from Maintainer <2>.
"""
    create_inbox_file(inbox_dir, "update_m3_dose_experiments_ground_truth.md", m3_content, -1.0)

    # Run CLI with per-actor SLA overrides and --fail-when-breached
    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--keywords", "acknowledged",
            "--keywords", "confirm",
            "--ack-actor", "Maintainer <2>",
            "--ack-actor", "Maintainer <3>",
            "--sla-hours", "2.5",
            "--ack-actor-sla", "Maintainer <2>=2.0",
            "--ack-actor-sla", "Maintainer <3>=6.0",
            "--fail-when-breached",
            "--output", str(output_dir)
        ],
        capture_output=True,
        text=True
    )

    # Should exit with code 0 because global SLA is NOT breached
    # (latest inbound is 1.0h from M3 which is < 2.5h global threshold)
    # Even though M2 has a per-actor breach, the global SLA governs --fail-when-breached
    assert result.returncode == 0, f"CLI should exit 0 (global SLA ok): {result.stderr}\nstdout: {result.stdout}"

    # Load JSON output
    json_path = output_dir / "inbox_scan_summary.json"
    assert json_path.exists(), "JSON summary not created"
    with open(json_path) as f:
        data = json.load(f)

    # === JSON Assertions ===
    # 1. ack_actor_summary is present
    assert "ack_actor_summary" in data, "JSON missing ack_actor_summary"
    summary = data["ack_actor_summary"]

    # 2. Bucket structure is correct (critical/warning/ok/unknown)
    assert "critical" in summary, "ack_actor_summary missing 'critical' bucket"
    assert "warning" in summary, "ack_actor_summary missing 'warning' bucket"
    assert "ok" in summary, "ack_actor_summary missing 'ok' bucket"
    assert "unknown" in summary, "ack_actor_summary missing 'unknown' bucket"

    # 3. M2 is in critical bucket (3.5 > 2.0 by >= 1 hour)
    critical_actors = summary["critical"]
    assert len(critical_actors) >= 1, "critical bucket should have at least 1 actor"
    m2_entry = next((e for e in critical_actors if e["actor_id"] == "maintainer_2"), None)
    assert m2_entry is not None, "maintainer_2 should be in critical bucket"
    assert m2_entry["actor_label"] == "Maintainer 2", f"M2 label wrong: {m2_entry['actor_label']}"
    assert m2_entry["sla_threshold_hours"] == 2.0, f"M2 threshold should be 2.0, got {m2_entry['sla_threshold_hours']}"
    assert m2_entry["sla_breached"] is True, "M2 should be breached"
    # Hours since inbound should be ~3.5 (allow tolerance for execution time)
    assert m2_entry["hours_since_inbound"] is not None, "M2 missing hours_since_inbound"
    assert m2_entry["hours_since_inbound"] >= 3.4, f"M2 hours should be ~3.5, got {m2_entry['hours_since_inbound']}"

    # 4. M3 is in ok bucket (1.0 < 6.0)
    ok_actors = summary["ok"]
    assert len(ok_actors) >= 1, "ok bucket should have at least 1 actor"
    m3_entry = next((e for e in ok_actors if e["actor_id"] == "maintainer_3"), None)
    assert m3_entry is not None, "maintainer_3 should be in ok bucket"
    assert m3_entry["actor_label"] == "Maintainer 3", f"M3 label wrong: {m3_entry['actor_label']}"
    assert m3_entry["sla_threshold_hours"] == 6.0, f"M3 threshold should be 6.0, got {m3_entry['sla_threshold_hours']}"
    assert m3_entry["sla_breached"] is False, "M3 should NOT be breached"
    # Hours since inbound should be ~1.0 (allow tolerance)
    assert m3_entry["hours_since_inbound"] is not None, "M3 missing hours_since_inbound"
    assert m3_entry["hours_since_inbound"] >= 0.9, f"M3 hours should be ~1.0, got {m3_entry['hours_since_inbound']}"

    # === Markdown Assertions ===
    md_path = output_dir / "inbox_scan_summary.md"
    assert md_path.exists(), "Markdown summary not created"
    md_content = md_path.read_text()

    # 5. Markdown includes "## Ack Actor SLA Summary"
    assert "## Ack Actor SLA Summary" in md_content, \
        "Markdown missing '## Ack Actor SLA Summary' section"

    # Check severity subsections appear
    assert "### Critical" in md_content, \
        "Markdown missing '### Critical' subsection in Ack Actor SLA Summary"
    assert "### OK" in md_content, \
        "Markdown missing '### OK' subsection in Ack Actor SLA Summary"

    # Check actor names appear in their sections
    assert "Maintainer 2" in md_content, \
        "Markdown should mention Maintainer 2 in summary"
    assert "Maintainer 3" in md_content, \
        "Markdown should mention Maintainer 3 in summary"

    # === CLI stdout Assertions ===
    # 6. CLI stdout includes "Ack Actor SLA Summary:"
    assert "Ack Actor SLA Summary:" in result.stdout, \
        "CLI stdout should show 'Ack Actor SLA Summary:'"

    # Check severity labels in stdout
    assert "[CRITICAL]" in result.stdout, \
        "CLI stdout should show [CRITICAL] severity bucket"
    assert "[OK]" in result.stdout, \
        "CLI stdout should show [OK] severity bucket"

    # Check actor names appear in stdout
    assert "Maintainer 2" in result.stdout, \
        "CLI stdout should mention Maintainer 2 in summary"
    assert "Maintainer 3" in result.stdout, \
        "CLI stdout should mention Maintainer 3 in summary"


def test_ack_actor_history_tracks_severity(tmp_path):
    """
    Test that --history-jsonl and --history-markdown persist ack_actor_summary.

    Creates a synthetic inbox with:
    - Maintainer <2> inbound 3.5 hours ago (no ack keywords) → critical breach with 2.0h threshold
    - Maintainer <3> absent (no inbound) → unknown

    Runs the CLI with --sla-hours 2.5, per-actor overrides, and history flags, then asserts:
    1. JSONL entry contains ack_actor_summary with critical[0]["actor_id"] == "maintainer_2"
    2. JSONL entry contains ack_actor_summary with unknown[0]["actor_id"] == "maintainer_3"
    3. Markdown row contains "[CRITICAL] Maintainer 2"
    4. Markdown row contains "[UNKNOWN] Maintainer 3"
    5. Markdown table has "Ack Actor Severity" column header
    """
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    history_jsonl = tmp_path / "history.jsonl"
    history_md = tmp_path / "history.md"

    # Create message from Maintainer <2> (3.5 hours ago, no ack keywords)
    m2_content = """# Request: dose_experiments_ground_truth

**From:** Maintainer <2>
**To:** Maintainer <1>
**Date:** 2026-01-22T10:00:00Z

Please provide the dose experiments ground truth bundle.
"""
    create_inbox_file(inbox_dir, "request_m2_dose_experiments_ground_truth.md", m2_content, -3.5)

    # NO message from Maintainer <3> - they should show as "unknown"

    # Run CLI with both M2 and M3 as ack actors, with per-actor SLA overrides and history flags
    result = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--keywords", "acknowledged",
            "--keywords", "confirm",
            "--ack-actor", "Maintainer <2>",
            "--ack-actor", "Maintainer <3>",
            "--sla-hours", "2.5",
            "--ack-actor-sla", "Maintainer <2>=2.0",
            "--ack-actor-sla", "Maintainer <3>=6.0",
            "--history-jsonl", str(history_jsonl),
            "--history-markdown", str(history_md),
            "--output", str(output_dir)
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"CLI failed: {result.stderr}\nstdout: {result.stdout}"

    # === JSONL Assertions ===
    assert history_jsonl.exists(), "JSONL history file not created"
    with open(history_jsonl) as f:
        lines = [json.loads(line) for line in f.readlines()]

    assert len(lines) >= 1, "Expected at least 1 JSONL entry"
    entry = lines[-1]  # Get the most recent entry

    # 1. JSONL contains ack_actor_summary
    assert "ack_actor_summary" in entry, \
        "JSONL entry missing ack_actor_summary"
    summary = entry["ack_actor_summary"]

    # 2. Verify critical bucket has maintainer_2
    assert "critical" in summary, "ack_actor_summary missing 'critical' bucket"
    critical_actors = summary["critical"]
    assert len(critical_actors) >= 1, "critical bucket should have at least 1 actor"
    m2_entry = next((a for a in critical_actors if a.get("actor_id") == "maintainer_2"), None)
    assert m2_entry is not None, \
        f"maintainer_2 should be in critical bucket, got: {critical_actors}"

    # 3. Verify unknown bucket has maintainer_3
    assert "unknown" in summary, "ack_actor_summary missing 'unknown' bucket"
    unknown_actors = summary["unknown"]
    assert len(unknown_actors) >= 1, "unknown bucket should have at least 1 actor"
    m3_entry = next((a for a in unknown_actors if a.get("actor_id") == "maintainer_3"), None)
    assert m3_entry is not None, \
        f"maintainer_3 should be in unknown bucket, got: {unknown_actors}"

    # === Markdown Assertions ===
    assert history_md.exists(), "Markdown history file not created"
    md_content = history_md.read_text()

    # 4. Markdown table has "Ack Actor Severity" column header
    assert "Ack Actor Severity" in md_content, \
        "Markdown history table missing 'Ack Actor Severity' column header"

    # 5. Markdown row contains [CRITICAL] Maintainer 2
    assert "[CRITICAL] Maintainer 2" in md_content, \
        f"Markdown history should contain '[CRITICAL] Maintainer 2', got: {md_content}"

    # 6. Markdown row contains [UNKNOWN] Maintainer 3
    assert "[UNKNOWN] Maintainer 3" in md_content, \
        f"Markdown history should contain '[UNKNOWN] Maintainer 3', got: {md_content}"

    # Verify header was written exactly once
    assert md_content.count("# Inbox Scan History") == 1, \
        "Markdown header should appear exactly once"


def test_history_dashboard_actor_severity_trends(tmp_path):
    """
    Test --history-dashboard generates an "## Ack Actor Severity Trends" table.

    Creates a synthetic inbox with:
    - Maintainer <2> inbound 3.5 hours ago (no ack keywords) → critical breach with 2.0h threshold
    - Maintainer <3> absent (no inbound) → unknown

    Runs the CLI twice to accumulate history entries, then verifies:
    1. Dashboard includes "## Ack Actor Severity Trends" section
    2. Table shows Maintainer 2 with critical count of 2
    3. Table shows Maintainer 3 with unknown count of 2
    4. Table includes Longest Wait and Latest Scan columns
    5. Actors are sorted by severity priority (critical > unknown)
    """
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    history_jsonl = tmp_path / "history.jsonl"
    history_md = tmp_path / "history.md"
    dashboard_path = tmp_path / "dashboard.md"

    # Create message from Maintainer <2> (3.5 hours ago, no ack keywords)
    m2_content = """# Request: dose_experiments_ground_truth

**From:** Maintainer <2>
**To:** Maintainer <1>
**Date:** 2026-01-22T10:00:00Z

Please provide the dose experiments ground truth bundle.
"""
    create_inbox_file(inbox_dir, "request_m2_dose_experiments_ground_truth.md", m2_content, -3.5)

    # NO message from Maintainer <3> - they should show as "unknown"

    # === Run 1: First scan ===
    result1 = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--keywords", "acknowledged",
            "--keywords", "confirm",
            "--ack-actor", "Maintainer <2>",
            "--ack-actor", "Maintainer <3>",
            "--sla-hours", "2.5",
            "--ack-actor-sla", "Maintainer <2>=2.0",
            "--ack-actor-sla", "Maintainer <3>=6.0",
            "--history-jsonl", str(history_jsonl),
            "--history-markdown", str(history_md),
            "--history-dashboard", str(dashboard_path),
            "--output", str(output_dir / "run1")
        ],
        capture_output=True,
        text=True
    )
    assert result1.returncode == 0, f"Run 1 failed: {result1.stderr}\nstdout: {result1.stdout}"

    # === Run 2: Second scan (accumulate history) ===
    result2 = subprocess.run(
        [
            sys.executable, str(CLI_SCRIPT),
            "--inbox", str(inbox_dir),
            "--request-pattern", "dose_experiments_ground_truth",
            "--keywords", "acknowledged",
            "--keywords", "confirm",
            "--ack-actor", "Maintainer <2>",
            "--ack-actor", "Maintainer <3>",
            "--sla-hours", "2.5",
            "--ack-actor-sla", "Maintainer <2>=2.0",
            "--ack-actor-sla", "Maintainer <3>=6.0",
            "--history-jsonl", str(history_jsonl),
            "--history-markdown", str(history_md),
            "--history-dashboard", str(dashboard_path),
            "--output", str(output_dir / "run2")
        ],
        capture_output=True,
        text=True
    )
    assert result2.returncode == 0, f"Run 2 failed: {result2.stderr}\nstdout: {result2.stdout}"

    # === Validate JSONL has 2 entries ===
    assert history_jsonl.exists(), "JSONL history file not created"
    with open(history_jsonl) as f:
        jsonl_lines = [json.loads(line) for line in f.readlines()]
    assert len(jsonl_lines) == 2, f"Expected 2 JSONL entries, got {len(jsonl_lines)}"

    # === Validate Dashboard ===
    assert dashboard_path.exists(), "Dashboard file not created"
    dashboard_content = dashboard_path.read_text()

    # 1. Dashboard includes "## Ack Actor Severity Trends"
    assert "## Ack Actor Severity Trends" in dashboard_content, \
        "Dashboard missing '## Ack Actor Severity Trends' section"

    # 2. Table shows Maintainer 2 with critical count (2 scans = 2 critical entries)
    assert "Maintainer 2" in dashboard_content, \
        "Dashboard missing Maintainer 2 in severity trends table"
    # The table format: | Actor | Critical | Warning | OK | Unknown | Longest Wait | Latest Scan |
    # Maintainer 2 should have Critical = 2 (both scans showed critical breach)
    # Look for a row with Maintainer 2 and verify the critical count
    lines = dashboard_content.split("\n")
    m2_row = None
    for line in lines:
        if "Maintainer 2" in line and "|" in line:
            m2_row = line
            break
    assert m2_row is not None, "Could not find Maintainer 2 row in dashboard table"
    # Parse columns: | Actor | Critical | Warning | OK | Unknown | Longest Wait | Latest Scan |
    cols = [c.strip() for c in m2_row.split("|")]
    # cols[0] is empty (before first |), cols[1] is Actor, cols[2] is Critical, etc.
    if len(cols) >= 6:
        critical_count = cols[2]
        assert critical_count == "2", \
            f"Maintainer 2 should have Critical count of 2, got '{critical_count}'"

    # 3. Table shows Maintainer 3 with unknown count (2 scans = 2 unknown entries)
    assert "Maintainer 3" in dashboard_content, \
        "Dashboard missing Maintainer 3 in severity trends table"
    m3_row = None
    for line in lines:
        if "Maintainer 3" in line and "|" in line:
            m3_row = line
            break
    assert m3_row is not None, "Could not find Maintainer 3 row in dashboard table"
    cols3 = [c.strip() for c in m3_row.split("|")]
    if len(cols3) >= 6:
        unknown_count = cols3[5]  # Unknown is the 5th data column (index 5 after split)
        assert unknown_count == "2", \
            f"Maintainer 3 should have Unknown count of 2, got '{unknown_count}'"

    # 4. Table includes Longest Wait and Latest Scan columns
    assert "Longest Wait" in dashboard_content, \
        "Dashboard missing 'Longest Wait' column header"
    assert "Latest Scan" in dashboard_content, \
        "Dashboard missing 'Latest Scan' column header"

    # 5. Maintainer 2 appears before Maintainer 3 (critical sorted before unknown)
    m2_pos = dashboard_content.find("Maintainer 2")
    m3_pos = dashboard_content.find("Maintainer 3")
    assert m2_pos < m3_pos, \
        "Maintainer 2 (critical) should appear before Maintainer 3 (unknown) in the table"

    # 6. Check "Entries with per-actor data:" line
    assert "Entries with per-actor data:" in dashboard_content, \
        "Dashboard missing 'Entries with per-actor data:' line"
    assert "2 of 2" in dashboard_content, \
        "Dashboard should show '2 of 2' entries with per-actor data"
