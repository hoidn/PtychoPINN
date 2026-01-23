#!/usr/bin/env python3
"""
run_inbox_cadence.py - Unified cadence CLI for maintainer follow-up runs.

This non-production CLI orchestrates a single cadence run:
1. Creates a timestamped reports directory
2. Runs check_inbox_for_ack.py with all history/status/escalation outputs
3. Reads inbox_scan_summary.json to check ack status
4. Unless ack detected and --skip-followup-on-ack, runs update_maintainer_status.py
5. Emits cadence_metadata.json and cadence_summary.md

Exit codes:
- 0: Success (normal cadence run completed)
- 3: Success but ack detected and follow-up skipped (--skip-followup-on-ack)
- Non-zero: Failure

Usage:
    python run_inbox_cadence.py \
        --inbox inbox \
        --request-pattern dose_experiments_ground_truth \
        --keywords acknowledged --keywords confirm \
        --ack-actor "Maintainer <2>" --ack-actor "Maintainer <3>" \
        --ack-actor-sla "Maintainer <2>=2.0" --ack-actor-sla "Maintainer <3>=6.0" \
        --sla-hours 2.5 --fail-when-breached \
        --output-root plans/active/DEBUG-SIM-LINES-DOSE-001/reports \
        --timestamp 2026-01-23T163500Z \
        --response-path inbox/response_dose_experiments_ground_truth.md \
        --followup-dir inbox \
        --followup-prefix followup_dose_experiments_ground_truth \
        --status-title "Maintainer Status Automation" \
        --to "Maintainer <2>" --cc "Maintainer <3>" \
        --escalation-brief-recipient "Maintainer <3>" \
        --escalation-brief-target "Maintainer <2>"

Author: Ralph (DEBUG-SIM-LINES-DOSE-001.F1)
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified cadence CLI for maintainer follow-up runs"
    )
    # Input paths
    parser.add_argument(
        "--inbox",
        type=str,
        required=True,
        help="Path to inbox directory to scan"
    )
    parser.add_argument(
        "--request-pattern",
        type=str,
        required=True,
        help="Pattern to match in filename or content"
    )
    parser.add_argument(
        "--keywords",
        action="append",
        default=[],
        help="Keywords to detect acknowledgement (can be repeated)"
    )
    parser.add_argument(
        "--ack-actor",
        action="append",
        dest="ack_actors",
        default=[],
        help="Actor(s) whose messages count as acknowledgements (can be repeated)"
    )
    parser.add_argument(
        "--ack-actor-sla",
        action="append",
        dest="ack_actor_sla",
        default=[],
        help="Per-actor SLA threshold override in 'actor=hours' format"
    )
    parser.add_argument(
        "--sla-hours",
        type=float,
        default=None,
        help="SLA threshold in hours"
    )
    parser.add_argument(
        "--fail-when-breached",
        action="store_true",
        help="Exit with error if SLA is breached"
    )

    # Output paths
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Root directory for reports (timestamp subdir will be created)"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Override timestamp for reproducibility (ISO8601, e.g., 2026-01-23T163500Z)"
    )
    parser.add_argument(
        "--response-path",
        type=str,
        required=True,
        help="Path to response document to append status block to"
    )
    parser.add_argument(
        "--followup-dir",
        type=str,
        required=True,
        help="Directory to write follow-up note"
    )
    parser.add_argument(
        "--followup-prefix",
        type=str,
        required=True,
        help="Prefix for follow-up filename"
    )
    parser.add_argument(
        "--status-title",
        type=str,
        default="Maintainer Status Update",
        help="Title for the status block"
    )
    parser.add_argument(
        "--to",
        type=str,
        default=None,
        help="Primary recipient for the follow-up note"
    )
    parser.add_argument(
        "--cc",
        type=str,
        default=None,
        help="CC recipient for the follow-up note"
    )
    parser.add_argument(
        "--escalation-brief-recipient",
        type=str,
        default="Maintainer <3>",
        help="Recipient for the escalation brief"
    )
    parser.add_argument(
        "--escalation-brief-target",
        type=str,
        default="Maintainer <2>",
        help="The blocking actor being escalated"
    )

    # Behavior flags
    parser.add_argument(
        "--skip-followup-on-ack",
        action="store_true",
        help="Skip update_maintainer_status.py if ack detected (exit 3)"
    )

    return parser.parse_args()


def get_timestamp(override: Optional[str]) -> str:
    """Get timestamp, either from override or generate current UTC."""
    if override:
        return override
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def run_check_inbox(
    args,
    timestamp: str,
    output_dir: Path,
    logs_dir: Path,
    history_jsonl_path: Path,
    history_md_path: Path,
    history_dashboard_path: Path,
    status_snippet_path: Path,
    escalation_note_path: Path,
    escalation_brief_path: Path
) -> int:
    """
    Run check_inbox_for_ack.py with full artifact generation.

    Returns the exit code from the subprocess.
    """
    # Find the CLI script (relative to this file)
    script_dir = Path(__file__).parent
    check_inbox_script = script_dir / "check_inbox_for_ack.py"

    if not check_inbox_script.exists():
        print(f"ERROR: check_inbox_for_ack.py not found at {check_inbox_script}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable, str(check_inbox_script),
        "--inbox", args.inbox,
        "--request-pattern", args.request_pattern,
        "--output", str(output_dir / "inbox_sla_watch"),
    ]

    # Add keywords
    for kw in args.keywords:
        cmd.extend(["--keywords", kw])

    # Add ack actors
    for actor in args.ack_actors:
        cmd.extend(["--ack-actor", actor])

    # Add per-actor SLA overrides
    for override in args.ack_actor_sla:
        cmd.extend(["--ack-actor-sla", override])

    # Add SLA threshold
    if args.sla_hours is not None:
        cmd.extend(["--sla-hours", str(args.sla_hours)])

    if args.fail_when_breached:
        cmd.append("--fail-when-breached")

    # Add history/status/escalation outputs
    cmd.extend(["--history-jsonl", str(history_jsonl_path)])
    cmd.extend(["--history-markdown", str(history_md_path)])
    cmd.extend(["--history-dashboard", str(history_dashboard_path)])
    cmd.extend(["--status-snippet", str(status_snippet_path)])
    cmd.extend(["--escalation-note", str(escalation_note_path)])
    cmd.extend(["--escalation-recipient", args.escalation_brief_target])
    cmd.extend(["--escalation-brief", str(escalation_brief_path)])
    cmd.extend(["--escalation-brief-recipient", args.escalation_brief_recipient])
    cmd.extend(["--escalation-brief-target", args.escalation_brief_target])

    # Log the command
    log_path = logs_dir / "check_inbox_for_ack.log"
    print(f"Running check_inbox_for_ack.py, logging to: {log_path}")

    with open(log_path, "w", encoding="utf-8") as log_file:
        result = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )

    return result.returncode


def run_update_maintainer_status(
    args,
    timestamp: str,
    output_dir: Path,
    logs_dir: Path,
    scan_json_path: Path,
    followup_path: Path
) -> int:
    """
    Run update_maintainer_status.py to append status block and create follow-up.

    Returns the exit code from the subprocess.
    """
    script_dir = Path(__file__).parent
    update_status_script = script_dir / "update_maintainer_status.py"

    if not update_status_script.exists():
        print(f"ERROR: update_maintainer_status.py not found at {update_status_script}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable, str(update_status_script),
        "--scan-json", str(scan_json_path),
        "--status-title", args.status_title,
        "--response-path", args.response_path,
        "--followup-path", str(followup_path),
    ]

    if args.to:
        cmd.extend(["--to", args.to])
    if args.cc:
        cmd.extend(["--cc", args.cc])

    # Add artifact references
    artifact_paths = [
        output_dir / "inbox_sla_watch" / "inbox_scan_summary.md",
        output_dir / "inbox_history" / "inbox_history_dashboard.md",
        output_dir / "inbox_status" / "status_snippet.md",
        output_dir / "inbox_status" / "escalation_note.md",
    ]

    # Add escalation brief if it exists
    escalation_brief_path = output_dir / "inbox_status" / f"escalation_brief_{args.escalation_brief_recipient.replace(' ', '_').replace('<', '').replace('>', '')}.md"
    if escalation_brief_path.exists():
        artifact_paths.append(escalation_brief_path)

    for artifact in artifact_paths:
        if artifact.exists():
            cmd.extend(["--artifact", str(artifact)])

    # Log the command
    log_path = logs_dir / "update_maintainer_status.log"
    print(f"Running update_maintainer_status.py, logging to: {log_path}")

    with open(log_path, "w", encoding="utf-8") as log_file:
        result = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )

    return result.returncode


def main():
    """Main entry point."""
    args = parse_args()

    # Generate or use provided timestamp
    timestamp = get_timestamp(args.timestamp)

    # Create directory structure
    output_root = Path(args.output_root)
    output_dir = output_root / timestamp
    logs_dir = output_dir / "logs"
    inbox_sla_watch_dir = output_dir / "inbox_sla_watch"
    inbox_history_dir = output_dir / "inbox_history"
    inbox_status_dir = output_dir / "inbox_status"

    for d in [logs_dir, inbox_sla_watch_dir, inbox_history_dir, inbox_status_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Derived paths for check_inbox_for_ack.py outputs
    history_jsonl_path = inbox_history_dir / "inbox_sla_watch.jsonl"
    history_md_path = inbox_history_dir / "inbox_history.md"
    history_dashboard_path = inbox_history_dir / "inbox_history_dashboard.md"
    status_snippet_path = inbox_status_dir / "status_snippet.md"
    escalation_note_path = inbox_status_dir / "escalation_note.md"
    escalation_brief_path = inbox_status_dir / f"escalation_brief_{args.escalation_brief_recipient.replace(' ', '_').replace('<', '').replace('>', '')}.md"

    # Followup path
    followup_dir = Path(args.followup_dir)
    followup_dir.mkdir(parents=True, exist_ok=True)
    followup_path = followup_dir / f"{args.followup_prefix}_{timestamp}.md"

    print(f"=== Inbox Cadence Run: {timestamp} ===")
    print(f"Output directory: {output_dir}")
    print(f"Request pattern: {args.request_pattern}")
    print(f"Response path: {args.response_path}")
    print(f"Follow-up path: {followup_path}")
    print("")

    # Step 1: Run check_inbox_for_ack.py
    print("Step 1: Scanning inbox for acknowledgements...")
    check_inbox_exit = run_check_inbox(
        args=args,
        timestamp=timestamp,
        output_dir=output_dir,
        logs_dir=logs_dir,
        history_jsonl_path=history_jsonl_path,
        history_md_path=history_md_path,
        history_dashboard_path=history_dashboard_path,
        status_snippet_path=status_snippet_path,
        escalation_note_path=escalation_note_path,
        escalation_brief_path=escalation_brief_path
    )

    # Load scan results
    scan_json_path = inbox_sla_watch_dir / "inbox_scan_summary.json"
    ack_detected = False
    ack_actor_summary = None

    if scan_json_path.exists():
        try:
            with open(scan_json_path, "r", encoding="utf-8") as f:
                scan_data = json.load(f)
            ack_detected = scan_data.get("ack_detected", False)
            ack_actor_summary = scan_data.get("ack_actor_summary")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"WARNING: Failed to parse scan JSON: {e}", file=sys.stderr)
    else:
        print(f"WARNING: Scan JSON not found at {scan_json_path}", file=sys.stderr)

    print(f"  Check inbox exit code: {check_inbox_exit}")
    print(f"  Acknowledgement detected: {ack_detected}")
    print("")

    # Determine if we should skip follow-up
    followup_written = False
    status_appended = False
    exit_code = 0

    if ack_detected and args.skip_followup_on_ack:
        # Step 2: Skip follow-up
        print("Step 2: Ack detected and --skip-followup-on-ack set, skipping follow-up...")
        print(f"  No follow-up written to: {followup_path}")
        exit_code = 3
    else:
        # Step 2: Run update_maintainer_status.py
        print("Step 2: Updating maintainer status and writing follow-up...")
        update_status_exit = run_update_maintainer_status(
            args=args,
            timestamp=timestamp,
            output_dir=output_dir,
            logs_dir=logs_dir,
            scan_json_path=scan_json_path,
            followup_path=followup_path
        )

        if update_status_exit == 0:
            followup_written = followup_path.exists()
            status_appended = True
            print(f"  Update maintainer status exit code: {update_status_exit}")
            print(f"  Follow-up written: {followup_written}")
        else:
            print(f"  ERROR: update_maintainer_status.py failed with exit code {update_status_exit}", file=sys.stderr)
            exit_code = update_status_exit

    # Step 3: Write cadence metadata
    print("")
    print("Step 3: Writing cadence metadata...")

    cadence_metadata = {
        "timestamp": timestamp,
        "ack_detected": ack_detected,
        "ack_actor_summary": ack_actor_summary,
        "followup_written": followup_written,
        "status_appended": status_appended,
        "followup_path": str(followup_path) if followup_written else None,
        "check_inbox_exit_code": check_inbox_exit,
        "parameters": {
            "inbox": args.inbox,
            "request_pattern": args.request_pattern,
            "keywords": args.keywords,
            "ack_actors": args.ack_actors,
            "ack_actor_sla": args.ack_actor_sla,
            "sla_hours": args.sla_hours,
            "fail_when_breached": args.fail_when_breached,
            "response_path": args.response_path,
            "followup_dir": args.followup_dir,
            "followup_prefix": args.followup_prefix,
            "status_title": args.status_title,
            "to": args.to,
            "cc": args.cc,
            "skip_followup_on_ack": args.skip_followup_on_ack,
        },
        "artifact_paths": {
            "output_dir": str(output_dir),
            "logs_dir": str(logs_dir),
            "scan_json": str(scan_json_path),
            "history_jsonl": str(history_jsonl_path),
            "history_dashboard": str(history_dashboard_path),
            "status_snippet": str(status_snippet_path),
            "escalation_note": str(escalation_note_path),
            "escalation_brief": str(escalation_brief_path),
        }
    }

    metadata_json_path = output_dir / "cadence_metadata.json"
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(cadence_metadata, f, indent=2)
    print(f"  Metadata written to: {metadata_json_path}")

    # Write cadence summary markdown
    cadence_summary_path = output_dir / "cadence_summary.md"
    with open(cadence_summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Cadence Summary: {timestamp}\n\n")
        f.write(f"**Request Pattern:** {args.request_pattern}\n")
        f.write(f"**Acknowledgement Detected:** {'Yes' if ack_detected else 'No'}\n")
        f.write(f"**Follow-up Written:** {'Yes' if followup_written else 'No'}\n")
        if followup_written:
            f.write(f"**Follow-up Path:** `{followup_path}`\n")
        f.write(f"**Exit Code:** {exit_code}\n\n")
        f.write("## Artifacts\n\n")
        f.write(f"- Scan JSON: `{scan_json_path}`\n")
        f.write(f"- History JSONL: `{history_jsonl_path}`\n")
        f.write(f"- History Dashboard: `{history_dashboard_path}`\n")
        f.write(f"- Status Snippet: `{status_snippet_path}`\n")
        f.write(f"- Escalation Note: `{escalation_note_path}`\n")
        f.write(f"- Escalation Brief: `{escalation_brief_path}`\n")
        f.write(f"- Logs: `{logs_dir}/`\n")
    print(f"  Summary written to: {cadence_summary_path}")

    print("")
    print(f"=== Cadence Run Complete: Exit Code {exit_code} ===")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
