#!/usr/bin/env python3
"""
update_maintainer_status.py - CLI to automate maintainer status updates.

This non-production CLI reads inbox_scan_summary.json and generates:
1. A Markdown status block appended to the response document
2. A follow-up note with SLA metrics and artifact references

Usage:
    python update_maintainer_status.py \
        --scan-json path/to/inbox_scan_summary.json \
        --status-title "Maintainer Status Automation" \
        --artifact path/to/artifact1.md \
        --artifact path/to/artifact2.md \
        --response-path inbox/response_dose_experiments_ground_truth.md \
        --followup-path inbox/followup_dose_experiments_ground_truth_2026-01-23T153500Z.md \
        --to "Maintainer <2>" --cc "Maintainer <3>"

Author: Ralph (DEBUG-SIM-LINES-DOSE-001)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Automate maintainer status updates from inbox scan results"
    )
    parser.add_argument(
        "--scan-json",
        type=str,
        required=True,
        help="Path to inbox_scan_summary.json from check_inbox_for_ack.py"
    )
    parser.add_argument(
        "--status-title",
        type=str,
        default="Maintainer Status Update",
        help="Title for the status block (default: Maintainer Status Update)"
    )
    parser.add_argument(
        "--artifact",
        action="append",
        dest="artifacts",
        default=[],
        help="Path to artifact file to reference (can be repeated)"
    )
    parser.add_argument(
        "--response-path",
        type=str,
        required=True,
        help="Path to response document to append status block to"
    )
    parser.add_argument(
        "--followup-path",
        type=str,
        required=True,
        help="Path to write the follow-up note"
    )
    parser.add_argument(
        "--to",
        type=str,
        default=None,
        help="Primary recipient for the follow-up note (e.g., 'Maintainer <2>')"
    )
    parser.add_argument(
        "--cc",
        type=str,
        default=None,
        help="CC recipient for the follow-up note (e.g., 'Maintainer <3>')"
    )
    return parser.parse_args()


def _format_hours(hours: Optional[float]) -> str:
    """Format hours value for display, handling None."""
    if hours is None:
        return "---"
    return f"{hours:.2f}"


def _format_timestamp(ts: Optional[str]) -> str:
    """Format timestamp for display, handling None."""
    if ts is None:
        return "---"
    # Parse ISO format and convert to compact display
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, AttributeError):
        return ts if ts else "---"


def _render_ack_actor_table(ack_actor_stats: dict) -> str:
    """
    Render the Ack Actor table showing per-actor SLA metrics.

    Args:
        ack_actor_stats: Dict mapping actor_id to stats from inbox_scan_summary.json

    Returns:
        Markdown table string
    """
    if not ack_actor_stats:
        return "| Actor | Status |\n|-------|--------|\n| (no actors configured) | --- |"

    lines = [
        "| Actor | Hrs Since Inbound | Hrs Since Outbound | Inbound Count | Outbound Count | SLA Threshold | Severity | Notes |",
        "|-------|-------------------|--------------------|--------------:|---------------:|--------------:|----------|-------|"
    ]

    for actor_id, stats in sorted(ack_actor_stats.items()):
        actor_label = actor_id.replace("_", " ").title()
        hrs_inbound = _format_hours(stats.get("hours_since_last_inbound"))
        hrs_outbound = _format_hours(stats.get("hours_since_last_outbound"))
        inbound_count = stats.get("inbound_count", 0)
        outbound_count = stats.get("outbound_count", 0)
        sla_threshold = _format_hours(stats.get("sla_threshold_hours"))
        severity = stats.get("sla_severity", "unknown").upper()
        notes = stats.get("sla_notes", "---")

        lines.append(
            f"| {actor_label} | {hrs_inbound} | {hrs_outbound} | {inbound_count} | {outbound_count} | {sla_threshold} | {severity} | {notes} |"
        )

    return "\n".join(lines)


def _render_global_sla_table(sla_watch: dict) -> str:
    """
    Render the global SLA watch metrics table.

    Args:
        sla_watch: Dict from inbox_scan_summary.json sla_watch field

    Returns:
        Markdown table string
    """
    if not sla_watch:
        return "| Metric | Value |\n|--------|-------|\n| (no SLA data) | --- |"

    threshold = _format_hours(sla_watch.get("threshold_hours"))
    hrs_inbound = _format_hours(sla_watch.get("hours_since_last_inbound"))
    breached = "Yes" if sla_watch.get("breached") else "No"
    severity = sla_watch.get("severity", "unknown").upper()
    notes = sla_watch.get("notes", "---")
    deadline = _format_timestamp(sla_watch.get("deadline_utc"))
    breach_duration = _format_hours(sla_watch.get("breach_duration_hours"))

    lines = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| SLA Threshold | {threshold} hours |",
        f"| Hours Since Last Inbound | {hrs_inbound} |",
        f"| SLA Breached | **{breached}** |",
        f"| Severity | {severity} |",
        f"| SLA Deadline | {deadline} |",
        f"| Breach Duration | {breach_duration} hours |",
        f"| Notes | {notes} |"
    ]

    return "\n".join(lines)


def _build_status_block(scan_data: dict, status_title: str, artifacts: list, generated_utc: str) -> str:
    """
    Build the Markdown status block to append to response document.

    Args:
        scan_data: Parsed inbox_scan_summary.json
        status_title: Title for the status block
        artifacts: List of artifact paths to reference
        generated_utc: UTC timestamp for this status update

    Returns:
        Markdown string for the status block
    """
    # Format timestamp for heading
    try:
        dt = datetime.fromisoformat(generated_utc.replace("Z", "+00:00"))
        ts_display = dt.strftime("%Y-%m-%dT%H:%MZ")
    except (ValueError, AttributeError):
        ts_display = generated_utc

    ack_detected = scan_data.get("ack_detected", False)
    ack_status = "Yes" if ack_detected else "No"

    waiting_clock = scan_data.get("waiting_clock", {})
    total_inbound = waiting_clock.get("total_inbound_count", 0)
    total_outbound = waiting_clock.get("total_outbound_count", 0)

    ack_actor_stats = scan_data.get("ack_actor_stats", {})
    sla_watch = scan_data.get("sla_watch", {})

    lines = [
        f"### Status as of {ts_display} ({status_title})",
        "",
        f"**Acknowledgement Detected:** {ack_status}",
        f"**Total Inbound Messages:** {total_inbound}",
        f"**Total Outbound Messages:** {total_outbound}",
        "",
        "**Global SLA Watch:**",
        "",
        _render_global_sla_table(sla_watch),
        "",
        "**Per-Actor SLA Status:**",
        "",
        _render_ack_actor_table(ack_actor_stats),
        ""
    ]

    if artifacts:
        lines.append("**Artifacts:**")
        lines.append("")
        for artifact in artifacts:
            lines.append(f"- `{artifact}`")
        lines.append("")

    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def _build_followup_note(
    scan_data: dict,
    generated_utc: str,
    artifacts: list,
    to_recipient: Optional[str],
    cc_recipient: Optional[str]
) -> str:
    """
    Build the follow-up note Markdown document.

    Args:
        scan_data: Parsed inbox_scan_summary.json
        generated_utc: UTC timestamp for this follow-up
        artifacts: List of artifact paths to reference
        to_recipient: Primary recipient
        cc_recipient: CC recipient

    Returns:
        Markdown string for the follow-up note
    """
    # Format timestamp
    try:
        dt = datetime.fromisoformat(generated_utc.replace("Z", "+00:00"))
        ts_display = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        ts_short = dt.strftime("%Y-%m-%dT%H%M%SZ")
    except (ValueError, AttributeError):
        ts_display = generated_utc
        ts_short = generated_utc

    request_pattern = scan_data.get("parameters", {}).get("request_pattern", "unknown")
    ack_detected = scan_data.get("ack_detected", False)
    ack_status = "Yes" if ack_detected else "No"

    waiting_clock = scan_data.get("waiting_clock", {})
    total_inbound = waiting_clock.get("total_inbound_count", 0)
    total_outbound = waiting_clock.get("total_outbound_count", 0)

    ack_actor_stats = scan_data.get("ack_actor_stats", {})
    sla_watch = scan_data.get("sla_watch", {})

    # Build header
    lines = [
        f"# Follow-up: {request_pattern} ({ts_short})",
        "",
        "**From:** Maintainer <1>",
    ]

    if to_recipient:
        lines.append(f"**To:** {to_recipient}")
    if cc_recipient:
        lines.append(f"**CC:** {cc_recipient}")

    lines.extend([
        f"**Date:** {ts_display}",
        f"**Re:** {request_pattern}",
        "",
        "---",
        "",
        "## Status Update",
        "",
        f"This is an automated follow-up regarding the `{request_pattern}` request.",
        "",
        "### Summary Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Acknowledgement Detected | {ack_status} |",
        f"| Total Inbound Messages | {total_inbound} |",
        f"| Total Outbound Messages | {total_outbound} |",
        "",
        "### Global SLA Watch",
        "",
        _render_global_sla_table(sla_watch),
        "",
        "### Per-Actor SLA Status",
        "",
        _render_ack_actor_table(ack_actor_stats),
        ""
    ])

    # Add ack actor follow-up activity section
    if ack_actor_stats:
        lines.extend([
            "### Ack Actor Follow-Up Activity",
            "",
            "| Actor | Last Outbound (UTC) | Hours Since Outbound | Outbound Count |",
            "|-------|---------------------|----------------------|---------------:|",
        ])

        for actor_id, stats in sorted(ack_actor_stats.items()):
            actor_label = actor_id.replace("_", " ").title()
            last_outbound = _format_timestamp(stats.get("last_outbound_utc"))
            hrs_outbound = _format_hours(stats.get("hours_since_last_outbound"))
            outbound_count = stats.get("outbound_count", 0)
            lines.append(f"| {actor_label} | {last_outbound} | {hrs_outbound} | {outbound_count} |")

        lines.append("")

    # Add artifacts section
    if artifacts:
        lines.extend([
            "### Artifacts",
            "",
        ])
        for artifact in artifacts:
            lines.append(f"- `{artifact}`")
        lines.append("")

    # Add action items
    lines.extend([
        "### Action Items",
        "",
        "- [ ] Confirm receipt of delivered artifacts",
        "- [ ] Verify SHA256 checksums match",
        "- [ ] Report any issues or blockers",
        "",
        "---",
        "",
        "Thank you,",
        "Maintainer <1>",
        ""
    ])

    return "\n".join(lines)


def main():
    """Main entry point."""
    args = parse_args()

    # Validate scan-json exists
    scan_json_path = Path(args.scan_json)
    if not scan_json_path.exists():
        print(f"ERROR: Scan JSON not found: {args.scan_json}", file=sys.stderr)
        sys.exit(1)

    # Load scan data
    try:
        with open(scan_json_path, "r", encoding="utf-8") as f:
            scan_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {args.scan_json}: {e}", file=sys.stderr)
        sys.exit(1)

    # Get current UTC timestamp
    now_utc = datetime.now(timezone.utc).isoformat()

    # Build status block
    status_block = _build_status_block(
        scan_data=scan_data,
        status_title=args.status_title,
        artifacts=args.artifacts,
        generated_utc=now_utc
    )

    # Build follow-up note
    followup_note = _build_followup_note(
        scan_data=scan_data,
        generated_utc=now_utc,
        artifacts=args.artifacts,
        to_recipient=args.to,
        cc_recipient=args.cc
    )

    # Append status block to response document
    response_path = Path(args.response_path)
    if not response_path.exists():
        print(f"ERROR: Response document not found: {args.response_path}", file=sys.stderr)
        sys.exit(1)

    # Ensure parent directories exist for follow-up note
    followup_path = Path(args.followup_path)
    followup_path.parent.mkdir(parents=True, exist_ok=True)

    # Append to response document
    with open(response_path, "a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(status_block)

    # Write follow-up note
    with open(followup_path, "w", encoding="utf-8") as f:
        f.write(followup_note)

    # Log summary
    print(f"Status block appended to: {args.response_path}")
    print(f"Follow-up note written to: {args.followup_path}")
    print(f"Status title: {args.status_title}")
    print(f"Artifacts referenced: {len(args.artifacts)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
