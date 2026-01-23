#!/usr/bin/env python3
"""
check_inbox_for_ack.py - CLI to scan inbox directories for maintainer acknowledgements.

This non-production CLI scans an inbox directory for files referencing a given
request pattern, detects acknowledgement keywords, tracks a timeline of
inbound/outbound messages, computes waiting-clock metrics, and emits JSON/Markdown
summaries.

Usage:
    python check_inbox_for_ack.py \\
        --inbox inbox \\
        --request-pattern dose_experiments_ground_truth \\
        --keywords acknowledged --keywords confirm --keywords received \\
        --output $ARTIFACT_DIR/inbox_check

Author: Ralph (DEBUG-SIM-LINES-DOSE-001)
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Scan an inbox directory for acknowledgement files"
    )
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
        help="Pattern to match in filename or content (e.g., dose_experiments_ground_truth)"
    )
    parser.add_argument(
        "--keywords",
        action="append",
        default=[],
        help="Keywords to detect acknowledgement (can be repeated)"
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="ISO8601 date filter (only consider files modified since this date)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for summary files"
    )
    parser.add_argument(
        "--sla-hours",
        type=float,
        default=None,
        help="SLA threshold in hours; if set, compute SLA watch metrics"
    )
    parser.add_argument(
        "--fail-when-breached",
        action="store_true",
        help="Exit with code 2 if SLA is breached and ack not yet detected"
    )
    parser.add_argument(
        "--history-jsonl",
        type=str,
        default=None,
        help="Path to JSONL file for appending scan history entries"
    )
    parser.add_argument(
        "--history-markdown",
        type=str,
        default=None,
        help="Path to Markdown file for appending scan history table rows"
    )
    parser.add_argument(
        "--status-snippet",
        type=str,
        default=None,
        help="Path to write a Markdown status snippet summarizing current wait state"
    )
    parser.add_argument(
        "--escalation-note",
        type=str,
        default=None,
        help="Path to write a Markdown escalation note when SLA is breached"
    )
    parser.add_argument(
        "--escalation-recipient",
        type=str,
        default="Maintainer <2>",
        help="Recipient name for the escalation note (default: Maintainer <2>)"
    )
    parser.add_argument(
        "--history-dashboard",
        type=str,
        default=None,
        help="Path to write a Markdown history dashboard (requires --history-jsonl)"
    )
    parser.add_argument(
        "--ack-actor",
        action="append",
        dest="ack_actors",
        default=[],
        help="Actor(s) whose messages count as acknowledgements (can be repeated). "
             "Default: Maintainer <2> only. Use --ack-actor 'Maintainer <3>' to add more."
    )
    parser.add_argument(
        "--ack-actor-sla",
        action="append",
        dest="ack_actor_sla",
        default=[],
        help="Per-actor SLA threshold override in 'actor=hours' format (can be repeated). "
             "E.g., --ack-actor-sla 'Maintainer <2>=2.0' --ack-actor-sla 'Maintainer <3>=6.0'. "
             "Actors without overrides inherit the global --sla-hours threshold."
    )
    parser.add_argument(
        "--escalation-brief",
        type=str,
        default=None,
        help="Path to write a Markdown escalation brief for a third-party escalation recipient"
    )
    parser.add_argument(
        "--escalation-brief-recipient",
        type=str,
        default="Maintainer <3>",
        help="Recipient for the escalation brief (default: Maintainer <3>)"
    )
    parser.add_argument(
        "--escalation-brief-target",
        type=str,
        default="Maintainer <2>",
        help="The blocking actor being escalated (default: Maintainer <2>)"
    )
    return parser.parse_args()


def read_file_tolerant(path: Path) -> str:
    """Read file with tolerant UTF-8 decoding (errors='ignore')."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def file_metadata(path: Path) -> dict:
    """Get file metadata (size, modified_utc)."""
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    }


def normalize_actor_alias(actor_str: str) -> str:
    """
    Normalize actor aliases to canonical form (maintainer_1, maintainer_2, maintainer_3).

    Accepts:
    - "Maintainer <1>", "maintainer <1>", "maintainer_1", etc. -> "maintainer_1"
    - "Maintainer <2>", "maintainer <2>", "maintainer_2", etc. -> "maintainer_2"
    - "Maintainer <3>", "maintainer <3>", "maintainer_3", etc. -> "maintainer_3"

    Returns the canonical form or the original string lowercased if not recognized.
    """
    s = actor_str.strip().lower()

    # Match patterns like "maintainer <N>", "maintainer_n", "maintainer N"
    # Allow flexible spacing around angle brackets
    m = re.match(r"maintainer\s*[<_]?\s*(\d+)\s*>?", s)
    if m:
        return f"maintainer_{m.group(1)}"

    return s


def parse_ack_actor_sla_overrides(sla_override_strings: list[str]) -> dict[str, float]:
    """
    Parse per-actor SLA override strings into a normalized {actor: hours} map.

    Args:
        sla_override_strings: List of strings like "Maintainer <2>=2.0", "maintainer_3=6.0"

    Returns:
        Dictionary mapping normalized actor IDs to SLA threshold hours.
        E.g., {"maintainer_2": 2.0, "maintainer_3": 6.0}
    """
    overrides = {}
    for override_str in sla_override_strings:
        if "=" not in override_str:
            continue  # Skip malformed entries

        parts = override_str.split("=", 1)
        if len(parts) != 2:
            continue

        actor_raw, hours_str = parts
        actor_id = normalize_actor_alias(actor_raw.strip())

        try:
            hours = float(hours_str.strip())
            overrides[actor_id] = hours
        except ValueError:
            continue  # Skip malformed hours

    return overrides


def detect_actor_and_direction(content: str) -> Tuple[str, str]:
    """
    Detect which maintainer authored the message and the communication direction.

    Returns (actor, direction) where:
    - actor: "maintainer_1", "maintainer_2", "maintainer_3", or "unknown"
    - direction: "inbound" (from M2/M3 to M1), "outbound" (from M1 to others), or "unknown"

    Looks for patterns like:
    - "From: Maintainer <1>" / "**From:** Maintainer <1>"
    - "From: Maintainer <2>" / "**From:** Maintainer <2>"
    - "From: Maintainer <3>" / "**From:** Maintainer <3>"
    """
    content_lower = content.lower()

    # Patterns for Maintainer <1>
    m1_patterns = [
        r"from:\s*\*?\*?maintainer\s*<\s*1\s*>",
        r"\*\*from:\*\*\s*maintainer\s*<\s*1\s*>",
    ]

    # Patterns for Maintainer <2>
    m2_patterns = [
        r"from:\s*\*?\*?maintainer\s*<\s*2\s*>",
        r"\*\*from:\*\*\s*maintainer\s*<\s*2\s*>",
    ]

    # Patterns for Maintainer <3>
    m3_patterns = [
        r"from:\s*\*?\*?maintainer\s*<\s*3\s*>",
        r"\*\*from:\*\*\s*maintainer\s*<\s*3\s*>",
    ]

    for pattern in m1_patterns:
        if re.search(pattern, content_lower):
            return ("maintainer_1", "outbound")

    for pattern in m2_patterns:
        if re.search(pattern, content_lower):
            return ("maintainer_2", "inbound")

    for pattern in m3_patterns:
        if re.search(pattern, content_lower):
            return ("maintainer_3", "inbound")

    return ("unknown", "unknown")


def is_from_maintainer_2(content: str) -> bool:
    """
    Check if the message is FROM Maintainer <2>.

    Looks for patterns like:
    - "From: Maintainer <2>"
    - "**From:** Maintainer <2>"
    """
    actor, _ = detect_actor_and_direction(content)
    return actor == "maintainer_2"


def is_acknowledgement(
    content: str,
    keywords: list[str],
    ack_actors: list[str] | None = None
) -> tuple[bool, list[str], str]:
    """
    Check if content contains acknowledgement keywords from a configured ack actor.

    An acknowledgement is detected when:
    - The message is FROM an actor in ack_actors (defaults to ["maintainer_2"])
    - AND contains any of the user-provided keywords (no hidden hard-coded list)

    Args:
        content: Message content to analyze
        keywords: User-provided keywords to detect (e.g., ["acknowledged", "confirm"])
        ack_actors: Normalized actor IDs that count as ack sources (e.g., ["maintainer_2", "maintainer_3"])

    Returns (ack_detected, keyword_hits, actor).
    """
    # Default to Maintainer <2> only
    if ack_actors is None:
        ack_actors = ["maintainer_2"]

    content_lower = content.lower()
    keyword_hits = []

    for kw in keywords:
        if kw.lower() in content_lower:
            keyword_hits.append(kw)

    # Detect actor from content
    actor, _ = detect_actor_and_direction(content)

    # Acknowledgement requires: message FROM an ack_actor AND at least one keyword hit
    actor_is_ack_source = actor in ack_actors
    ack_detected = actor_is_ack_source and len(keyword_hits) > 0

    return ack_detected, keyword_hits, actor


def truncate_preview(content: str, max_chars: int = 320) -> str:
    """Truncate content to max_chars for preview, adding ellipsis if truncated."""
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "..."


def compute_hours_since(timestamp_iso: Optional[str], now: datetime) -> Optional[float]:
    """Compute hours elapsed since the given ISO timestamp."""
    if timestamp_iso is None:
        return None
    ts = datetime.fromisoformat(timestamp_iso)
    delta = now - ts
    return round(delta.total_seconds() / 3600, 2)


def scan_inbox(
    inbox_path: Path,
    request_pattern: str,
    keywords: list[str],
    since_date: Optional[datetime],
    sla_hours: Optional[float] = None,
    current_time: Optional[datetime] = None,
    ack_actors: list[str] | None = None,
    ack_actor_sla_overrides: dict[str, float] | None = None
) -> dict:
    """
    Scan inbox directory for files matching the request pattern.

    Returns a summary dict with scanned/matches/ack_detected status,
    plus timeline and waiting-clock metadata.

    If sla_hours is provided, also computes sla_watch metrics.
    current_time can be injected for testing; defaults to now(UTC).
    ack_actors is a list of normalized actor IDs (e.g., ["maintainer_2", "maintainer_3"])
    that are considered valid acknowledgement sources.
    ack_actor_sla_overrides is a dict mapping normalized actor IDs to their specific
    SLA thresholds in hours (e.g., {"maintainer_2": 2.0}). Actors without overrides
    inherit the global sla_hours value.
    """
    # Normalize ack_actors, default to ["maintainer_2"]
    if ack_actors is None:
        ack_actors = ["maintainer_2"]
    else:
        ack_actors = [normalize_actor_alias(a) for a in ack_actors]

    # Ensure ack_actor_sla_overrides is a dict (empty dict means no overrides)
    if ack_actor_sla_overrides is None:
        ack_actor_sla_overrides = {}

    now = current_time if current_time is not None else datetime.now(timezone.utc)
    results = {
        "scanned": 0,
        "matches": [],
        "ack_detected": False,
        "ack_files": [],
        "generated_utc": now.isoformat(),
        "parameters": {
            "inbox": str(inbox_path),
            "request_pattern": request_pattern,
            "keywords": keywords,
            "since": since_date.isoformat() if since_date else None,
            "ack_actors": ack_actors,
            "ack_actor_sla_hours": ack_actor_sla_overrides if ack_actor_sla_overrides else None
        },
        # Timeline: list of entries sorted by modified_utc ascending
        "timeline": [],
        # Waiting clock metrics
        "waiting_clock": {
            "last_inbound_utc": None,
            "last_outbound_utc": None,
            "hours_since_last_inbound": None,
            "hours_since_last_outbound": None,
            "total_inbound_count": 0,
            "total_outbound_count": 0,
        }
    }

    if not inbox_path.exists():
        print(f"WARNING: Inbox path does not exist: {inbox_path}")
        return results

    pattern_lower = request_pattern.lower()

    # Collect all matches first, then sort by modified_utc
    raw_matches = []

    # Scan all files in inbox (non-recursive by default for safety)
    for item in sorted(inbox_path.iterdir()):
        # Skip directories
        if item.is_dir():
            continue

        results["scanned"] += 1

        # Get metadata
        meta = file_metadata(item)

        # Apply since filter
        if since_date:
            file_mtime = datetime.fromisoformat(meta["modified_utc"])
            if file_mtime < since_date:
                continue

        # Check filename match
        filename_match = pattern_lower in item.name.lower()

        # Read content and check content match
        content = read_file_tolerant(item)
        content_match = pattern_lower in content.lower()

        if not (filename_match or content_match):
            continue

        # We have a match - analyze it
        ack_detected, keyword_hits, actor = is_acknowledgement(content, keywords, ack_actors)
        _, direction = detect_actor_and_direction(content)

        # Compute is_from_ack_actor for backwards compatibility in output
        is_from_ack_actor = actor in ack_actors

        match_entry = {
            "file": item.name,
            "path": str(item),
            "size_bytes": meta["size_bytes"],
            "modified_utc": meta["modified_utc"],
            "match_reason": [],
            "keywords_found": keyword_hits,
            "is_from_maintainer_2": actor == "maintainer_2",  # Backwards compat
            "is_from_ack_actor": is_from_ack_actor,
            "actor": actor,
            "direction": direction,
            "ack_detected": ack_detected,
            "preview": truncate_preview(content)
        }

        if filename_match:
            match_entry["match_reason"].append("filename")
        if content_match:
            match_entry["match_reason"].append("content")

        raw_matches.append(match_entry)

        if ack_detected:
            results["ack_detected"] = True
            results["ack_files"].append(item.name)

    # Sort matches by modified_utc ascending for timeline
    raw_matches.sort(key=lambda x: x["modified_utc"])
    results["matches"] = raw_matches

    # Build timeline and compute waiting clock
    last_inbound_utc = None
    last_outbound_utc = None
    inbound_count = 0
    outbound_count = 0

    for m in raw_matches:
        timeline_entry = {
            "timestamp_utc": m["modified_utc"],
            "file": m["file"],
            "actor": m["actor"],
            "direction": m["direction"],
            "ack": m["ack_detected"],
            "keywords": m["keywords_found"],
        }
        results["timeline"].append(timeline_entry)

        if m["direction"] == "inbound":
            last_inbound_utc = m["modified_utc"]
            inbound_count += 1
        elif m["direction"] == "outbound":
            last_outbound_utc = m["modified_utc"]
            outbound_count += 1

    results["waiting_clock"]["last_inbound_utc"] = last_inbound_utc
    results["waiting_clock"]["last_outbound_utc"] = last_outbound_utc
    results["waiting_clock"]["hours_since_last_inbound"] = compute_hours_since(last_inbound_utc, now)
    results["waiting_clock"]["hours_since_last_outbound"] = compute_hours_since(last_outbound_utc, now)
    results["waiting_clock"]["total_inbound_count"] = inbound_count
    results["waiting_clock"]["total_outbound_count"] = outbound_count

    # Compute per-actor wait metrics (ack_actor_stats)
    # Tracks last_inbound_utc, hours_since_last_inbound, inbound_count, ack_files per ack_actor
    # Also computes per-actor SLA fields when sla_hours is provided
    ack_actor_stats = {}
    for actor_id in sorted(ack_actors):  # Sort for determinism
        actor_messages = [m for m in raw_matches if m["actor"] == actor_id]
        actor_inbound = [m for m in actor_messages if m["direction"] == "inbound"]
        actor_ack_files = [m["file"] for m in actor_messages if m["ack_detected"]]

        last_inbound_from_actor = None
        if actor_inbound:
            # Sort by timestamp and get the latest
            actor_inbound_sorted = sorted(actor_inbound, key=lambda x: x["modified_utc"])
            last_inbound_from_actor = actor_inbound_sorted[-1]["modified_utc"]

        hours_since = compute_hours_since(last_inbound_from_actor, now)
        actor_ack_detected = len(actor_ack_files) > 0

        actor_stats = {
            "last_inbound_utc": last_inbound_from_actor,
            "hours_since_last_inbound": hours_since,
            "inbound_count": len(actor_inbound),
            "ack_files": actor_ack_files,
            "ack_detected": actor_ack_detected,
        }

        # Compute per-actor SLA fields when sla_hours is provided
        if sla_hours is not None:
            # Use per-actor override if available, otherwise fall back to global threshold
            actor_threshold = ack_actor_sla_overrides.get(actor_id, sla_hours)
            actor_stats["sla_threshold_hours"] = actor_threshold

            if last_inbound_from_actor is None or hours_since is None:
                # No inbound messages from this actor
                actor_stats["sla_deadline_utc"] = None
                actor_stats["sla_breached"] = False
                actor_stats["sla_breach_duration_hours"] = None
                actor_stats["sla_severity"] = "unknown"
                actor_stats["sla_notes"] = f"No inbound messages from {actor_id.replace('_', ' ').title()}"
            else:
                # Compute SLA deadline: last_inbound + actor_threshold
                from datetime import timedelta
                last_inbound_ts = datetime.fromisoformat(last_inbound_from_actor)
                deadline_ts = last_inbound_ts + timedelta(hours=actor_threshold)
                actor_stats["sla_deadline_utc"] = deadline_ts.isoformat()

                # Compute breach status for this actor using their specific threshold
                actor_breached = hours_since > actor_threshold and not actor_ack_detected

                if actor_breached:
                    breach_duration = round(hours_since - actor_threshold, 2)
                    if breach_duration < 1.0:
                        severity = "warning"
                    else:
                        severity = "critical"
                    notes = f"SLA breach: {hours_since:.2f} hours since last inbound exceeds {actor_threshold:.2f} hour threshold"
                elif actor_ack_detected:
                    breach_duration = 0.0
                    severity = "ok"
                    notes = "Acknowledgement received"
                else:
                    breach_duration = 0.0
                    severity = "ok"
                    notes = f"Within SLA: {hours_since:.2f} hours since last inbound (threshold: {actor_threshold:.2f})"

                actor_stats["sla_breached"] = actor_breached
                actor_stats["sla_breach_duration_hours"] = breach_duration
                actor_stats["sla_severity"] = severity
                actor_stats["sla_notes"] = notes

        ack_actor_stats[actor_id] = actor_stats

    results["ack_actor_stats"] = ack_actor_stats

    # Build ack_actor_summary: group actors by severity (critical/warning/ok/unknown)
    # This provides a concise overview of which actors are breaching SLA vs within SLA
    if sla_hours is not None:
        ack_actor_summary = {
            "critical": [],
            "warning": [],
            "ok": [],
            "unknown": []
        }
        for actor_id in sorted(ack_actor_stats.keys()):
            stats = ack_actor_stats[actor_id]
            severity = stats.get("sla_severity", "unknown")
            entry = {
                "actor_id": actor_id,
                "actor_label": actor_id.replace("_", " ").title(),
                "hours_since_inbound": stats.get("hours_since_last_inbound"),
                "sla_threshold_hours": stats.get("sla_threshold_hours"),
                "sla_deadline_utc": stats.get("sla_deadline_utc"),
                "sla_breached": stats.get("sla_breached", False),
                "sla_notes": stats.get("sla_notes", ""),
            }
            if severity in ack_actor_summary:
                ack_actor_summary[severity].append(entry)
            else:
                ack_actor_summary["unknown"].append(entry)
        results["ack_actor_summary"] = ack_actor_summary

    # SLA Watch computation (if sla_hours provided)
    if sla_hours is not None:
        hours_since = results["waiting_clock"]["hours_since_last_inbound"]
        last_inbound_utc = results["waiting_clock"]["last_inbound_utc"]
        breached = False
        notes = ""
        sla_deadline_utc = None
        breach_duration_hours = None
        severity = "unknown"

        if hours_since is None or last_inbound_utc is None:
            # No inbound messages at all
            notes = "No inbound messages from Maintainer <2> found"
            breached = False  # Cannot be breached if no inbound exists
            severity = "unknown"
        else:
            # Compute SLA deadline: last_inbound + sla_hours
            from datetime import timedelta
            last_inbound_ts = datetime.fromisoformat(last_inbound_utc)
            deadline_ts = last_inbound_ts + timedelta(hours=sla_hours)
            sla_deadline_utc = deadline_ts.isoformat()

            breached = hours_since > sla_hours and not results["ack_detected"]

            if breached:
                # Compute breach duration
                breach_duration_hours = round(hours_since - sla_hours, 2)
                # Determine severity: warning (<1 hour late), critical (>=1 hour late)
                if breach_duration_hours < 1.0:
                    severity = "warning"
                else:
                    severity = "critical"
                notes = f"SLA breach: {hours_since:.2f} hours since last inbound exceeds {sla_hours:.2f} hour threshold and no acknowledgement detected"
            elif results["ack_detected"]:
                notes = "Acknowledgement received; SLA not applicable"
                severity = "ok"
                breach_duration_hours = 0.0
            else:
                notes = f"Within SLA: {hours_since:.2f} hours since last inbound (threshold: {sla_hours:.2f})"
                severity = "ok"
                breach_duration_hours = 0.0

        results["sla_watch"] = {
            "threshold_hours": sla_hours,
            "hours_since_last_inbound": hours_since,
            "breached": breached,
            "notes": notes,
            "deadline_utc": sla_deadline_utc,
            "breach_duration_hours": breach_duration_hours,
            "severity": severity
        }

    return results


def write_json_summary(results: dict, output_path: Path) -> None:
    """Write JSON summary to file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def write_markdown_summary(results: dict, output_path: Path) -> None:
    """Write Markdown summary to file."""
    wc = results.get("waiting_clock", {})

    # Get ack_actors from parameters (may be missing in older results)
    ack_actors = results['parameters'].get('ack_actors', ['maintainer_2'])
    ack_actors_str = ', '.join(f'`{a}`' for a in ack_actors)

    lines = [
        "# Inbox Scan Summary",
        "",
        f"**Generated:** {results['generated_utc']}",
        "",
        "## Parameters",
        "",
        f"- **Inbox:** `{results['parameters']['inbox']}`",
        f"- **Request Pattern:** `{results['parameters']['request_pattern']}`",
        f"- **Keywords:** {', '.join(f'`{k}`' for k in results['parameters']['keywords'])}",
        f"- **Ack Actors:** {ack_actors_str}",
        f"- **Since Filter:** {results['parameters']['since'] or 'None'}",
        "",
        "## Summary",
        "",
        f"- **Files Scanned:** {results['scanned']}",
        f"- **Matches Found:** {len(results['matches'])}",
        f"- **Acknowledgement Detected:** {'Yes' if results['ack_detected'] else 'No'}",
        "",
    ]

    # Waiting Clock section
    lines.extend([
        "## Waiting Clock",
        "",
    ])
    last_inbound = wc.get("last_inbound_utc") or "N/A"
    last_outbound = wc.get("last_outbound_utc") or "N/A"
    hours_in = wc.get("hours_since_last_inbound")
    hours_out = wc.get("hours_since_last_outbound")
    hours_in_str = f"{hours_in:.2f} hours" if hours_in is not None else "N/A"
    hours_out_str = f"{hours_out:.2f} hours" if hours_out is not None else "N/A"

    lines.extend([
        f"- **Last Inbound (from Maintainer <2>):** {last_inbound}",
        f"- **Hours Since Last Inbound:** {hours_in_str}",
        f"- **Last Outbound (from Maintainer <1>):** {last_outbound}",
        f"- **Hours Since Last Outbound:** {hours_out_str}",
        f"- **Total Inbound Messages:** {wc.get('total_inbound_count', 0)}",
        f"- **Total Outbound Messages:** {wc.get('total_outbound_count', 0)}",
        "",
    ])

    if not results["ack_detected"]:
        lines.extend([
            "> **Note:** No acknowledgement from Maintainer <2> found. The bundle has been delivered",
            "> but we are still awaiting confirmation from the receiving maintainer.",
            "",
        ])

    # SLA Watch section (if present)
    sla_watch = results.get("sla_watch")
    if sla_watch:
        lines.extend([
            "## SLA Watch",
            "",
            f"- **Threshold:** {sla_watch['threshold_hours']:.2f} hours",
        ])
        hrs = sla_watch.get("hours_since_last_inbound")
        hrs_str = f"{hrs:.2f} hours" if hrs is not None else "N/A"
        deadline_utc = sla_watch.get("deadline_utc") or "N/A"
        breach_dur = sla_watch.get("breach_duration_hours")
        breach_dur_str = f"{breach_dur:.2f} hours" if breach_dur is not None else "N/A"
        severity = sla_watch.get("severity", "unknown")
        lines.extend([
            f"- **Hours Since Last Inbound:** {hrs_str}",
            f"- **Deadline (UTC):** {deadline_utc}",
            f"- **Breached:** {'Yes' if sla_watch['breached'] else 'No'}",
            f"- **Breach Duration:** {breach_dur_str}",
            f"- **Severity:** {severity}",
            f"- **Notes:** {sla_watch['notes']}",
            "",
        ])
        if sla_watch["breached"]:
            lines.extend([
                "> **SLA BREACH:** The waiting time has exceeded the configured threshold and no acknowledgement has been received.",
                "",
            ])

    # Ack Actor SLA Summary section (groups actors by severity)
    ack_actor_summary = results.get("ack_actor_summary")
    if ack_actor_summary:
        lines.extend([
            "## Ack Actor SLA Summary",
            "",
            "Per-actor SLA status grouped by severity:",
            "",
        ])
        # Emit severity categories in order: critical, warning, ok, unknown
        severity_order = ["critical", "warning", "ok", "unknown"]
        severity_labels = {
            "critical": "Critical (Breach >= 1h)",
            "warning": "Warning (Breach < 1h)",
            "ok": "OK (Within SLA)",
            "unknown": "Unknown (No Inbound)"
        }
        for sev in severity_order:
            actors_in_sev = ack_actor_summary.get(sev, [])
            if actors_in_sev:
                lines.append(f"### {severity_labels[sev]}")
                lines.append("")
                for entry in actors_in_sev:
                    hrs = entry.get("hours_since_inbound")
                    hrs_str = f"{hrs:.2f}" if hrs is not None else "N/A"
                    threshold = entry.get("sla_threshold_hours")
                    threshold_str = f"{threshold:.2f}" if threshold is not None else "N/A"
                    breached = "Yes" if entry.get("sla_breached", False) else "No"
                    notes = sanitize_for_markdown(entry.get("sla_notes", "-"))
                    lines.append(f"- **{entry['actor_label']}**: {hrs_str} hrs since inbound (threshold: {threshold_str} hrs) — Breached: {breached}")
                    if entry.get("sla_notes"):
                        lines.append(f"  - {notes}")
                lines.append("")

    # Ack Actor Coverage table (per-actor wait metrics with SLA fields)
    ack_actor_stats = results.get("ack_actor_stats", {})
    if ack_actor_stats:
        # Check if any actor has SLA fields (i.e., sla_hours was provided)
        has_sla = any("sla_deadline_utc" in stats for stats in ack_actor_stats.values())
        # Check if any actor has per-actor threshold overrides
        has_thresholds = any("sla_threshold_hours" in stats for stats in ack_actor_stats.values())

        lines.extend([
            "## Ack Actor Coverage",
            "",
            "Per-actor wait metrics for monitored acknowledgement actors:",
            "",
        ])

        if has_sla:
            # Include SLA columns when --sla-hours was used
            if has_thresholds:
                # Include Threshold column when per-actor SLA thresholds are tracked
                lines.extend([
                    "| Actor | Hours Since Inbound | Threshold (hrs) | Inbound Count | Ack | Deadline (UTC) | Breached | Severity | Notes |",
                    "|-------|---------------------|-----------------|---------------|-----|----------------|----------|----------|-------|",
                ])
            else:
                lines.extend([
                    "| Actor | Hours Since Inbound | Inbound Count | Ack | Deadline (UTC) | Breached | Severity | Notes |",
                    "|-------|---------------------|---------------|-----|----------------|----------|----------|-------|",
                ])
            for actor_id in sorted(ack_actor_stats.keys()):
                stats = ack_actor_stats[actor_id]
                actor_label = actor_id.replace("_", " ").title()
                hrs = stats.get("hours_since_last_inbound")
                hrs_str = f"{hrs:.2f}" if hrs is not None else "N/A"
                inbound_count = stats.get("inbound_count", 0)
                ack_detected = "Yes" if stats.get("ack_detected", False) else "No"
                deadline = stats.get("sla_deadline_utc") or "N/A"
                breached = "Yes" if stats.get("sla_breached", False) else "No"
                severity = stats.get("sla_severity", "unknown")
                notes = sanitize_for_markdown(stats.get("sla_notes", "-"))
                if has_thresholds:
                    threshold = stats.get("sla_threshold_hours")
                    threshold_str = f"{threshold:.2f}" if threshold is not None else "N/A"
                    lines.append(f"| {actor_label} | {hrs_str} | {threshold_str} | {inbound_count} | {ack_detected} | {deadline} | {breached} | {severity} | {notes} |")
                else:
                    lines.append(f"| {actor_label} | {hrs_str} | {inbound_count} | {ack_detected} | {deadline} | {breached} | {severity} | {notes} |")
        else:
            # Original columns without SLA
            lines.extend([
                "| Actor | Last Inbound (UTC) | Hours Since Inbound | Inbound Count | Ack Detected | Ack Files |",
                "|-------|-------------------|---------------------|---------------|--------------|-----------|",
            ])
            for actor_id in sorted(ack_actor_stats.keys()):
                stats = ack_actor_stats[actor_id]
                actor_label = actor_id.replace("_", " ").title()
                last_inbound = stats.get("last_inbound_utc") or "N/A"
                hrs = stats.get("hours_since_last_inbound")
                hrs_str = f"{hrs:.2f}" if hrs is not None else "N/A"
                inbound_count = stats.get("inbound_count", 0)
                ack_detected = "Yes" if stats.get("ack_detected", False) else "No"
                ack_files = ", ".join(f"`{f}`" for f in stats.get("ack_files", [])) if stats.get("ack_files") else "-"
                lines.append(f"| {actor_label} | {last_inbound} | {hrs_str} | {inbound_count} | {ack_detected} | {ack_files} |")
        lines.append("")

    if results["ack_detected"]:
        lines.extend([
            "### Acknowledgement Files",
            "",
        ])
        for f in results["ack_files"]:
            lines.append(f"- `{f}`")
        lines.append("")

    # Timeline section
    timeline = results.get("timeline", [])
    if timeline:
        lines.extend([
            "## Timeline",
            "",
            "Messages sorted by timestamp (ascending):",
            "",
            "| Timestamp (UTC) | Actor | Direction | Ack | Keywords | File |",
            "|-----------------|-------|-----------|-----|----------|------|",
        ])
        for t in timeline:
            actor_label = t["actor"].replace("_", " ").title() if t["actor"] != "unknown" else "Unknown"
            direction_label = t["direction"].capitalize() if t["direction"] != "unknown" else "Unknown"
            ack_label = "Yes" if t["ack"] else "No"
            kw_label = ", ".join(t["keywords"]) if t["keywords"] else "-"
            lines.append(f"| {t['timestamp_utc']} | {actor_label} | {direction_label} | {ack_label} | {kw_label} | `{t['file']}` |")
        lines.append("")

    if results["matches"]:
        lines.extend([
            "## Matching Files",
            "",
            "| File | Match Reason | Actor | Direction | Keywords | Ack | Modified |",
            "|------|--------------|-------|-----------|----------|-----|----------|",
        ])
        for m in results["matches"]:
            reason = ", ".join(m["match_reason"])
            actor_label = m.get("actor", "unknown").replace("_", " ").title()
            direction_label = m.get("direction", "unknown").capitalize()
            keywords = ", ".join(m["keywords_found"]) if m["keywords_found"] else "-"
            ack = "Yes" if m["ack_detected"] else "No"
            lines.append(f"| `{m['file']}` | {reason} | {actor_label} | {direction_label} | {keywords} | {ack} | {m['modified_utc']} |")

        lines.append("")
        lines.extend([
            "## File Previews",
            "",
        ])
        for m in results["matches"]:
            lines.extend([
                f"### {m['file']}",
                "",
                "```",
                m["preview"],
                "```",
                "",
            ])
    else:
        lines.extend([
            "## Matching Files",
            "",
            "*No files matched the request pattern.*",
            "",
        ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def sanitize_for_markdown(text: str) -> str:
    """Sanitize text for Markdown table cells by removing/escaping special chars."""
    if text is None:
        return ""
    # Replace pipe and newlines that would break Markdown tables
    text = str(text).replace("|", "\\|").replace("\n", " ").replace("\r", "")
    return text.strip()


def append_history_jsonl(results: dict, output_path: Path) -> None:
    """
    Append a single history entry to a JSONL file.

    Each line captures: generated_utc, ack_detected, hours_since_inbound,
    hours_since_outbound, sla_breached, ack_files, and ack_actor_summary.
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wc = results.get("waiting_clock", {})
    sla = results.get("sla_watch", {})

    entry = {
        "generated_utc": results.get("generated_utc"),
        "ack_detected": results.get("ack_detected", False),
        "hours_since_inbound": wc.get("hours_since_last_inbound"),
        "hours_since_outbound": wc.get("hours_since_last_outbound"),
        "sla_breached": sla.get("breached") if sla else None,
        "sla_threshold_hours": sla.get("threshold_hours") if sla else None,
        "sla_deadline_utc": sla.get("deadline_utc") if sla else None,
        "sla_breach_duration_hours": sla.get("breach_duration_hours") if sla else None,
        "sla_severity": sla.get("severity") if sla else None,
        "ack_files": results.get("ack_files", []),
        "total_matches": len(results.get("matches", [])),
        "total_inbound": wc.get("total_inbound_count", 0),
        "total_outbound": wc.get("total_outbound_count", 0),
    }

    # Include ack_actor_summary if present (per-actor severity grouping)
    ack_actor_summary = results.get("ack_actor_summary")
    if ack_actor_summary:
        entry["ack_actor_summary"] = ack_actor_summary

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _format_ack_actor_severity_cell(ack_actor_summary: dict | None) -> str:
    """
    Format the ack_actor_summary into a string for Markdown table cell.

    Returns entries like "[CRITICAL] Maintainer 2 (4.20h > 2.00h); [UNKNOWN] Maintainer 3"
    joined by "; " to fit in a single cell without breaking the table.
    """
    if not ack_actor_summary:
        return "-"

    entries = []
    severity_order = ["critical", "warning", "ok", "unknown"]
    for sev in severity_order:
        actors_in_sev = ack_actor_summary.get(sev, [])
        for actor in actors_in_sev:
            label = actor.get("actor_label", "Unknown")
            hrs = actor.get("hours_since_inbound")
            threshold = actor.get("sla_threshold_hours")

            if sev == "unknown" or hrs is None:
                entry_str = f"[{sev.upper()}] {label}"
            else:
                hrs_str = f"{hrs:.2f}" if hrs is not None else "N/A"
                threshold_str = f"{threshold:.2f}" if threshold is not None else "N/A"
                entry_str = f"[{sev.upper()}] {label} ({hrs_str}h > {threshold_str}h)"

            entries.append(entry_str)

    if not entries:
        return "-"

    # Join with <br> for Markdown line breaks within cell
    return "<br>".join(entries)


def append_history_markdown(results: dict, output_path: Path) -> None:
    """
    Append a single history row to a Markdown file.

    If the file doesn't exist or is empty, write the header first.
    Table columns: Generated (UTC) | Ack Detected | Hours Since Inbound |
                   Hours Since Outbound | SLA Breach | Severity | Ack Actor Severity | Ack Files
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wc = results.get("waiting_clock", {})
    sla = results.get("sla_watch", {})

    # Check if we need to write header
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with open(output_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("# Inbox Scan History\n\n")
            f.write("| Generated (UTC) | Ack | Hrs Inbound | Hrs Outbound | SLA Breach | Severity | Ack Actor Severity | Ack Files |\n")
            f.write("|-----------------|-----|-------------|--------------|------------|----------|--------------------|-----------|\n")

        # Format values
        gen_utc = results.get("generated_utc", "")[:19]  # Trim to readable length
        ack = "Yes" if results.get("ack_detected", False) else "No"

        hrs_in = wc.get("hours_since_last_inbound")
        hrs_in_str = f"{hrs_in:.2f}" if hrs_in is not None else "N/A"

        hrs_out = wc.get("hours_since_last_outbound")
        hrs_out_str = f"{hrs_out:.2f}" if hrs_out is not None else "N/A"

        breach = sla.get("breached") if sla else None
        breach_str = "Yes" if breach else ("No" if breach is False else "N/A")

        severity = sla.get("severity") if sla else "N/A"

        # Format per-actor severity summary
        ack_actor_summary = results.get("ack_actor_summary")
        actor_severity_str = _format_ack_actor_severity_cell(ack_actor_summary)
        # Sanitize pipes/newlines to not break the table
        actor_severity_str = sanitize_for_markdown(actor_severity_str)

        ack_files = results.get("ack_files", [])
        ack_files_str = ", ".join(sanitize_for_markdown(f) for f in ack_files) if ack_files else "-"

        f.write(f"| {gen_utc} | {ack} | {hrs_in_str} | {hrs_out_str} | {breach_str} | {severity} | {actor_severity_str} | {ack_files_str} |\n")


def write_status_snippet(
    results: dict,
    output_path: Path,
    breach_timeline_lines: list = None
) -> None:
    """
    Write a concise Markdown status snippet summarizing current wait state.

    The snippet includes:
    - Generated timestamp
    - Ack status (detected or waiting)
    - Hours since inbound/outbound
    - SLA threshold and breach status (if applicable)
    - Ack files (if any)
    - Notes
    - Condensed timeline table
    - Ack Actor Breach Timeline (if breach_timeline_lines provided)

    Args:
        results: Scan results dictionary
        output_path: Path to write the snippet
        breach_timeline_lines: Optional list of Markdown lines from
            _build_actor_breach_timeline_section(), added when --history-jsonl
            is provided. Skipped entirely when None (history logging disabled).

    The output file is overwritten (not appended) to provide a single,
    idempotent snapshot of current status.
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wc = results.get("waiting_clock", {})
    sla = results.get("sla_watch", {})
    timeline = results.get("timeline", [])

    lines = [
        "# Maintainer Status Snapshot",
        "",
        f"**Generated:** {results.get('generated_utc', 'N/A')}",
        "",
    ]

    # Ack status
    if results.get("ack_detected", False):
        ack_files = results.get("ack_files", [])
        ack_files_str = ", ".join(f"`{f}`" for f in ack_files) if ack_files else "None"
        lines.extend([
            "## Ack Detected: Yes",
            "",
            f"**Ack Files:** {ack_files_str}",
            "",
        ])
    else:
        # Build list of monitored actors
        ack_actors = results.get("parameters", {}).get("ack_actors", ["maintainer_2"])
        actors_str = ", ".join(a.replace("_", " ").title() for a in ack_actors)
        lines.extend([
            "## Ack Detected: No",
            "",
            f"Waiting for acknowledgement from monitored actor(s): {actors_str}.",
            "",
        ])

    # Waiting clock metrics
    hrs_in = wc.get("hours_since_last_inbound")
    hrs_out = wc.get("hours_since_last_outbound")
    hrs_in_str = f"{hrs_in:.2f}" if hrs_in is not None else "N/A"
    hrs_out_str = f"{hrs_out:.2f}" if hrs_out is not None else "N/A"

    lines.extend([
        "## Wait Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Hours since last inbound | {hrs_in_str} |",
        f"| Hours since last outbound | {hrs_out_str} |",
        f"| Total inbound messages | {wc.get('total_inbound_count', 0)} |",
        f"| Total outbound messages | {wc.get('total_outbound_count', 0)} |",
        "",
    ])

    # SLA watch (if applicable)
    if sla:
        threshold = sla.get("threshold_hours")
        breached = sla.get("breached", False)
        notes = sla.get("notes", "")
        threshold_str = f"{threshold:.2f}" if threshold is not None else "N/A"
        breach_str = "Yes" if breached else "No"
        deadline_utc = sla.get("deadline_utc") or "N/A"
        breach_dur = sla.get("breach_duration_hours")
        breach_dur_str = f"{breach_dur:.2f}" if breach_dur is not None else "N/A"
        severity = sla.get("severity", "unknown")

        lines.extend([
            "## SLA Watch",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Threshold | {threshold_str} hours |",
            f"| Deadline (UTC) | {deadline_utc} |",
            f"| Breached | {breach_str} |",
            f"| Breach Duration | {breach_dur_str} hours |",
            f"| Severity | {severity} |",
            "",
            f"**Notes:** {sanitize_for_markdown(notes)}",
            "",
        ])

        if breached:
            lines.extend([
                "> **SLA BREACH:** Action required.",
                "",
            ])

    # Ack Actor SLA Summary section (groups actors by severity)
    ack_actor_summary = results.get("ack_actor_summary")
    if ack_actor_summary:
        lines.extend([
            "## Ack Actor SLA Summary",
            "",
        ])
        severity_order = ["critical", "warning", "ok", "unknown"]
        severity_labels = {
            "critical": "Critical (Breach >= 1h)",
            "warning": "Warning (Breach < 1h)",
            "ok": "OK (Within SLA)",
            "unknown": "Unknown (No Inbound)"
        }
        for sev in severity_order:
            actors_in_sev = ack_actor_summary.get(sev, [])
            if actors_in_sev:
                lines.append(f"### {severity_labels[sev]}")
                lines.append("")
                for entry in actors_in_sev:
                    hrs = entry.get("hours_since_inbound")
                    hrs_str = f"{hrs:.2f}" if hrs is not None else "N/A"
                    threshold = entry.get("sla_threshold_hours")
                    threshold_str = f"{threshold:.2f}" if threshold is not None else "N/A"
                    breached_str = "Yes" if entry.get("sla_breached", False) else "No"
                    lines.append(f"- **{entry['actor_label']}**: {hrs_str} hrs (threshold: {threshold_str} hrs) — Breached: {breached_str}")
                lines.append("")

    # Ack Actor Coverage table (per-actor wait metrics with SLA fields)
    ack_actor_stats = results.get("ack_actor_stats", {})
    if ack_actor_stats:
        # Check if any actor has SLA fields
        has_sla = any("sla_deadline_utc" in stats for stats in ack_actor_stats.values())
        # Check if any actor has per-actor threshold overrides
        has_thresholds = any("sla_threshold_hours" in stats for stats in ack_actor_stats.values())

        lines.extend([
            "## Ack Actor Coverage",
            "",
        ])

        if has_sla:
            if has_thresholds:
                lines.extend([
                    "| Actor | Hrs Since Inbound | Threshold (hrs) | Deadline (UTC) | Breached | Severity | Notes |",
                    "|-------|-------------------|-----------------|----------------|----------|----------|-------|",
                ])
            else:
                lines.extend([
                    "| Actor | Hrs Since Inbound | Deadline (UTC) | Breached | Severity | Notes |",
                    "|-------|-------------------|----------------|----------|----------|-------|",
                ])
            for actor_id in sorted(ack_actor_stats.keys()):
                stats = ack_actor_stats[actor_id]
                actor_label = actor_id.replace("_", " ").title()
                hrs = stats.get("hours_since_last_inbound")
                hrs_str = f"{hrs:.2f}" if hrs is not None else "N/A"
                deadline = stats.get("sla_deadline_utc") or "N/A"
                breached = "Yes" if stats.get("sla_breached", False) else "No"
                severity = stats.get("sla_severity", "unknown")
                notes = sanitize_for_markdown(stats.get("sla_notes", "-"))
                if has_thresholds:
                    threshold = stats.get("sla_threshold_hours")
                    threshold_str = f"{threshold:.2f}" if threshold is not None else "N/A"
                    lines.append(f"| {actor_label} | {hrs_str} | {threshold_str} | {deadline} | {breached} | {severity} | {notes} |")
                else:
                    lines.append(f"| {actor_label} | {hrs_str} | {deadline} | {breached} | {severity} | {notes} |")
        else:
            lines.extend([
                "| Actor | Hours Since Inbound | Inbound Count | Ack |",
                "|-------|---------------------|---------------|-----|",
            ])
            for actor_id in sorted(ack_actor_stats.keys()):
                stats = ack_actor_stats[actor_id]
                actor_label = actor_id.replace("_", " ").title()
                hrs = stats.get("hours_since_last_inbound")
                hrs_str = f"{hrs:.2f}" if hrs is not None else "N/A"
                inbound_count = stats.get("inbound_count", 0)
                ack_detected = "Yes" if stats.get("ack_detected", False) else "No"
                lines.append(f"| {actor_label} | {hrs_str} | {inbound_count} | {ack_detected} |")
        lines.append("")

    # Condensed timeline table
    if timeline:
        lines.extend([
            "## Timeline",
            "",
            "| Timestamp | Actor | Direction | Ack |",
            "|-----------|-------|-----------|-----|",
        ])
        for t in timeline:
            ts_short = t.get("timestamp_utc", "")[:19]  # Trim to readable length
            actor = t.get("actor", "unknown").replace("_", " ").title()
            direction = t.get("direction", "unknown").capitalize()
            ack_label = "Yes" if t.get("ack") else "No"
            lines.append(f"| {ts_short} | {actor} | {direction} | {ack_label} |")
        lines.append("")

    # Ack Actor Breach Timeline section (only when history logging enabled)
    if breach_timeline_lines:
        lines.extend(breach_timeline_lines)

    # Write the snippet (overwrite mode for idempotency)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_escalation_note(
    results: dict,
    output_path: Path,
    recipient: str = "Maintainer <2>",
    breach_timeline_lines: list = None
) -> None:
    """
    Write a Markdown escalation note when SLA is breached.

    The escalation note includes:
    - Summary Metrics (ack status, hours waiting, message counts)
    - SLA Watch (threshold, breach status, notes)
    - Action Items (what needs to happen)
    - Proposed Message (blockquote with prefilled text for the follow-up)
    - Timeline (condensed table of messages)
    - Ack Actor Breach Timeline (if breach_timeline_lines provided)

    Args:
        results: Scan results dictionary
        output_path: Path to write the escalation note
        recipient: Label for the escalation recipient
        breach_timeline_lines: Optional list of Markdown lines from
            _build_actor_breach_timeline_section(), added when --history-jsonl
            is provided. Skipped entirely when None (history logging disabled).

    If ack is already detected or no SLA info exists, the note indicates
    "No escalation required" instead of a breach warning.

    The output file is overwritten (not appended) to provide a single,
    idempotent escalation draft.
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wc = results.get("waiting_clock", {})
    sla = results.get("sla_watch", {})
    timeline = results.get("timeline", [])
    params = results.get("parameters", {})
    request_pattern = params.get("request_pattern", "unknown")

    # Compute hours since inbound for the proposed message
    hrs_in = wc.get("hours_since_last_inbound")
    hrs_in_str = f"{hrs_in:.2f}" if hrs_in is not None else "N/A"

    lines = [
        "# Escalation Note",
        "",
        f"**Generated:** {results.get('generated_utc', 'N/A')}",
        f"**Recipient:** {sanitize_for_markdown(recipient)}",
        f"**Request Pattern:** `{sanitize_for_markdown(request_pattern)}`",
        "",
    ]

    # Check if escalation is needed
    ack_detected = results.get("ack_detected", False)
    sla_breached = sla.get("breached", False) if sla else False

    if ack_detected:
        lines.extend([
            "## Status: No Escalation Required",
            "",
            "Acknowledgement has been received. No further action needed.",
            "",
        ])
    elif not sla:
        lines.extend([
            "## Status: No Escalation Required",
            "",
            "No SLA information available (--sla-hours not specified).",
            "",
        ])
    elif not sla_breached:
        lines.extend([
            "## Status: No Escalation Required",
            "",
            f"SLA has not been breached. Current wait: {hrs_in_str} hours (threshold: {sla.get('threshold_hours', 'N/A')} hours).",
            "",
        ])
    else:
        # SLA breach - full escalation note
        lines.extend([
            "## Status: SLA Breach - Escalation Required",
            "",
        ])

    # Summary Metrics section
    hrs_out = wc.get("hours_since_last_outbound")
    hrs_out_str = f"{hrs_out:.2f}" if hrs_out is not None else "N/A"

    lines.extend([
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Ack Detected | {'Yes' if ack_detected else 'No'} |",
        f"| Hours Since Last Inbound | {hrs_in_str} |",
        f"| Hours Since Last Outbound | {hrs_out_str} |",
        f"| Total Inbound Messages | {wc.get('total_inbound_count', 0)} |",
        f"| Total Outbound Messages | {wc.get('total_outbound_count', 0)} |",
        "",
    ])

    # SLA Watch section
    if sla:
        threshold = sla.get("threshold_hours")
        threshold_str = f"{threshold:.2f}" if threshold is not None else "N/A"
        breach_str = "Yes" if sla_breached else "No"
        notes = sanitize_for_markdown(sla.get("notes", ""))
        deadline_utc = sla.get("deadline_utc") or "N/A"
        breach_dur = sla.get("breach_duration_hours")
        breach_dur_str = f"{breach_dur:.2f}" if breach_dur is not None else "N/A"
        severity = sla.get("severity", "unknown")

        lines.extend([
            "## SLA Watch",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Threshold | {threshold_str} hours |",
            f"| Deadline (UTC) | {deadline_utc} |",
            f"| Breached | {breach_str} |",
            f"| Breach Duration | {breach_dur_str} hours |",
            f"| Severity | {severity} |",
            "",
            f"**Notes:** {notes}",
            "",
        ])

        if sla_breached:
            lines.extend([
                "> **SLA BREACH:** Immediate attention required.",
                "",
            ])

    # Action Items section (only when breach)
    if sla_breached and not ack_detected:
        lines.extend([
            "## Action Items",
            "",
            f"1. Send follow-up message to {sanitize_for_markdown(recipient)} requesting acknowledgement",
            f"2. Reference the `{sanitize_for_markdown(request_pattern)}` request pattern",
            f"3. Cite SLA breach: {hrs_in_str} hours since last inbound exceeds threshold",
            "4. Request explicit confirmation or next steps",
            "",
        ])

    # Proposed Message section (only when breach)
    if sla_breached and not ack_detected:
        lines.extend([
            "## Proposed Message",
            "",
            "The following blockquote can be used as a starting point for the follow-up:",
            "",
            "> **Follow-up: Acknowledgement Request**",
            ">",
            f"> Hello {sanitize_for_markdown(recipient)},",
            ">",
            f"> This is a follow-up regarding the `{sanitize_for_markdown(request_pattern)}` request.",
            f"> It has been approximately **{hrs_in_str} hours** since the last inbound message,",
            "> which exceeds our SLA threshold.",
            ">",
            "> Could you please confirm receipt of the delivered artifacts and provide any feedback?",
            ">",
            "> If there are any issues or blockers, please let us know so we can assist.",
            ">",
            "> Thank you,",
            "> Maintainer <1>",
            "",
        ])

    # Ack Actor SLA Summary section (groups actors by severity)
    ack_actor_summary = results.get("ack_actor_summary")
    if ack_actor_summary:
        lines.extend([
            "## Ack Actor SLA Summary",
            "",
        ])
        severity_order = ["critical", "warning", "ok", "unknown"]
        severity_labels = {
            "critical": "Critical (Breach >= 1h)",
            "warning": "Warning (Breach < 1h)",
            "ok": "OK (Within SLA)",
            "unknown": "Unknown (No Inbound)"
        }
        for sev in severity_order:
            actors_in_sev = ack_actor_summary.get(sev, [])
            if actors_in_sev:
                lines.append(f"### {severity_labels[sev]}")
                lines.append("")
                for entry in actors_in_sev:
                    hrs = entry.get("hours_since_inbound")
                    hrs_str = f"{hrs:.2f}" if hrs is not None else "N/A"
                    threshold = entry.get("sla_threshold_hours")
                    threshold_str = f"{threshold:.2f}" if threshold is not None else "N/A"
                    breached_str = "Yes" if entry.get("sla_breached", False) else "No"
                    lines.append(f"- **{entry['actor_label']}**: {hrs_str} hrs (threshold: {threshold_str} hrs) — Breached: {breached_str}")
                lines.append("")

    # Ack Actor Coverage table (per-actor wait metrics with SLA fields)
    ack_actor_stats = results.get("ack_actor_stats", {})
    if ack_actor_stats:
        # Check if any actor has SLA fields
        has_sla = any("sla_deadline_utc" in stats for stats in ack_actor_stats.values())
        # Check if any actor has per-actor threshold overrides
        has_thresholds = any("sla_threshold_hours" in stats for stats in ack_actor_stats.values())

        lines.extend([
            "## Ack Actor Coverage",
            "",
        ])

        if has_sla:
            if has_thresholds:
                lines.extend([
                    "| Actor | Hrs Since Inbound | Threshold (hrs) | Deadline (UTC) | Breached | Severity | Notes |",
                    "|-------|-------------------|-----------------|----------------|----------|----------|-------|",
                ])
            else:
                lines.extend([
                    "| Actor | Hrs Since Inbound | Deadline (UTC) | Breached | Severity | Notes |",
                    "|-------|-------------------|----------------|----------|----------|-------|",
                ])
            for actor_id in sorted(ack_actor_stats.keys()):
                stats = ack_actor_stats[actor_id]
                actor_label = actor_id.replace("_", " ").title()
                hrs = stats.get("hours_since_last_inbound")
                hrs_str = f"{hrs:.2f}" if hrs is not None else "N/A"
                deadline = stats.get("sla_deadline_utc") or "N/A"
                breached = "Yes" if stats.get("sla_breached", False) else "No"
                severity = stats.get("sla_severity", "unknown")
                notes = sanitize_for_markdown(stats.get("sla_notes", "-"))
                if has_thresholds:
                    threshold = stats.get("sla_threshold_hours")
                    threshold_str = f"{threshold:.2f}" if threshold is not None else "N/A"
                    lines.append(f"| {actor_label} | {hrs_str} | {threshold_str} | {deadline} | {breached} | {severity} | {notes} |")
                else:
                    lines.append(f"| {actor_label} | {hrs_str} | {deadline} | {breached} | {severity} | {notes} |")
        else:
            lines.extend([
                "| Actor | Hours Since Inbound | Inbound Count | Ack |",
                "|-------|---------------------|---------------|-----|",
            ])
            for actor_id in sorted(ack_actor_stats.keys()):
                stats = ack_actor_stats[actor_id]
                actor_label = actor_id.replace("_", " ").title()
                hrs = stats.get("hours_since_last_inbound")
                hrs_str = f"{hrs:.2f}" if hrs is not None else "N/A"
                inbound_count = stats.get("inbound_count", 0)
                ack_detected = "Yes" if stats.get("ack_detected", False) else "No"
                lines.append(f"| {actor_label} | {hrs_str} | {inbound_count} | {ack_detected} |")
        lines.append("")

    # Timeline section
    if timeline:
        lines.extend([
            "## Timeline",
            "",
            "| Timestamp | Actor | Direction | Ack |",
            "|-----------|-------|-----------|-----|",
        ])
        for t in timeline:
            ts_short = t.get("timestamp_utc", "")[:19]  # Trim to readable length
            actor = t.get("actor", "unknown").replace("_", " ").title()
            direction = t.get("direction", "unknown").capitalize()
            ack_label = "Yes" if t.get("ack") else "No"
            lines.append(f"| {ts_short} | {actor} | {direction} | {ack_label} |")
        lines.append("")

    # Ack Actor Breach Timeline section (only when history logging enabled)
    if breach_timeline_lines:
        lines.extend(breach_timeline_lines)

    # Write the escalation note (overwrite mode for idempotency)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_escalation_brief(
    results: dict,
    output_path: Path,
    recipient: str = "Maintainer <3>",
    target_actor: str = "Maintainer <2>",
    breach_timeline_lines: list = None,
    breach_timeline_data: dict = None
) -> None:
    """
    Write a Markdown escalation brief for a third-party recipient about a blocking actor.

    The brief is designed for escalation to a third party (e.g., Maintainer <3>) about
    a blocking actor (e.g., Maintainer <2>) who has not acknowledged a request.

    The brief includes:
    - Header and metadata
    - Blocking Actor Snapshot (hours since inbound, SLA threshold, deadline, hours past SLA, severity, ack files)
    - Breach Streak Summary (current streak, breach start/latest scan if timeline data present)
    - Action Items
    - Proposed Message (targeting the recipient, referencing the blocking actor)

    Args:
        results: Scan results dictionary
        output_path: Path to write the brief
        recipient: Label for the escalation brief recipient (e.g., "Maintainer <3>")
        target_actor: The blocking actor being escalated (e.g., "Maintainer <2>")
        breach_timeline_lines: Optional list of Markdown lines from
            _build_actor_breach_timeline_section()
        breach_timeline_data: Optional dict mapping actor_id -> breach state from
            _build_actor_breach_timeline_section(), used for breach streak info

    The output file is overwritten (not appended) to provide a single,
    idempotent escalation brief.
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params = results.get("parameters", {})
    request_pattern = params.get("request_pattern", "unknown")

    # Normalize target actor for lookup in stats
    target_actor_id = normalize_actor_alias(target_actor)
    target_actor_label = target_actor_id.replace("_", " ").title()

    lines = [
        "# Escalation Brief",
        "",
        f"**Generated:** {results.get('generated_utc', 'N/A')}",
        f"**Recipient:** {sanitize_for_markdown(recipient)}",
        f"**Blocking Actor:** {sanitize_for_markdown(target_actor)}",
        f"**Request Pattern:** `{sanitize_for_markdown(request_pattern)}`",
        "",
    ]

    # Blocking Actor Snapshot section
    ack_actor_stats = results.get("ack_actor_stats", {})
    target_stats = ack_actor_stats.get(target_actor_id, {})

    if target_stats:
        hrs_since = target_stats.get("hours_since_last_inbound")
        hrs_since_str = f"{hrs_since:.2f}" if hrs_since is not None else "N/A"
        threshold = target_stats.get("sla_threshold_hours")
        threshold_str = f"{threshold:.2f}" if threshold is not None else "N/A"
        deadline = target_stats.get("sla_deadline_utc") or "N/A"
        breach_dur = target_stats.get("sla_breach_duration_hours")
        breach_dur_str = f"{breach_dur:.2f}" if breach_dur is not None else "0.00"
        severity = target_stats.get("sla_severity", "unknown")
        ack_files = target_stats.get("ack_files", [])
        ack_files_str = ", ".join(f"`{f}`" for f in ack_files) if ack_files else "None"

        lines.extend([
            "## Blocking Actor Snapshot",
            "",
            f"**Actor:** {target_actor_label}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Hours Since Inbound | {hrs_since_str} |",
            f"| SLA Threshold | {threshold_str} hours |",
            f"| Deadline (UTC) | {deadline} |",
            f"| Hours Past SLA | {breach_dur_str} |",
            f"| Severity | {severity.upper() if severity else 'UNKNOWN'} |",
            f"| Ack Files | {ack_files_str} |",
            "",
        ])

        if severity in ("warning", "critical"):
            lines.extend([
                f"> **SLA BREACH ({severity.upper()}):** {target_actor_label} has exceeded the SLA threshold.",
                "",
            ])
    else:
        lines.extend([
            "## Blocking Actor Snapshot",
            "",
            f"**Actor:** {target_actor_label}",
            "",
            "*Data unavailable for this actor. The actor may not be in the monitored ack_actors list.*",
            "",
        ])

    # Breach Streak Summary section (from timeline data if available)
    lines.extend([
        "## Breach Streak Summary",
        "",
    ])

    if breach_timeline_data and target_actor_id in breach_timeline_data:
        breach_state = breach_timeline_data[target_actor_id]
        breach_start = breach_state.get("breach_start")
        latest_scan = breach_state.get("latest_scan")
        current_streak = breach_state.get("current_streak", 0)
        hours_past = breach_state.get("hours_past_sla", 0.0)

        breach_start_str = breach_start[:19] if breach_start else "N/A"
        latest_scan_str = latest_scan[:19] if latest_scan else "N/A"
        hours_past_str = f"{hours_past:.2f}" if hours_past > 0 else "0.00"

        lines.extend([
            "| Metric | Value |",
            "|--------|-------|",
            f"| Current Streak | {current_streak} consecutive scan(s) |",
            f"| Breach Start | {breach_start_str} |",
            f"| Latest Scan | {latest_scan_str} |",
            f"| Peak Hours Past SLA | {hours_past_str} |",
            "",
        ])
    else:
        lines.extend([
            "*Breach streak data unavailable. Use `--history-jsonl` to enable breach tracking.*",
            "",
        ])

    # Action Items section
    lines.extend([
        "## Action Items",
        "",
        f"1. Review the SLA breach evidence for {target_actor_label}",
        f"2. Contact {sanitize_for_markdown(recipient)} to escalate the blocking issue",
        f"3. Reference request pattern: `{sanitize_for_markdown(request_pattern)}`",
        f"4. Request acknowledgement or status update from {target_actor_label}",
        "",
    ])

    # Proposed Message section (targeting the brief recipient, referencing blocking actor)
    hrs_since = target_stats.get("hours_since_last_inbound") if target_stats else None
    hrs_since_str = f"{hrs_since:.2f}" if hrs_since is not None else "N/A"

    lines.extend([
        "## Proposed Message",
        "",
        f"The following can be used when contacting {sanitize_for_markdown(recipient)}:",
        "",
        "> **Escalation: SLA Breach on Request**",
        ">",
        f"> Hello {sanitize_for_markdown(recipient)},",
        ">",
        f"> I am escalating an SLA breach regarding the `{sanitize_for_markdown(request_pattern)}` request.",
        f"> {target_actor_label} has not acknowledged receipt, and it has been **{hrs_since_str} hours**",
        "> since the last inbound message from them, exceeding our SLA threshold.",
        ">",
        f"> Could you please assist in obtaining a response or status update from {target_actor_label}?",
        ">",
        "> If there are any blockers or issues on their end, please let us know so we can adjust our plans.",
        ">",
        "> Thank you,",
        "> Maintainer <1>",
        "",
    ])

    # Ack Actor Breach Timeline section (if provided)
    if breach_timeline_lines:
        lines.extend(breach_timeline_lines)

    # Write the brief (overwrite mode for idempotency)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _build_actor_breach_timeline_section(entries: list) -> tuple[list, dict]:
    """
    Build the Ack Actor Breach Timeline section for the history dashboard.

    Scans JSONL entries chronologically to track per-actor breach state:
    - Breach start timestamp (when actor first entered warning/critical)
    - Latest scan timestamp
    - Current streak count (consecutive scans in warning/critical)
    - Hours past SLA (hours_since_inbound - sla_threshold)

    Severity priority for sorting: critical > warning > ok > unknown

    Returns:
        (lines, active_breaches) where:
        - lines: list of Markdown lines for the section
        - active_breaches: dict mapping actor_id -> breach state dict with keys:
            actor_label, breach_start, latest_scan, current_streak, hours_past_sla, severity

    Gracefully handles entries without ack_actor_summary data.
    """
    lines = []

    # Track per-actor breach state chronologically
    # actor_id -> {
    #   breach_start: str (first warning/critical timestamp),
    #   latest_scan: str,
    #   current_streak: int,
    #   hours_past_sla: float,
    #   severity: str (latest),
    #   actor_label: str
    # }
    actor_breach_state: dict = {}

    entries_with_summary = 0
    for entry in entries:
        summary = entry.get("ack_actor_summary")
        if not summary:
            continue
        entries_with_summary += 1

        entry_ts = entry.get("generated_utc", "")

        severity_order = ["critical", "warning", "ok", "unknown"]
        for sev in severity_order:
            actors_in_sev = summary.get(sev, [])
            for actor_data in actors_in_sev:
                actor_id = actor_data.get("actor_id", "unknown")
                actor_label = actor_data.get("actor_label", actor_id.replace("_", " ").title())
                hrs_since = actor_data.get("hours_since_inbound")
                sla_threshold = actor_data.get("sla_threshold_hours")

                # Initialize state if new actor
                if actor_id not in actor_breach_state:
                    actor_breach_state[actor_id] = {
                        "actor_label": actor_label,
                        "breach_start": None,
                        "latest_scan": "",
                        "current_streak": 0,
                        "hours_past_sla": 0.0,
                        "severity": sev,
                    }

                state = actor_breach_state[actor_id]

                # Update latest scan timestamp
                if entry_ts > state["latest_scan"]:
                    state["latest_scan"] = entry_ts

                # Track breach state for warning/critical actors
                if sev in ("warning", "critical"):
                    if state["breach_start"] is None:
                        # First time entering breach state
                        state["breach_start"] = entry_ts
                        state["current_streak"] = 1
                    else:
                        # Continuing breach streak
                        state["current_streak"] += 1

                    # Compute hours past SLA
                    if hrs_since is not None and sla_threshold is not None:
                        past_sla = max(hrs_since - sla_threshold, 0.0)
                        if past_sla > state["hours_past_sla"]:
                            state["hours_past_sla"] = past_sla

                    state["severity"] = sev
                else:
                    # Actor returned to ok/unknown - reset streak
                    state["breach_start"] = None
                    state["current_streak"] = 0
                    state["hours_past_sla"] = 0.0
                    state["severity"] = sev

    # Filter to only actors with active breaches (warning/critical with streak > 0)
    active_breaches = {
        actor_id: state
        for actor_id, state in actor_breach_state.items()
        if state["severity"] in ("warning", "critical") and state["current_streak"] > 0
    }

    if not active_breaches:
        lines.extend([
            "## Ack Actor Breach Timeline",
            "",
            "**Status:** No active breaches detected.",
            "",
            "All tracked actors are currently within SLA thresholds or have returned to OK status.",
            "",
        ])
        return lines, {}

    # Sort by severity priority (critical > warning), then by hours_past_sla descending
    def breach_sort_key(item):
        actor_id, state = item
        sev_priority = {"critical": 0, "warning": 1}.get(state["severity"], 2)
        return (sev_priority, -state["hours_past_sla"], actor_id)

    sorted_breaches = sorted(active_breaches.items(), key=breach_sort_key)

    lines.extend([
        "## Ack Actor Breach Timeline",
        "",
        f"**Active breaches:** {len(active_breaches)} actor(s)",
        "",
        "| Actor | Breach Start | Latest Scan | Current Streak | Hours Past SLA | Severity |",
        "|-------|--------------|-------------|----------------|----------------|----------|",
    ])

    for actor_id, state in sorted_breaches:
        label = _sanitize_md(state["actor_label"])
        breach_start = state["breach_start"][:19] if state["breach_start"] else "N/A"
        latest_scan = state["latest_scan"][:19] if state["latest_scan"] else "N/A"
        streak = state["current_streak"]
        hours_past = f"{state['hours_past_sla']:.2f}h" if state["hours_past_sla"] > 0 else "N/A"
        severity = state["severity"].upper()

        lines.append(
            f"| {label} | {breach_start} | {latest_scan} | {streak} | {hours_past} | {severity} |"
        )

    lines.append("")
    return lines, active_breaches


def _build_actor_severity_trends_section(entries: list) -> list:
    """
    Build the Ack Actor Severity Trends section for the history dashboard.

    Aggregates ack_actor_summary data across all JSONL entries to compute:
    - Per-actor severity counts (critical/warning/ok/unknown)
    - Longest wait per actor across all scans
    - Latest timestamp per actor

    Returns a list of Markdown lines for the section.
    Gracefully handles entries without ack_actor_summary data.
    """
    lines = []

    # Collect per-actor data across all entries
    # actor_id -> {severity_counts: {critical:N, warning:N, ok:N, unknown:N},
    #              longest_wait: float|None, latest_timestamp: str, actor_label: str}
    actor_trends: dict = {}

    entries_with_summary = 0
    for entry in entries:
        summary = entry.get("ack_actor_summary")
        if not summary:
            continue
        entries_with_summary += 1

        entry_ts = entry.get("generated_utc", "")

        severity_order = ["critical", "warning", "ok", "unknown"]
        for sev in severity_order:
            actors_in_sev = summary.get(sev, [])
            for actor_data in actors_in_sev:
                actor_id = actor_data.get("actor_id", "unknown")
                actor_label = actor_data.get("actor_label", actor_id.replace("_", " ").title())
                hrs = actor_data.get("hours_since_inbound")

                if actor_id not in actor_trends:
                    actor_trends[actor_id] = {
                        "actor_label": actor_label,
                        "severity_counts": {"critical": 0, "warning": 0, "ok": 0, "unknown": 0},
                        "longest_wait": None,
                        "latest_timestamp": "",
                    }

                # Increment severity count
                actor_trends[actor_id]["severity_counts"][sev] += 1

                # Track longest wait
                if hrs is not None:
                    cur_longest = actor_trends[actor_id]["longest_wait"]
                    if cur_longest is None or hrs > cur_longest:
                        actor_trends[actor_id]["longest_wait"] = hrs

                # Track latest timestamp
                if entry_ts > actor_trends[actor_id]["latest_timestamp"]:
                    actor_trends[actor_id]["latest_timestamp"] = entry_ts

    if not actor_trends:
        lines.extend([
            "## Ack Actor Severity Trends",
            "",
            "**Status:** No per-actor data available in the history log.",
            "",
            "Per-actor severity data is captured when `--ack-actor` and `--ack-actor-sla`",
            "flags are used with the inbox scan CLI.",
            "",
        ])
        return lines

    # Sort actors by severity priority: actors with most critical counts first,
    # then warning, then ok, then unknown
    def actor_sort_key(item):
        actor_id, data = item
        counts = data["severity_counts"]
        # Negative so higher counts sort first, severity priority: critical > warning > ok > unknown
        return (-counts["critical"], -counts["warning"], -counts["ok"], -counts["unknown"], actor_id)

    sorted_actors = sorted(actor_trends.items(), key=actor_sort_key)

    lines.extend([
        "## Ack Actor Severity Trends",
        "",
        f"**Entries with per-actor data:** {entries_with_summary} of {len(entries)}",
        "",
        "| Actor | Critical | Warning | OK | Unknown | Longest Wait | Latest Scan |",
        "|-------|----------|---------|----|---------|--------------| ------------|",
    ])

    for actor_id, data in sorted_actors:
        label = _sanitize_md(data["actor_label"])
        counts = data["severity_counts"]
        longest = data["longest_wait"]
        longest_str = f"{longest:.2f}h" if longest is not None else "N/A"
        latest = data["latest_timestamp"][:19] if data["latest_timestamp"] else "N/A"

        lines.append(
            f"| {label} | {counts['critical']} | {counts['warning']} | {counts['ok']} | {counts['unknown']} | {longest_str} | {latest} |"
        )

    lines.append("")
    return lines


def _sanitize_md(text: str) -> str:
    """Sanitize text for safe inclusion in Markdown table cells."""
    if not text:
        return ""
    # Replace pipe characters and newlines that would break table structure
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", "")


def write_history_dashboard(
    jsonl_path: Path,
    output_path: Path,
    max_entries: int = 10
) -> None:
    """
    Read the JSONL history log and emit an aggregated Markdown dashboard.

    The dashboard includes:
    - Summary Metrics (total scans, ack count, breach count)
    - SLA Breach Stats (longest wait, most recent ack timestamp)
    - Recent Scans table (last N entries from the JSONL)
    - Ack Actor Severity Trends (per-actor severity counts/longest wait/latest timestamps)
    - Ack Actor Breach Timeline (per-actor breach start/streak/hours-past-SLA)

    If the JSONL file does not exist or is empty, emits a "No history" message.
    The output file is overwritten (not appended) to provide an idempotent snapshot.
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Inbox History Dashboard",
        "",
    ]

    # Read and parse JSONL entries
    entries = []
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    # Skip invalid/partial lines
                    continue

    if not entries:
        lines.extend([
            "**Status:** No history data available.",
            "",
            "The history JSONL file is empty or does not exist. Run the inbox scan CLI",
            "with `--history-jsonl` to begin collecting history entries.",
            "",
        ])
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return

    # Compute summary metrics
    total_scans = len(entries)
    ack_count = sum(1 for e in entries if e.get("ack_detected", False))
    breach_count = sum(1 for e in entries if e.get("sla_breached", False))

    # Compute SLA breach stats
    hours_since_inbound_values = [
        e.get("hours_since_inbound")
        for e in entries
        if e.get("hours_since_inbound") is not None
    ]
    longest_wait = max(hours_since_inbound_values) if hours_since_inbound_values else None

    # Find most recent ack timestamp (if any)
    ack_entries = [e for e in entries if e.get("ack_detected", False)]
    last_ack_timestamp = None
    if ack_entries:
        # Sort by generated_utc and take the latest
        ack_entries_sorted = sorted(
            ack_entries,
            key=lambda x: x.get("generated_utc", ""),
            reverse=True
        )
        last_ack_timestamp = ack_entries_sorted[0].get("generated_utc")

    # Most recent scan timestamp
    last_scan_timestamp = entries[-1].get("generated_utc") if entries else None

    # Build dashboard
    lines.extend([
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**History Source:** `{jsonl_path}`",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Scans | {total_scans} |",
        f"| Ack Count | {ack_count} |",
        f"| Breach Count | {breach_count} |",
        "",
    ])

    # SLA Breach Stats
    longest_wait_str = f"{longest_wait:.2f} hours" if longest_wait is not None else "N/A"
    last_ack_str = last_ack_timestamp[:19] if last_ack_timestamp else "None"
    last_scan_str = last_scan_timestamp[:19] if last_scan_timestamp else "N/A"

    lines.extend([
        "## SLA Breach Stats",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Longest Wait | {longest_wait_str} |",
        f"| Last Ack Timestamp | {last_ack_str} |",
        f"| Last Scan Timestamp | {last_scan_str} |",
        "",
    ])

    # Recent Scans table (last N entries)
    recent_entries = entries[-max_entries:]

    lines.extend([
        f"## Recent Scans (last {len(recent_entries)} of {total_scans})",
        "",
        "| Timestamp | Ack | Hrs Inbound | Hrs Outbound | SLA Breach | Severity | Matches |",
        "|-----------|-----|-------------|--------------|------------|----------|---------|",
    ])

    for e in recent_entries:
        ts = e.get("generated_utc", "")[:19]
        ack = "Yes" if e.get("ack_detected", False) else "No"
        hrs_in = e.get("hours_since_inbound")
        hrs_in_str = f"{hrs_in:.2f}" if hrs_in is not None else "N/A"
        hrs_out = e.get("hours_since_outbound")
        hrs_out_str = f"{hrs_out:.2f}" if hrs_out is not None else "N/A"
        breach = e.get("sla_breached")
        breach_str = "Yes" if breach else ("No" if breach is False else "N/A")
        severity = e.get("sla_severity") or "N/A"
        matches = e.get("total_matches", 0)
        lines.append(f"| {ts} | {ack} | {hrs_in_str} | {hrs_out_str} | {breach_str} | {severity} | {matches} |")

    lines.append("")

    # Ack Actor Severity Trends - aggregate per-actor data across all entries
    lines.extend(_build_actor_severity_trends_section(entries))

    # Ack Actor Breach Timeline - per-actor breach state tracking
    breach_timeline_lines, _ = _build_actor_breach_timeline_section(entries)
    lines.extend(breach_timeline_lines)

    # Write the dashboard (overwrite mode for idempotency)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    """Main entry point."""
    args = parse_args()

    # Validate --history-dashboard requires --history-jsonl
    if args.history_dashboard and not args.history_jsonl:
        print("ERROR: --history-dashboard requires --history-jsonl to be specified")
        return 1

    inbox_path = Path(args.inbox)
    output_path = Path(args.output)

    # Parse since date if provided
    since_date = None
    if args.since:
        since_date = datetime.fromisoformat(args.since.replace("Z", "+00:00"))

    # Use default keywords if none provided
    keywords = args.keywords if args.keywords else [
        "acknowledged", "ack", "confirm", "received", "thanks"
    ]

    # Use default ack_actors if none provided (backwards compatible: only Maintainer <2>)
    ack_actors = args.ack_actors if args.ack_actors else ["Maintainer <2>"]

    # Parse per-actor SLA overrides
    ack_actor_sla_overrides = parse_ack_actor_sla_overrides(args.ack_actor_sla) if args.ack_actor_sla else {}

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning inbox: {inbox_path}")
    print(f"Request pattern: {args.request_pattern}")
    print(f"Keywords: {keywords}")
    print(f"Ack actors: {ack_actors}")
    if args.sla_hours is not None:
        print(f"SLA threshold (global): {args.sla_hours} hours")
        if ack_actor_sla_overrides:
            print(f"Per-actor SLA overrides: {ack_actor_sla_overrides}")
        print(f"Fail when breached: {args.fail_when_breached}")
    print(f"Output: {output_path}")
    print("")

    # Scan inbox
    results = scan_inbox(
        inbox_path, args.request_pattern, keywords, since_date,
        sla_hours=args.sla_hours,
        ack_actors=ack_actors,
        ack_actor_sla_overrides=ack_actor_sla_overrides
    )

    # Write outputs
    json_path = output_path / "inbox_scan_summary.json"
    md_path = output_path / "inbox_scan_summary.md"

    write_json_summary(results, json_path)
    write_markdown_summary(results, md_path)

    # Append to history files if specified
    breach_timeline_lines = None  # Only populated when history JSONL is enabled
    breach_timeline_data = None   # Dict of actor_id -> breach state (for escalation brief)
    if args.history_jsonl:
        history_jsonl_path = Path(args.history_jsonl)
        append_history_jsonl(results, history_jsonl_path)
        print(f"History JSONL appended: {history_jsonl_path}")

        # Read back the JSONL to build breach timeline for snippet/escalation note
        # (read AFTER append so newest entry is included)
        history_entries = []
        if history_jsonl_path.exists():
            with open(history_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        history_entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        if history_entries:
            breach_timeline_lines, breach_timeline_data = _build_actor_breach_timeline_section(history_entries)

    if args.history_markdown:
        history_md_path = Path(args.history_markdown)
        append_history_markdown(results, history_md_path)
        print(f"History Markdown appended: {history_md_path}")

    # Write history dashboard if specified
    if args.history_dashboard:
        history_dashboard_path = Path(args.history_dashboard)
        history_jsonl_path = Path(args.history_jsonl)
        write_history_dashboard(history_jsonl_path, history_dashboard_path)
        print(f"History dashboard written: {history_dashboard_path}")

    # Write status snippet if specified
    if args.status_snippet:
        status_snippet_path = Path(args.status_snippet)
        write_status_snippet(results, status_snippet_path, breach_timeline_lines)
        print(f"Status snippet written: {status_snippet_path}")

    # Write escalation note if specified
    if args.escalation_note:
        escalation_note_path = Path(args.escalation_note)
        write_escalation_note(
            results, escalation_note_path, args.escalation_recipient,
            breach_timeline_lines
        )
        print(f"Escalation note written: {escalation_note_path}")

    # Write escalation brief if specified
    if args.escalation_brief:
        escalation_brief_path = Path(args.escalation_brief)
        write_escalation_brief(
            results, escalation_brief_path,
            recipient=args.escalation_brief_recipient,
            target_actor=args.escalation_brief_target,
            breach_timeline_lines=breach_timeline_lines,
            breach_timeline_data=breach_timeline_data
        )
        print(f"Escalation brief written: {escalation_brief_path}")

    # Print summary
    print(f"Files scanned: {results['scanned']}")
    print(f"Matches found: {len(results['matches'])}")
    print(f"Acknowledgement detected: {results['ack_detected']}")
    print("")

    # Print waiting clock
    wc = results.get("waiting_clock", {})
    print("Waiting Clock:")
    print(f"  Last inbound (from M2): {wc.get('last_inbound_utc') or 'N/A'}")
    hrs_in = wc.get("hours_since_last_inbound")
    print(f"  Hours since last inbound: {f'{hrs_in:.2f}' if hrs_in is not None else 'N/A'}")
    print(f"  Last outbound (from M1): {wc.get('last_outbound_utc') or 'N/A'}")
    hrs_out = wc.get("hours_since_last_outbound")
    print(f"  Hours since last outbound: {f'{hrs_out:.2f}' if hrs_out is not None else 'N/A'}")
    print(f"  Total inbound: {wc.get('total_inbound_count', 0)}")
    print(f"  Total outbound: {wc.get('total_outbound_count', 0)}")
    print("")

    if results["matches"]:
        print("Matching files:")
        for m in results["matches"]:
            reason = ", ".join(m["match_reason"])
            direction = m.get("direction", "unknown")
            ack_str = " [ACK]" if m["ack_detected"] else ""
            print(f"  - {m['file']} ({reason}) [{direction}]{ack_str}")

    print("")
    if results.get("timeline"):
        print("Timeline (chronological):")
        for t in results["timeline"]:
            ts_short = t["timestamp_utc"][:19]  # Trim to readable length
            actor = t["actor"].replace("_", " ").title()
            direction = t["direction"].capitalize()
            ack_flag = " [ACK]" if t["ack"] else ""
            print(f"  {ts_short} | {actor} | {direction}{ack_flag}")
        print("")

    print(f"JSON summary: {json_path}")
    print(f"Markdown summary: {md_path}")

    # Print SLA watch if present
    sla_watch = results.get("sla_watch")
    if sla_watch:
        print("")
        print("SLA Watch:")
        print(f"  Threshold: {sla_watch['threshold_hours']:.2f} hours")
        hrs = sla_watch.get("hours_since_last_inbound")
        print(f"  Hours since last inbound: {f'{hrs:.2f}' if hrs is not None else 'N/A'}")
        deadline_utc = sla_watch.get("deadline_utc") or "N/A"
        print(f"  Deadline (UTC): {deadline_utc}")
        print(f"  Breached: {'Yes' if sla_watch['breached'] else 'No'}")
        breach_dur = sla_watch.get("breach_duration_hours")
        breach_dur_str = f"{breach_dur:.2f}" if breach_dur is not None else "N/A"
        print(f"  Breach duration: {breach_dur_str} hours")
        severity = sla_watch.get("severity", "unknown")
        print(f"  Severity: {severity}")
        print(f"  Notes: {sla_watch['notes']}")

    # Print per-actor wait metrics if present
    ack_actor_stats = results.get("ack_actor_stats", {})
    if ack_actor_stats:
        print("")
        print("Ack Actor Coverage:")
        for actor_id in sorted(ack_actor_stats.keys()):
            stats = ack_actor_stats[actor_id]
            actor_label = actor_id.replace("_", " ").title()
            hrs = stats.get("hours_since_last_inbound")
            hrs_str = f"{hrs:.2f}" if hrs is not None else "N/A"
            inbound_count = stats.get("inbound_count", 0)
            ack_detected = "Yes" if stats.get("ack_detected", False) else "No"
            ack_files = ", ".join(stats.get("ack_files", [])) if stats.get("ack_files") else "-"
            print(f"  {actor_label}:")
            print(f"    Hours since last inbound: {hrs_str}")
            print(f"    Inbound count: {inbound_count}")
            print(f"    Ack detected: {ack_detected}")
            print(f"    Ack files: {ack_files}")
            # Print per-actor SLA fields if present
            if "sla_deadline_utc" in stats:
                # Print threshold if present (for per-actor overrides)
                if "sla_threshold_hours" in stats:
                    threshold = stats.get("sla_threshold_hours")
                    threshold_str = f"{threshold:.2f}" if threshold is not None else "N/A"
                    print(f"    SLA Threshold: {threshold_str} hours")
                deadline = stats.get("sla_deadline_utc") or "N/A"
                breached = "Yes" if stats.get("sla_breached", False) else "No"
                breach_dur = stats.get("sla_breach_duration_hours")
                breach_dur_str = f"{breach_dur:.2f}" if breach_dur is not None else "N/A"
                severity = stats.get("sla_severity", "unknown")
                notes = stats.get("sla_notes", "-")
                print(f"    SLA Deadline (UTC): {deadline}")
                print(f"    SLA Breached: {breached}")
                print(f"    SLA Breach Duration: {breach_dur_str} hours")
                print(f"    SLA Severity: {severity}")
                print(f"    SLA Notes: {notes}")

    # Print Ack Actor SLA Summary (grouped by severity)
    ack_actor_summary = results.get("ack_actor_summary")
    if ack_actor_summary:
        print("")
        print("Ack Actor SLA Summary:")
        severity_order = ["critical", "warning", "ok", "unknown"]
        severity_labels = {
            "critical": "Critical (Breach >= 1h)",
            "warning": "Warning (Breach < 1h)",
            "ok": "OK (Within SLA)",
            "unknown": "Unknown (No Inbound)"
        }
        for sev in severity_order:
            actors_in_sev = ack_actor_summary.get(sev, [])
            if actors_in_sev:
                print(f"  [{sev.upper()}] {severity_labels[sev]}:")
                for entry in actors_in_sev:
                    hrs = entry.get("hours_since_inbound")
                    hrs_str = f"{hrs:.2f}" if hrs is not None else "N/A"
                    threshold = entry.get("sla_threshold_hours")
                    threshold_str = f"{threshold:.2f}" if threshold is not None else "N/A"
                    breached = "Yes" if entry.get("sla_breached", False) else "No"
                    print(f"    - {entry['actor_label']}: {hrs_str} hrs since inbound (threshold: {threshold_str} hrs) — Breached: {breached}")

    # Return exit code based on ack_detected and SLA breach
    exit_code = 0
    if args.fail_when_breached and sla_watch and sla_watch.get("breached"):
        print("")
        print("ERROR: SLA breach detected and --fail-when-breached is set")
        exit_code = 2
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
