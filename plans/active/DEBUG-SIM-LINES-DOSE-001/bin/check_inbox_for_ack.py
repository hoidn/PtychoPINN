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
    ack_actors: list[str] | None = None
) -> dict:
    """
    Scan inbox directory for files matching the request pattern.

    Returns a summary dict with scanned/matches/ack_detected status,
    plus timeline and waiting-clock metadata.

    If sla_hours is provided, also computes sla_watch metrics.
    current_time can be injected for testing; defaults to now(UTC).
    ack_actors is a list of normalized actor IDs (e.g., ["maintainer_2", "maintainer_3"])
    that are considered valid acknowledgement sources.
    """
    # Normalize ack_actors, default to ["maintainer_2"]
    if ack_actors is None:
        ack_actors = ["maintainer_2"]
    else:
        ack_actors = [normalize_actor_alias(a) for a in ack_actors]

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
            "ack_actors": ack_actors
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

        ack_actor_stats[actor_id] = {
            "last_inbound_utc": last_inbound_from_actor,
            "hours_since_last_inbound": compute_hours_since(last_inbound_from_actor, now),
            "inbound_count": len(actor_inbound),
            "ack_files": actor_ack_files,
            "ack_detected": len(actor_ack_files) > 0,
        }

    results["ack_actor_stats"] = ack_actor_stats

    # SLA Watch computation (if sla_hours provided)
    if sla_hours is not None:
        hours_since = results["waiting_clock"]["hours_since_last_inbound"]
        breached = False
        notes = ""
        if hours_since is None:
            # No inbound messages at all
            notes = "No inbound messages from Maintainer <2> found"
            breached = False  # Cannot be breached if no inbound exists
        else:
            breached = hours_since > sla_hours and not results["ack_detected"]
            if breached:
                notes = f"SLA breach: {hours_since:.2f} hours since last inbound exceeds {sla_hours:.2f} hour threshold and no acknowledgement detected"
            elif results["ack_detected"]:
                notes = "Acknowledgement received; SLA not applicable"
            else:
                notes = f"Within SLA: {hours_since:.2f} hours since last inbound (threshold: {sla_hours:.2f})"

        results["sla_watch"] = {
            "threshold_hours": sla_hours,
            "hours_since_last_inbound": hours_since,
            "breached": breached,
            "notes": notes
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
        lines.extend([
            f"- **Hours Since Last Inbound:** {hrs_str}",
            f"- **Breached:** {'Yes' if sla_watch['breached'] else 'No'}",
            f"- **Notes:** {sla_watch['notes']}",
            "",
        ])
        if sla_watch["breached"]:
            lines.extend([
                "> **SLA BREACH:** The waiting time has exceeded the configured threshold and no acknowledgement has been received.",
                "",
            ])

    # Ack Actor Coverage table (per-actor wait metrics)
    ack_actor_stats = results.get("ack_actor_stats", {})
    if ack_actor_stats:
        lines.extend([
            "## Ack Actor Coverage",
            "",
            "Per-actor wait metrics for monitored acknowledgement actors:",
            "",
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
    hours_since_outbound, sla_breached, ack_files.
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
        "ack_files": results.get("ack_files", []),
        "total_matches": len(results.get("matches", [])),
        "total_inbound": wc.get("total_inbound_count", 0),
        "total_outbound": wc.get("total_outbound_count", 0),
    }

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def append_history_markdown(results: dict, output_path: Path) -> None:
    """
    Append a single history row to a Markdown file.

    If the file doesn't exist or is empty, write the header first.
    Table columns: Generated (UTC) | Ack Detected | Hours Since Inbound |
                   Hours Since Outbound | SLA Breach | Ack Files
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
            f.write("| Generated (UTC) | Ack | Hrs Inbound | Hrs Outbound | SLA Breach | Ack Files |\n")
            f.write("|-----------------|-----|-------------|--------------|------------|----------|\n")

        # Format values
        gen_utc = results.get("generated_utc", "")[:19]  # Trim to readable length
        ack = "Yes" if results.get("ack_detected", False) else "No"

        hrs_in = wc.get("hours_since_last_inbound")
        hrs_in_str = f"{hrs_in:.2f}" if hrs_in is not None else "N/A"

        hrs_out = wc.get("hours_since_last_outbound")
        hrs_out_str = f"{hrs_out:.2f}" if hrs_out is not None else "N/A"

        breach = sla.get("breached") if sla else None
        breach_str = "Yes" if breach else ("No" if breach is False else "N/A")

        ack_files = results.get("ack_files", [])
        ack_files_str = ", ".join(sanitize_for_markdown(f) for f in ack_files) if ack_files else "-"

        f.write(f"| {gen_utc} | {ack} | {hrs_in_str} | {hrs_out_str} | {breach_str} | {ack_files_str} |\n")


def write_status_snippet(results: dict, output_path: Path) -> None:
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

        lines.extend([
            "## SLA Watch",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Threshold | {threshold_str} hours |",
            f"| Breached | {breach_str} |",
            "",
            f"**Notes:** {sanitize_for_markdown(notes)}",
            "",
        ])

        if breached:
            lines.extend([
                "> **SLA BREACH:** Action required.",
                "",
            ])

    # Ack Actor Coverage table (per-actor wait metrics)
    ack_actor_stats = results.get("ack_actor_stats", {})
    if ack_actor_stats:
        lines.extend([
            "## Ack Actor Coverage",
            "",
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

    # Write the snippet (overwrite mode for idempotency)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_escalation_note(
    results: dict,
    output_path: Path,
    recipient: str = "Maintainer <2>"
) -> None:
    """
    Write a Markdown escalation note when SLA is breached.

    The escalation note includes:
    - Summary Metrics (ack status, hours waiting, message counts)
    - SLA Watch (threshold, breach status, notes)
    - Action Items (what needs to happen)
    - Proposed Message (blockquote with prefilled text for the follow-up)
    - Timeline (condensed table of messages)

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

        lines.extend([
            "## SLA Watch",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Threshold | {threshold_str} hours |",
            f"| Breached | {breach_str} |",
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

    # Ack Actor Coverage table (per-actor wait metrics)
    ack_actor_stats = results.get("ack_actor_stats", {})
    if ack_actor_stats:
        lines.extend([
            "## Ack Actor Coverage",
            "",
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

    # Write the escalation note (overwrite mode for idempotency)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


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
        "| Timestamp | Ack | Hrs Inbound | Hrs Outbound | SLA Breach | Matches |",
        "|-----------|-----|-------------|--------------|------------|---------|",
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
        matches = e.get("total_matches", 0)
        lines.append(f"| {ts} | {ack} | {hrs_in_str} | {hrs_out_str} | {breach_str} | {matches} |")

    lines.append("")

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

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning inbox: {inbox_path}")
    print(f"Request pattern: {args.request_pattern}")
    print(f"Keywords: {keywords}")
    print(f"Ack actors: {ack_actors}")
    if args.sla_hours is not None:
        print(f"SLA threshold: {args.sla_hours} hours")
        print(f"Fail when breached: {args.fail_when_breached}")
    print(f"Output: {output_path}")
    print("")

    # Scan inbox
    results = scan_inbox(
        inbox_path, args.request_pattern, keywords, since_date,
        sla_hours=args.sla_hours,
        ack_actors=ack_actors
    )

    # Write outputs
    json_path = output_path / "inbox_scan_summary.json"
    md_path = output_path / "inbox_scan_summary.md"

    write_json_summary(results, json_path)
    write_markdown_summary(results, md_path)

    # Append to history files if specified
    if args.history_jsonl:
        history_jsonl_path = Path(args.history_jsonl)
        append_history_jsonl(results, history_jsonl_path)
        print(f"History JSONL appended: {history_jsonl_path}")

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
        write_status_snippet(results, status_snippet_path)
        print(f"Status snippet written: {status_snippet_path}")

    # Write escalation note if specified
    if args.escalation_note:
        escalation_note_path = Path(args.escalation_note)
        write_escalation_note(results, escalation_note_path, args.escalation_recipient)
        print(f"Escalation note written: {escalation_note_path}")

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
        print(f"  Breached: {'Yes' if sla_watch['breached'] else 'No'}")
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

    # Return exit code based on ack_detected and SLA breach
    exit_code = 0
    if args.fail_when_breached and sla_watch and sla_watch.get("breached"):
        print("")
        print("ERROR: SLA breach detected and --fail-when-breached is set")
        exit_code = 2
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
