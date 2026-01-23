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


def detect_actor_and_direction(content: str) -> Tuple[str, str]:
    """
    Detect which maintainer authored the message and the communication direction.

    Returns (actor, direction) where:
    - actor: "maintainer_1", "maintainer_2", or "unknown"
    - direction: "inbound" (from M2 to M1), "outbound" (from M1 to M2), or "unknown"

    Looks for patterns like:
    - "From: Maintainer <1>" / "**From:** Maintainer <1>"
    - "From: Maintainer <2>" / "**From:** Maintainer <2>"
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

    for pattern in m1_patterns:
        if re.search(pattern, content_lower):
            return ("maintainer_1", "outbound")

    for pattern in m2_patterns:
        if re.search(pattern, content_lower):
            return ("maintainer_2", "inbound")

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
    keywords: list[str]
) -> tuple[bool, list[str], bool]:
    """
    Check if content contains acknowledgement keywords FROM Maintainer <2>.

    An acknowledgement is detected when:
    - The message is FROM Maintainer <2> (not TO Maintainer <2>)
    - AND contains any core ack keyword ('acknowledg', 'confirm', 'received')

    Returns (ack_detected, keyword_hits, is_from_m2).
    """
    content_lower = content.lower()
    keyword_hits = []

    for kw in keywords:
        if kw.lower() in content_lower:
            keyword_hits.append(kw)

    # Check if message is FROM Maintainer <2>
    is_from_m2 = is_from_maintainer_2(content)

    # Core acknowledgement keywords
    ack_core_keywords = ["acknowledg", "confirm", "received"]

    # Check if we have a core ack keyword
    has_ack_keyword = any(k in content_lower for k in ack_core_keywords)

    # Acknowledgement requires: message FROM Maintainer <2> with ack keyword
    ack_detected = is_from_m2 and has_ack_keyword

    return ack_detected, keyword_hits, is_from_m2


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
    since_date: Optional[datetime]
) -> dict:
    """
    Scan inbox directory for files matching the request pattern.

    Returns a summary dict with scanned/matches/ack_detected status,
    plus timeline and waiting-clock metadata.
    """
    now = datetime.now(timezone.utc)
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
            "since": since_date.isoformat() if since_date else None
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
        ack_detected, keyword_hits, is_from_m2 = is_acknowledgement(content, keywords)
        actor, direction = detect_actor_and_direction(content)

        match_entry = {
            "file": item.name,
            "path": str(item),
            "size_bytes": meta["size_bytes"],
            "modified_utc": meta["modified_utc"],
            "match_reason": [],
            "keywords_found": keyword_hits,
            "is_from_maintainer_2": is_from_m2,
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

    return results


def write_json_summary(results: dict, output_path: Path) -> None:
    """Write JSON summary to file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def write_markdown_summary(results: dict, output_path: Path) -> None:
    """Write Markdown summary to file."""
    wc = results.get("waiting_clock", {})

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


def main():
    """Main entry point."""
    args = parse_args()

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

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning inbox: {inbox_path}")
    print(f"Request pattern: {args.request_pattern}")
    print(f"Keywords: {keywords}")
    print(f"Output: {output_path}")
    print("")

    # Scan inbox
    results = scan_inbox(inbox_path, args.request_pattern, keywords, since_date)

    # Write outputs
    json_path = output_path / "inbox_scan_summary.json"
    md_path = output_path / "inbox_scan_summary.md"

    write_json_summary(results, json_path)
    write_markdown_summary(results, md_path)

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

    # Return exit code based on ack_detected
    return 0


if __name__ == "__main__":
    sys.exit(main())
