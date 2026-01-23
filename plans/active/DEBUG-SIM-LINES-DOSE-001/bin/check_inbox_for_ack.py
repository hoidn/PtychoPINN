#!/usr/bin/env python3
"""
check_inbox_for_ack.py - CLI to scan inbox directories for maintainer acknowledgements.

This non-production CLI scans an inbox directory for files referencing a given
request pattern, detects acknowledgement keywords, and emits JSON/Markdown summaries.

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
from typing import Optional


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


def is_from_maintainer_2(content: str) -> bool:
    """
    Check if the message is FROM Maintainer <2>.

    Looks for patterns like:
    - "From: Maintainer <2>"
    - "**From:** Maintainer <2>"
    """
    content_lower = content.lower()

    # Pattern: From: Maintainer <2> (with optional markdown bold)
    from_patterns = [
        r"from:\s*\*?\*?maintainer\s*<\s*2\s*>",
        r"\*\*from:\*\*\s*maintainer\s*<\s*2\s*>",
    ]

    for pattern in from_patterns:
        if re.search(pattern, content_lower):
            return True

    return False


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


def scan_inbox(
    inbox_path: Path,
    request_pattern: str,
    keywords: list[str],
    since_date: Optional[datetime]
) -> dict:
    """
    Scan inbox directory for files matching the request pattern.

    Returns a summary dict with scanned/matches/ack_detected status.
    """
    results = {
        "scanned": 0,
        "matches": [],
        "ack_detected": False,
        "ack_files": [],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "inbox": str(inbox_path),
            "request_pattern": request_pattern,
            "keywords": keywords,
            "since": since_date.isoformat() if since_date else None
        }
    }

    if not inbox_path.exists():
        print(f"WARNING: Inbox path does not exist: {inbox_path}")
        return results

    pattern_lower = request_pattern.lower()

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

        match_entry = {
            "file": item.name,
            "path": str(item),
            "size_bytes": meta["size_bytes"],
            "modified_utc": meta["modified_utc"],
            "match_reason": [],
            "keywords_found": keyword_hits,
            "is_from_maintainer_2": is_from_m2,
            "ack_detected": ack_detected,
            "preview": truncate_preview(content)
        }

        if filename_match:
            match_entry["match_reason"].append("filename")
        if content_match:
            match_entry["match_reason"].append("content")

        results["matches"].append(match_entry)

        if ack_detected:
            results["ack_detected"] = True
            results["ack_files"].append(item.name)

    return results


def write_json_summary(results: dict, output_path: Path) -> None:
    """Write JSON summary to file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def write_markdown_summary(results: dict, output_path: Path) -> None:
    """Write Markdown summary to file."""
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

    if results["matches"]:
        lines.extend([
            "## Matching Files",
            "",
            "| File | Match Reason | From M2? | Keywords | Ack | Modified |",
            "|------|--------------|----------|----------|-----|----------|",
        ])
        for m in results["matches"]:
            reason = ", ".join(m["match_reason"])
            from_m2 = "Yes" if m.get("is_from_maintainer_2") else "No"
            keywords = ", ".join(m["keywords_found"]) if m["keywords_found"] else "-"
            ack = "Yes" if m["ack_detected"] else "No"
            lines.append(f"| `{m['file']}` | {reason} | {from_m2} | {keywords} | {ack} | {m['modified_utc']} |")

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

    if results["matches"]:
        print("Matching files:")
        for m in results["matches"]:
            reason = ", ".join(m["match_reason"])
            ack_str = " [ACK]" if m["ack_detected"] else ""
            print(f"  - {m['file']} ({reason}){ack_str}")

    print("")
    print(f"JSON summary: {json_path}")
    print(f"Markdown summary: {md_path}")

    # Return exit code based on ack_detected
    return 0


if __name__ == "__main__":
    sys.exit(main())
