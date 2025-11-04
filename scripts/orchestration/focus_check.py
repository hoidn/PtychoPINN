#!/usr/bin/env python3
"""
Focus Check — Validate plan premises against repository reality.

Usage:
  focus_check.py --artifact docs/TESTING_GUIDE.md --expect exists
  focus_check.py --artifact docs/some_new_file.md --expect missing

Exits non-zero on mismatch. Use in supervisor preflight before drafting input.md.
"""
import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="Path to file/dir to check")
    ap.add_argument("--expect", choices=["exists", "missing"], required=True)
    args = ap.parse_args()

    exists = os.path.exists(args.artifact)
    if args.expect == "exists" and not exists:
        print(f"ERROR: Expected to find {args.artifact}, but it is missing", file=sys.stderr)
        return 2
    if args.expect == "missing" and exists:
        print(f"ERROR: Expected {args.artifact} to be missing, but it exists", file=sys.stderr)
        return 3
    print(f"OK: {args.artifact} -> {args.expect}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

