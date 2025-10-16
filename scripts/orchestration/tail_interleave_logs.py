#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from subprocess import run, PIPE
from typing import Dict, List, Tuple

LOG_NAME_RE = re.compile(r"^iter-(\d+)_.*\.log$")


def find_logs(root: Path) -> Dict[int, Path]:
    """Scan a role directory for iter-*.log files and return {iter: path}."""
    found: Dict[int, Path] = {}
    if not root.exists():
        return found
    for p in sorted(root.glob("iter-*.log")):
        m = LOG_NAME_RE.match(p.name)
        if not m:
            continue
        try:
            it = int(m.group(1))
        except ValueError:
            continue
        found[it] = p
    return found


def interleave_last(
    prefix: Path,
    count: int,
    out,
    include_ls: bool = True,
    ls_roots: List[str] | None = None,
    min_iter: int | None = None,
    max_iter: int | None = None,
) -> int:
    """Print last N interleaved galph & ralph logs under logs/<prefix>/*.

    Output uses XML-like tags per log with CDATA wrapping.
    Returns 0 on success, non-zero otherwise.
    """
    galph_dir = Path("logs") / prefix / "galph"
    ralph_dir = Path("logs") / prefix / "ralph"

    galph_logs = find_logs(galph_dir)
    ralph_logs = find_logs(ralph_dir)

    # Load recent SYNC commits to annotate post-state commits
    post_commit = load_post_state_commits()
    ls_cache: Dict[str, Dict[str, List[str]]] = {}
    if ls_roots is None:
        ls_roots = ["docs", "plans", "reports"]

    if not galph_logs and not ralph_logs:
        print(f"No logs found under {galph_dir} or {ralph_dir}", file=sys.stderr)
        return 2

    # Build union of iterations and take last N by numeric iteration
    all_iters = sorted(set(galph_logs.keys()) | set(ralph_logs.keys()))
    if not all_iters:
        print(f"No iter-*.log files found under {prefix}", file=sys.stderr)
        return 3
    # Apply optional min/max iteration filtering
    if min_iter is not None or max_iter is not None:
        filt = []
        for it in all_iters:
            if min_iter is not None and it < min_iter:
                continue
            if max_iter is not None and it > max_iter:
                continue
            filt.append(it)
        all_iters = filt
        if not all_iters:
            print("No iterations within the requested min/max bounds.", file=sys.stderr)
            return 4
    # Apply tail selection (count<=0 means include all)
    tail_iters = all_iters[-count:] if count and count > 0 else all_iters

    # Emit a header for clarity
    out.write(f"<logs prefix=\"{prefix}\" count=\"{count}\">\n")

    for it in tail_iters:
        # Supervisor first, then Loop for each iteration
        if it in galph_logs:
            p = galph_logs[it]
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                content = f"<error reading {p}: {e}>\n"
            csha, csubj = resolve_post_commit("galph", it, post_commit)
            commit_attr = f" commit=\"{csha}\" commit_subject=\"{xml_attr_escape(csubj)}\"" if csha else ""
            out.write(f"  <log role=\"galph\" iter=\"{it}\" path=\"{p}\"{commit_attr}>\n")
            out.write("    <![CDATA[\n")
            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")
            out.write("    ]]>\n")
            if include_ls and csha:
                ls_map = ls_cache.get(csha)
                if ls_map is None:
                    ls_map = ls_tree_at(csha, ls_roots)
                    ls_cache[csha] = ls_map
                for root in ls_roots:
                    files = ls_map.get(root, [])
                    out.write(f"    <ls path=\"{root}\" commit=\"{csha}\">\n")
                    out.write("      <![CDATA[\n")
                    for f in files:
                        out.write(f"{f}\n")
                    out.write("      ]]>\n")
                    out.write("    </ls>\n")
            out.write("  </log>\n")

        if it in ralph_logs:
            p = ralph_logs[it]
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                content = f"<error reading {p}: {e}>\n"
            csha, csubj = resolve_post_commit("ralph", it, post_commit)
            commit_attr = f" commit=\"{csha}\" commit_subject=\"{xml_attr_escape(csubj)}\"" if csha else ""
            out.write(f"  <log role=\"ralph\" iter=\"{it}\" path=\"{p}\"{commit_attr}>\n")
            out.write("    <![CDATA[\n")
            out.write(content)
            if not content.endswith("\n"):
                out.write("\n")
            out.write("    ]]>\n")
            if include_ls and csha:
                ls_map = ls_cache.get(csha)
                if ls_map is None:
                    ls_map = ls_tree_at(csha, ls_roots)
                    ls_cache[csha] = ls_map
                for root in ls_roots:
                    files = ls_map.get(root, [])
                    out.write(f"    <ls path=\"{root}\" commit=\"{csha}\">\n")
                    out.write("      <![CDATA[\n")
                    for f in files:
                        out.write(f"{f}\n")
                    out.write("      ]]>\n")
                    out.write("    </ls>\n")
            out.write("  </log>\n")

    out.write("</logs>\n")
    return 0


def xml_attr_escape(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def load_post_state_commits(max_commits: int = 2000):
    """
    Parse recent SYNC/state commits to map post-state commits for galph and ralph.
    Returns a dict with keys:
      { ('galph','ok',iter): (sha, subj), ('galph','fail',iter): (...),
        ('ralph','ok_for_log',iter): (sha, subj), ('ralph','fail',iter): (...) }
    Note: ralph OK commits appear at iter+1; we pre-map them to the log's iter via 'ok_for_log'.
    """
    cmd = [
        "git", "log", f"--max-count={max_commits}", "--pretty=format:%h\t%s", "--", "sync/state.json"
    ]
    cp = run(cmd, stdout=PIPE, stderr=PIPE, text=True)
    lines = (cp.stdout or "").splitlines()
    mapping = {}
    re_sync = re.compile(r"\[SYNC i=(\d+)\]\s+actor=(galph|ralph)(?:\s+→ next=(galph|ralph)\s+status=(\w+))?\s*(.*)")
    for line in lines:
        try:
            sha, subj = line.split("\t", 1)
        except ValueError:
            continue
        m = re_sync.search(subj)
        if not m:
            continue
        it = int(m.group(1))
        actor = m.group(2)
        next_actor = m.group(3)
        status = m.group(4) or ""
        # Success handoff commits have the arrow; failure are 'status=fail'
        if actor == "galph":
            if next_actor == "ralph":  # success at same iter
                mapping[("galph", "ok", it)] = (sha, subj)
            elif "status=fail" in subj:
                mapping[("galph", "fail", it)] = (sha, subj)
        elif actor == "ralph":
            if next_actor == "galph":  # success appears at iter+1; map to log iter (iter-1)
                mapping[("ralph", "ok_for_log", it - 1)] = (sha, subj)
            elif "status=fail" in subj:
                mapping[("ralph", "fail", it)] = (sha, subj)
    return mapping


def resolve_post_commit(role: str, log_iter: int, mapping) -> Tuple[str, str]:
    """Return (sha, subject) for the post-state commit associated with this log.
    - galph: prefer ('galph','ok',iter), else ('galph','fail',iter)
    - ralph: prefer ('ralph','ok_for_log',iter), else ('ralph','fail',iter)
    """
    if role == "galph":
        return mapping.get(("galph", "ok", log_iter), mapping.get(("galph", "fail", log_iter), ("", "")))
    else:
        return mapping.get(("ralph", "ok_for_log", log_iter), mapping.get(("ralph", "fail", log_iter), ("", "")))


def ls_tree_at(sha: str, roots: List[str]) -> Dict[str, List[str]]:
    """Return {root: files} for the given commit sha using git ls-tree.
    Roots should be top-level paths like 'docs', 'plans', 'reports'.
    """
    if not sha:
        return {}
    # Prepare args; only include existing roots to avoid noisy stderr
    args = ["git", "ls-tree", "-r", "--name-only", sha, "--", *roots]
    cp = run(args, stdout=PIPE, stderr=PIPE, text=True)
    files = [ln for ln in (cp.stdout or "").splitlines() if ln.strip()]
    by_root: Dict[str, List[str]] = {r: [] for r in roots}
    for f in files:
        for r in roots:
            if f == r or f.startswith(r + "/"):
                by_root[r].append(f)
                break
    return by_root


def main() -> int:
    ap = argparse.ArgumentParser(description="Interleave the last N galph/ralph logs with matching iteration numbers, and annotate with post-state commits and file listings.")
    ap.add_argument("prefix", type=str, help="Branch prefix under logs/ (e.g., 'feature-spec-based-2')")
    ap.add_argument("-n", "--count", type=int, default=5, help="How many iterations to include (default: 5)")
    ap.add_argument("--no-ls", dest="include_ls", action="store_false", help="Do not include git ls-tree listings for docs/plans/reports")
    ap.add_argument("--ls-paths", type=str, default="docs,plans,reports", help="Comma-separated roots to ls-tree (default: docs,plans,reports)")
    ap.add_argument("--min-iter", type=int, default=None, help="Minimum iteration to include (inclusive)")
    ap.add_argument("--max-iter", type=int, default=None, help="Maximum iteration to include (inclusive)")
    args = ap.parse_args()

    prefix = Path(args.prefix)
    if prefix.parts and (prefix.parts[0] == "logs"):
        # Accept both 'feature-...' and 'logs/feature-...'
        prefix = Path(*prefix.parts[1:])

    ls_roots = [p.strip() for p in args.ls_paths.split(',') if p.strip()]
    return interleave_last(
        prefix,
        args.count,
        sys.stdout,
        include_ls=args.include_ls,
        ls_roots=ls_roots,
        min_iter=args.min_iter,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    raise SystemExit(main())
