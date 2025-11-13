#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from subprocess import run, PIPE
from typing import Any, Dict, Optional


def sh(cmd: list[str], check: bool = False) -> tuple[int, str, str]:
    cp = run(cmd, stdout=PIPE, stderr=PIPE, text=True)
    if check and cp.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{cp.stderr}")
    return cp.returncode, cp.stdout, cp.stderr


def current_branch() -> str:
    return sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], True)[1].strip()


def short_head() -> str:
    return sh(["git", "rev-parse", "--short", "HEAD"], True)[1].strip()


def safe_pull() -> bool:
    rc, out, err = sh(["git", "pull", "--rebase"])
    if rc != 0:
        # Try fallback
        sh(["git", "rebase", "--abort"])
        rc2, out2, err2 = sh(["git", "pull", "--no-rebase"])
        return rc2 == 0
    return True


def _parse_state(raw: str, ctx: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: state data for {ctx} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"ERROR: state data for {ctx} must contain a JSON object")
    return data


def _default_state() -> Dict[str, Any]:
    return {
        "iteration": 1,
        "expected_actor": "galph",
        "status": "idle",
        "galph_commit": None,
        "ralph_commit": None,
    }


def _load_state_from_history(path: Path) -> Optional[Dict[str, Any]]:
    rel = path.as_posix()
    rc, log_out, _ = sh(["git", "log", "--format=%H", "--", rel])
    if rc != 0:
        return None
    revs = [sha.strip() for sha in log_out.splitlines() if sha.strip()]
    if not revs:
        return None

    best_state: Optional[Dict[str, Any]] = None
    best_iteration = -1
    for sha in revs:
        rc_blob, blob, _ = sh(["git", "show", f"{sha}:{rel}"])
        if rc_blob != 0:
            continue
        raw = blob.strip()
        if not raw:
            continue
        try:
            data = _parse_state(raw, f"{sha}:{rel}")
        except SystemExit as exc:
            print(f"WARNING: {exc}", file=sys.stderr)
            continue
        try:
            iter_val = int(data.get("iteration", 0))
        except (TypeError, ValueError):
            iter_val = 0
        if iter_val > best_iteration:
            best_iteration = iter_val
            best_state = data
    return best_state


def _load_state(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    stripped = raw.strip()
    state: Optional[Dict[str, Any]] = None
    if stripped:
        state = _parse_state(stripped, str(path))

    history = _load_state_from_history(path)
    if state is None:
        if history is not None:
            return history
        return _default_state()

    if history is not None:
        try:
            cur_iter = int(state.get("iteration", 0))
        except (TypeError, ValueError):
            cur_iter = 0
        try:
            hist_iter = int(history.get("iteration", 0))
        except (TypeError, ValueError):
            hist_iter = 0
        if hist_iter > cur_iter:
            print(f"WARNING: local {path} iteration={cur_iter} is behind git history iteration={hist_iter}; using history snapshot",
                  file=sys.stderr)
            return history

    return state


def main() -> int:
    ap = argparse.ArgumentParser(description="Stamp sync/state.json handoff without running the loop body.")
    ap.add_argument("actor", choices=["galph", "ralph"], help="Actor performing the stamp")
    ap.add_argument("result", choices=["ok", "fail"], help="Result of the loop body to stamp")
    ap.add_argument("--state-file", type=Path, default=Path("sync/state.json"))
    ap.add_argument("--branch", type=str, default="", help="Expected branch; assert before stamping")
    ap.add_argument("--no-pull", action="store_true", help="Do not pull before stamping")
    ap.add_argument("--no-push", action="store_true", help="Do not push after stamping")
    ap.add_argument("--allow-dirty", action="store_true", help="Allow dirty working tree (use with care)")
    args = ap.parse_args()

    # Branch guard
    if args.branch:
        cur = current_branch()
        if cur != args.branch:
            raise SystemExit(f"ERROR: on branch '{cur}', expected '{args.branch}'")

    # Pull for freshness
    if not args.no_pull and not safe_pull():
        print("WARNING: git pull failed; proceeding with local state", file=sys.stderr)

    # Dirty guard
    if not args.allow_dirty:
        rc, _, _ = sh(["git", "diff", "--quiet"])
        rc2, _, _ = sh(["git", "diff", "--cached", "--quiet"])
        if rc != 0 or rc2 != 0:
            raise SystemExit("ERROR: working tree has uncommitted changes; commit/stash or pass --allow-dirty")

    # Load state
    if not args.state_file.exists():
        raise SystemExit(f"ERROR: state file not found: {args.state_file}")
    state = _load_state(args.state_file)
    iter_no = int(state.get("iteration", 1))

    now = datetime.now(timezone.utc)
    state["last_update"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    state["lease_expires_at"] = (now + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")

    sha = short_head()
    commit_iter = iter_no

    # Apply stamp semantics
    if args.actor == "galph":
        state["galph_commit"] = sha
        if args.result == "ok":
            state["expected_actor"] = "ralph"
            state["status"] = "waiting-ralph"
            subject = f"[SYNC i={commit_iter}] actor=galph → next=ralph status=ok galph_commit={sha}"
        else:
            state["expected_actor"] = "galph"
            state["status"] = "failed"
            subject = f"[SYNC i={commit_iter}] actor=galph status=fail galph_commit={sha}"
    else:  # ralph
        state["ralph_commit"] = sha
        if args.result == "ok":
            # Increment iteration on success
            state["iteration"] = iter_no + 1
            commit_iter = iter_no + 1
            state["expected_actor"] = "galph"
            state["status"] = "complete"
            subject = f"[SYNC i={commit_iter}] actor=ralph → next=galph status=ok ralph_commit={sha}"
        else:
            state["expected_actor"] = "ralph"
            state["status"] = "failed"
            subject = f"[SYNC i={commit_iter}] actor=ralph status=fail ralph_commit={sha}"

    # Write state
    args.state_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    # Commit
    sh(["git", "add", str(args.state_file)], True)
    sh(["git", "commit", "-m", subject], True)

    if not args.no_push:
        # Push to current branch explicitly
        branch = current_branch()
        rc, out, err = sh(["git", "push", "origin", f"HEAD:{branch}"])
        if rc != 0:
            # Try rebase/push
            safe_pull()
            sh(["git", "push", "origin", f"HEAD:{branch}"], True)

    print(f"Stamped: {subject}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
