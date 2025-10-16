from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePath
from subprocess import Popen, PIPE

from .state import OrchestrationState
from .git_bus import safe_pull, add, commit, push_to, short_head, assert_on_branch, current_branch, has_unpushed_commits, push_with_rebase
from .autocommit import autocommit_reports


def _log_file(prefix: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("tmp").mkdir(parents=True, exist_ok=True)
    p = Path("tmp") / f"{prefix}{ts}.txt"
    latest = Path("tmp") / f"{prefix}latest.txt"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(p.name)
    except Exception:
        # Best effort on platforms without symlink support
        pass
    return p


def tee_run(cmd: list[str], stdin_file: Path, log_path: Path) -> int:
    with open(stdin_file, "rb") as fin, open(log_path, "a", encoding="utf-8") as flog:
        flog.write(f"$ {' '.join(cmd)}\n")
        flog.flush()
        proc = Popen(cmd, stdin=fin, stdout=PIPE, stderr=PIPE, text=True, bufsize=1)
        # Stream stdout
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                break
            sys.stdout.write(line)
            flog.write(line)
        # Stream remaining stderr
        err = proc.stderr.read() if proc.stderr else ""
        if err:
            sys.stderr.write(err)
            flog.write(err)
        return proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser(description="Supervisor (galph) orchestrator")
    ap.add_argument("--sync-via-git", action="store_true", help="Enable cross-machine synchronous mode via Git state")
    ap.add_argument("--sync-loops", type=int, default=int(os.getenv("SYNC_LOOPS", 20)), help="Number of iterations to run")
    ap.add_argument("--poll-interval", type=int, default=int(os.getenv("POLL_INTERVAL", 5)))
    ap.add_argument("--max-wait-sec", type=int, default=int(os.getenv("MAX_WAIT_SEC", 0)))
    ap.add_argument("--state-file", type=Path, default=Path(os.getenv("STATE_FILE", "sync/state.json")))
    ap.add_argument("--codex-cmd", type=str, default=os.getenv("CODEX_CMD", "codex"))
    ap.add_argument("--branch", type=str, default=os.getenv("ORCHESTRATION_BRANCH", ""), help="Expected Git branch to operate on")
    ap.add_argument("--verbose", action="store_true", help="Print state changes to console during polling")
    ap.add_argument("--heartbeat-secs", type=int, default=int(os.getenv("HEARTBEAT_SECS", "0")), help="Console heartbeat interval while polling (0=off)")
    ap.add_argument("--logdir", type=Path, default=Path("logs"), help="Base directory for per-iteration logs (default: logs/)")
    # Auto-commit doc/meta hygiene (supervisor only)
    ap.add_argument("--auto-commit-docs", dest="auto_commit_docs", action="store_true",
                    help="Auto-stage+commit doc/meta whitelist when dirty (default: on)")
    ap.add_argument("--no-auto-commit-docs", dest="auto_commit_docs", action="store_false",
                    help="Disable auto commit of doc/meta whitelist")
    ap.set_defaults(auto_commit_docs=True)
    ap.add_argument("--autocommit-whitelist", type=str,
                    default="input.md,galph_memory.md,docs/fix_plan.md,plans/**/*.md,prompts/**/*.md",
                    help="Comma-separated glob whitelist for supervisor auto-commit (doc/meta only)")
    ap.add_argument("--max-autocommit-bytes", type=int, default=int(os.getenv("MAX_AUTOCOMMIT_BYTES", "1048576")),
                    help="Maximum per-file size (bytes) eligible for auto-commit")
    # Pre-pull auto-commit: try to auto-commit doc/meta before initial pull if pull fails due to local mods
    ap.add_argument("--prepull-auto-commit-docs", dest="prepull_auto_commit_docs", action="store_true",
                    help="If initial git pull fails, attempt doc/meta whitelist auto-commit then retry (default: on)")
    ap.add_argument("--no-prepull-auto-commit-docs", dest="prepull_auto_commit_docs", action="store_false",
                    help="Disable pre-pull doc/meta auto-commit fallback")
    ap.set_defaults(prepull_auto_commit_docs=True)
    # Reports auto-commit (supervisor evidence publishing)
    ap.add_argument("--auto-commit-reports", dest="auto_commit_reports", action="store_true",
                    help="Auto-stage+commit report artifacts by file extension after run (default: on)")
    ap.add_argument("--no-auto-commit-reports", dest="auto_commit_reports", action="store_false",
                    help="Disable auto commit of report artifacts")
    ap.set_defaults(auto_commit_reports=True)
    ap.add_argument("--report-extensions", type=str,
                    default=os.getenv("SUPERVISOR_REPORT_EXTENSIONS", ".png,.jpeg,.npy,.txt,.md,.json,.log,.py,.c,.h,.sh"),
                    help="Comma-separated list of allowed report file extensions (lowercase, with dots)")
    ap.add_argument("--max-report-file-bytes", type=int, default=int(os.getenv("SUPERVISOR_MAX_REPORT_FILE_BYTES", "5242880")),
                    help="Maximum per-file size (bytes) eligible for reports auto-commit (default 5 MiB)")
    ap.add_argument("--max-report-total-bytes", type=int, default=int(os.getenv("SUPERVISOR_MAX_REPORT_TOTAL_BYTES", "20971520")),
                    help="Maximum total size (bytes) staged per iteration for reports (default 20 MiB)")
    ap.add_argument("--force-add-reports", dest="force_add_reports", action="store_true",
                    help="Force-add report files even if ignored (.gitignore) (default: on)")
    ap.add_argument("--no-force-add-reports", dest="force_add_reports", action="store_false",
                    help="Do not force-add ignored report files")
    ap.set_defaults(force_add_reports=True)
    ap.add_argument("--report-path-globs", type=str,
                    default=os.getenv("SUPERVISOR_REPORT_PATH_GLOBS", ""),
                    help="Comma-separated glob allowlist for report auto-commit paths (default: none)")
    args, unknown = ap.parse_known_args()

    report_path_globs = tuple(p.strip() for p in args.report_path_globs.split(',') if p.strip())
    logdir_prefix_parts = tuple(part for part in PurePath(args.logdir).parts if part not in {"", "."})

    def _within(parts: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
        return bool(prefix) and parts[:len(prefix)] == prefix

    def _skip_reports(path: str) -> bool:
        parts = PurePath(path).parts
        if _within(parts, logdir_prefix_parts):
            return True
        if parts and parts[0] == "tmp":
            return True
        return False

    # Helpers shared by pre-pull and post-run auto-commit paths
    def _list(cmd: list[str]) -> list[str]:
        from subprocess import run, PIPE
        cp = run(cmd, stdout=PIPE, stderr=PIPE, text=True)
        if cp.returncode != 0:
            return []
        return [p for p in (cp.stdout or "").splitlines() if p.strip()]

    def _matches_any(path: str, patterns: list[str]) -> bool:
        return any(fnmatch.fnmatch(path, pat) for pat in patterns)

    def _supervisor_autocommit_docs(args_ns, log_func) -> tuple[bool, list[str]]:
        """
        Attempt to auto-stage+commit doc/meta whitelist changes.
        Returns (committed: bool, forbidden_paths: list[str]).
        If forbidden_paths non-empty, caller should abort handoff.
        """
        whitelist = [p.strip() for p in args_ns.autocommit_whitelist.split(',') if p.strip()]
        max_bytes = args_ns.max_autocommit_bytes
        # Collect dirty paths (modifications and untracked)
        unstaged_mod = _list(["git", "diff", "--name-only", "--diff-filter=M"])
        staged_mod = _list(["git", "diff", "--cached", "--name-only", "--diff-filter=AM"])
        untracked = _list(["git", "ls-files", "--others", "--exclude-standard"])
        dirty_all = sorted(set(unstaged_mod) | set(staged_mod) | set(untracked))
        allowed: list[str] = []
        forbidden: list[str] = []
        for p in dirty_all:
            if _matches_any(p, whitelist):
                try:
                    if os.path.isfile(p) and os.path.getsize(p) <= max_bytes:
                        allowed.append(p)
                    else:
                        forbidden.append(p)
                except FileNotFoundError:
                    forbidden.append(p)
            else:
                forbidden.append(p)

        if forbidden:
            if log_func:
                log_func("Auto-commit blocked by forbidden dirty paths: " + ", ".join(forbidden))
            return False, forbidden

        committed = False
        if allowed:
            add(allowed)
            body = "\n\nFiles:\n" + "\n".join(f" - {p}" for p in allowed)
            committed = commit("SUPERVISOR AUTO: doc/meta hygiene — tests: not run" + body)
            # Do not push here; let subsequent logic push state or later commits
        return committed, []

    def _pull_with_error(log_func, ctx: str) -> bool:
        """Run safe_pull while capturing log lines; on failure, print last error."""
        buf: list[str] = []
        def _cap(msg: str) -> None:
            if log_func:
                log_func(msg)
            buf.append(msg)
        ok = safe_pull(_cap)
        if not ok:
            err_line = None
            for line in reversed(buf):
                low = line.lower()
                if ("error" in low) or ("fatal" in low) or ("would be overwritten" in low):
                    err_line = line
                    break
            if err_line:
                print(f"[sync] ERROR ({ctx}): {err_line}")
            else:
                print(f"[sync] ERROR ({ctx}): git pull failed; see iter log.")
        return ok

    # per-iteration logger factory
    def make_logger(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        def _log(msg: str) -> None:
            with open(path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        return _log

    # Branch guard (if provided) and target branch resolution
    if args.branch:
        assert_on_branch(args.branch, lambda m: None)
        branch_target = args.branch
    else:
        branch_target = current_branch()

    if not args.sync_via_git:
        # Legacy async mode: run N iterations back-to-back
        for _ in range(args.sync_loops):
            rc = tee_run([args.codex_cmd, "exec", "-m", "gpt-5-codex", "-c", "model_reasoning_effort=high", "--dangerously-bypass-approvals-and-sandbox"], Path("prompts/supervisor.md"), log_path)
            if rc != 0:
                return rc
        return 0

    # Sync via Git
    # session notice (to console only)
    print(f"[sync] supervisor: SYNC via git for {args.sync_loops} iteration(s)")
    args.state_file.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(args.sync_loops):
        # Determine iteration-specific log file
        # Use current state when available to derive iteration number
        if not _pull_with_error(lambda m: None, "pre-pull"):
            # Pre-pull fallback: attempt doc/meta auto-commit if enabled, then retry pull
            if args.prepull_auto_commit_docs:
                committed, forbidden = _supervisor_autocommit_docs(args, lambda m: None)
                if committed and not forbidden:
                    if not _pull_with_error(lambda m: None, "pre-pull (after autocommit)"):
                        return 1
                else:
                    print("[sync] ERROR: git pull failed; pre-pull auto-commit not applicable or blocked.")
                    return 1
            else:
                # Error already printed by _pull_with_error
                return 1
        st_probe = OrchestrationState.read(str(args.state_file))
        itnum = st_probe.iteration
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        iter_log = args.logdir / branch_target.replace('/', '-') / "galph" / f"iter-{itnum:05d}_{ts}.log"
        logp = make_logger(iter_log)

        if not _pull_with_error(logp, "pre-wait"):
            if args.prepull_auto_commit_docs:
                committed, forbidden = _supervisor_autocommit_docs(args, logp)
                if committed and not forbidden:
                    if not _pull_with_error(logp, "pre-wait (after autocommit)"):
                        return 1
                else:
                    print("[sync] ERROR: git pull failed; pre-pull auto-commit not applicable or blocked.")
                    return 1
            else:
                # Error already printed by _pull_with_error
                return 1

        # Resume mode: if a local stamped handoff exists but isn't pushed yet, publish and skip work
        st_local_probe = OrchestrationState.read(str(args.state_file))
        if (st_local_probe.expected_actor != "galph" or st_local_probe.status in {"waiting-ralph", "failed"}) and has_unpushed_commits():
            logp("[sync] Detected local stamped handoff with unpushed commits; attempting push-only reconciliation.")
            if not push_with_rebase(branch_target, logp):
                print("[sync] ERROR: failed to push local stamped handoff; resolve and retry.")
                return 1
            # Continue to next supervisor iteration
            continue

        # Initialize state if missing
        if not args.state_file.exists():
            st_init = OrchestrationState()
            st_init.expected_actor = "galph"
            st_init.status = "idle"
            st_init.write(str(args.state_file))
            add([str(args.state_file)])
            commit("[SYNC init] actor=galph status=idle")
            push_to(branch_target, logp)

        # Wait for our turn (always executed)
        logp("Waiting for expected_actor=galph...")
        start = time.time()
        last_hb = start
        prev_state = None
        while True:
            if not _pull_with_error(logp, "polling"):
                return 1
            st = OrchestrationState.read(str(args.state_file))
            cur_state = (st.expected_actor, st.status, st.iteration)
            if args.verbose and cur_state != prev_state:
                print(f"[sync] state: actor={st.expected_actor} status={st.status} iter={st.iteration}")
                logp(f"[sync] state: actor={st.expected_actor} status={st.status} iter={st.iteration}")
                prev_state = cur_state
            if st.expected_actor == "galph":
                break
            if args.max_wait_sec and (time.time() - start) > args.max_wait_sec:
                logp("Timeout waiting for galph turn")
                return 1
            if args.heartbeat_secs and (time.time() - last_hb) >= args.heartbeat_secs:
                elapsed = int(time.time() - start)
                msg = f"[sync] waiting for turn (galph)… {elapsed}s elapsed"
                print(msg)
                logp(msg)
                last_hb = time.time()
            time.sleep(args.poll_interval)

        # Mark running
        st = OrchestrationState.read(str(args.state_file))
        st.status = "running-galph"
        st.write(str(args.state_file))
        add([str(args.state_file)])
        commit(f"[SYNC i={st.iteration}] actor=galph status=running")
        push_to(branch_target, logp)

        # Execute one supervisor iteration
        rc = tee_run([args.codex_cmd, "exec", "-m", "gpt-5-codex", "-c", "model_reasoning_effort=high", "--dangerously-bypass-approvals-and-sandbox"], Path("prompts/supervisor.md"), iter_log)

        sha = short_head()

        # Auto-commit reports evidence (before stamping)
        if args.auto_commit_reports:
            allowed = {e.strip().lower() for e in args.report_extensions.split(',') if e.strip()}
            autocommit_reports(
                allowed_extensions=allowed,
                max_file_bytes=args.max_report_file_bytes,
                max_total_bytes=args.max_report_total_bytes,
                force_add=args.force_add_reports,
                logger=logp,
                commit_message_prefix="SUPERVISOR AUTO: reports evidence — tests: not run",
                skip_predicate=_skip_reports,
                allowed_path_globs=report_path_globs,
            )

        # Determine post-run success without early-returning
        post_ok = (rc == 0)
        if args.auto_commit_docs:
            committed_docs, forbidden = _supervisor_autocommit_docs(args, logp)
            if forbidden:
                logp("[sync] Doc/meta whitelist violation detected; marking post-run as failed.")
                post_ok = False

        # Stamp FIRST (idempotent): write local handoff or failure before any pushes
        st = OrchestrationState.read(str(args.state_file))
        if post_ok:
            st.stamp(expected_actor="ralph", status="waiting-ralph", galph_commit=sha)
            st.write(str(args.state_file))
            add([str(args.state_file)])
            commit(f"[SYNC i={st.iteration}] actor=galph → next=ralph status=ok galph_commit={sha}")
        else:
            st.stamp(expected_actor="galph", status="failed", galph_commit=sha)
            st.write(str(args.state_file))
            add([str(args.state_file)])
            commit(f"[SYNC i={st.iteration}] actor=galph status=fail galph_commit={sha}")

        # Publish stamped state. If push fails, exit; restart will resume push without re-running work.
        if not push_with_rebase(branch_target, logp):
            print("[sync] ERROR: failed to push stamped state; resolve and relaunch to resume push.")
            return 1
        if not post_ok:
            logp(f"Supervisor iteration failed rc={rc}. Stamped failure and pushed; exiting.")
            return rc

        # Wait for Ralph to finish and increment iteration
        logp(f"Waiting for Ralph to complete i={st.iteration}...")
        current_iter = st.iteration
        start2 = time.time()
        last_hb2 = start2
        prev_state2 = None
        while True:
            if not _pull_with_error(logp, "wait-for-ralph"):
                return 1
            st2 = OrchestrationState.read(str(args.state_file))
            cur_state2 = (st2.expected_actor, st2.status, st2.iteration)
            if args.verbose and cur_state2 != prev_state2:
                print(f"[sync] state: actor={st2.expected_actor} status={st2.status} iter={st2.iteration}")
                logp(f"[sync] state: actor={st2.expected_actor} status={st2.status} iter={st2.iteration}")
                prev_state2 = cur_state2
            if st2.expected_actor == "galph" and st2.iteration > current_iter:
                logp(f"Ralph completed iteration {current_iter}; proceeding to {st2.iteration}")
                break
            if args.heartbeat_secs and (time.time() - last_hb2) >= args.heartbeat_secs:
                elapsed = int(time.time() - start2)
                msg = f"[sync] waiting for ralph… {elapsed}s elapsed (i={current_iter})"
                print(msg)
                logp(msg)
                last_hb2 = time.time()
            time.sleep(args.poll_interval)

    logp("Supervisor SYNC loop finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
