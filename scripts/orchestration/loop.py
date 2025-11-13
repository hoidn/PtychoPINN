from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path, PurePath
from subprocess import Popen, PIPE

from .state import OrchestrationState
from .git_bus import safe_pull, add, commit, push_to, short_head, has_unpushed_commits, assert_on_branch, current_branch, push_with_rebase, git_lock
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
        pass
    return p


def tee_run(cmd: list[str], stdin_file: Path, log_path: Path) -> int:
    with open(stdin_file, "rb") as fin, open(log_path, "a", encoding="utf-8") as flog:
        flog.write(f"$ {' '.join(cmd)}\n")
        flog.flush()
        proc = Popen(cmd, stdin=fin, stdout=PIPE, stderr=PIPE, text=True, bufsize=1)
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                break
            sys.stdout.write(line)
            flog.write(line)
        err = proc.stderr.read() if proc.stderr else ""
        if err:
            sys.stderr.write(err)
            flog.write(err)
        return proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser(description="Engineer (ralph) orchestrator")
    ap.add_argument("--sync-via-git", action="store_true", help="Enable cross-machine synchronous mode via Git state")
    ap.add_argument("--sync-loops", type=int, default=int(os.getenv("SYNC_LOOPS", 20)))
    ap.add_argument("--poll-interval", type=int, default=int(os.getenv("POLL_INTERVAL", 5)))
    ap.add_argument("--max-wait-sec", type=int, default=int(os.getenv("MAX_WAIT_SEC", 0)))
    ap.add_argument("--state-file", type=Path, default=Path(os.getenv("STATE_FILE", "sync/state.json")))
    ap.add_argument("--claude-cmd", type=str, default=os.getenv("CLAUDE_CMD", "/home/ollie/.claude/local/claude"))
    ap.add_argument("--prompt", type=str, choices=["main", "debug"], default=os.getenv("LOOP_PROMPT", "main"), help="Select which prompt to run (default: main)")
    ap.add_argument("--branch", type=str, default=os.getenv("ORCHESTRATION_BRANCH", ""))
    ap.add_argument("--logdir", type=Path, default=Path("logs"), help="Base directory for per-iteration logs (default: logs/)")
    # Reports auto-commit (engineer evidence publishing)
    ap.add_argument("--auto-commit-reports", dest="auto_commit_reports", action="store_true",
                    help="Auto-stage+commit report artifacts by file extension after run (default: on)")
    ap.add_argument("--no-auto-commit-reports", dest="auto_commit_reports", action="store_false",
                    help="Disable auto commit of report artifacts")
    ap.set_defaults(auto_commit_reports=True)
    ap.add_argument("--report-extensions", type=str,
                    default=os.getenv("REPORT_EXTENSIONS", ".png,.jpeg,.npy,.log,.txt,.md,.json,.py,.c,.h,.sh"),
                    help="Comma-separated list of allowed report file extensions (lowercase, with dots)")
    ap.add_argument("--max-report-file-bytes", type=int, default=int(os.getenv("MAX_REPORT_FILE_BYTES", "5242880")),
                    help="Maximum per-file size (bytes) eligible for reports auto-commit (default 5 MiB)")
    ap.add_argument("--max-report-total-bytes", type=int, default=int(os.getenv("MAX_REPORT_TOTAL_BYTES", "20971520")),
                    help="Maximum total size (bytes) staged per iteration for reports (default 20 MiB)")
    ap.add_argument("--force-add-reports", dest="force_add_reports", action="store_true",
                    help="Force-add report files even if ignored (.gitignore) (default: on)")
    ap.add_argument("--no-force-add-reports", dest="force_add_reports", action="store_false",
                    help="Do not force-add ignored report files")
    ap.set_defaults(force_add_reports=True)
    ap.add_argument("--report-path-globs", type=str,
                    default=os.getenv("REPORT_PATH_GLOBS", ""),
                    help="Comma-separated glob allowlist for report auto-commit paths (default: none)")

    args, unknown = ap.parse_known_args()

    log_path = _log_file("claudelog")
    report_path_globs = tuple(p.strip() for p in args.report_path_globs.split(',') if p.strip())
    logdir_prefix_parts = tuple(part for part in PurePath(args.logdir).parts if part not in {"", "."})
    skip_config_path = Path(os.getenv("REPORT_SKIP_CONFIG", ".reportsignore"))
    skip_prefix_specs: tuple[tuple[str, ...], ...] = tuple()
    if skip_config_path.exists():
        specs: list[tuple[str, ...]] = []
        for raw_line in skip_config_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.partition("#")[0].strip()
            if not line:
                continue
            parts = tuple(part for part in PurePath(line).parts if part not in {"", "."})
            if parts:
                specs.append(parts)
        skip_prefix_specs = tuple(specs)

    def _within(parts: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
        return bool(prefix) and parts[:len(prefix)] == prefix

    def _skip_reports(path: str) -> bool:
        parts = PurePath(path).parts
        if _within(parts, logdir_prefix_parts):
            return True
        if parts and parts[0] == "tmp":
            return True
        for spec in skip_prefix_specs:
            if spec and parts[:len(spec)] == spec:
                return True
        return False

    def logp(msg: str) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _pull_with_error(logger, ctx: str) -> bool:
        buf: list[str] = []
        def cap(m: str) -> None:
            logger(m)
            buf.append(m)
        # When a specific branch is known, pass it to avoid ambiguous pull configuration
        try:
            bt = branch_target  # may be set later; closure resolves at call time
        except NameError:
            bt = None
        ok = safe_pull(cap, "origin", bt) if bt else safe_pull(cap)
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

    # (reports auto-commit now shared via autocommit.autocommit_reports)

    def _prepull_reports_autocommit(logger) -> None:
        """Proactively commit reports under plans/active/**/reports/** before pulling.

        This reduces chances that local evidence files block `git pull --rebase`.
        Large/binary artifacts remain skipped via extension and size caps.
        """
        if not args.auto_commit_reports:
            return
        try:
            allowed_exts = {e.strip().lower() for e in args.report_extensions.split(',') if e.strip()}
            autocommit_reports(
                allowed_extensions=allowed_exts,
                max_file_bytes=args.max_report_file_bytes,
                max_total_bytes=args.max_report_total_bytes,
                force_add=args.force_add_reports,
                logger=logger,
                commit_message_prefix="RALPH AUTO PRE-PULL: reports evidence — autopublish",
                skip_predicate=_skip_reports,
                allowed_path_globs=("plans/active/**/reports/**",),
            )
        except Exception:
            # Pre-pull hygiene is best-effort; continue to pull fallback if this fails
            pass

    # Branch guard / resolution
    if args.branch:
        assert_on_branch(args.branch, lambda m: None)
        branch_target = args.branch
    else:
        branch_target = current_branch()

    # Always keep up to date (pre-pull reports publish, then pull)
    _prepull_reports_autocommit(logp)
    ok_initial = _pull_with_error(logp, "initial")
    if not ok_initial:
        print("[sync] ERROR: git pull failed; see iter log for details (likely untracked-file collisions).")
        print("[sync] Remediation: move or remove the conflicting untracked files, then re-run the loop.")
        return 1

    for _ in range(args.sync_loops):
        # Compute per-iteration log path (branch/prompt aware)
        _prepull_reports_autocommit(logp)
        ok_probe = _pull_with_error(lambda m: None, "probe")
        if not ok_probe:
            # Error line already printed
            return 1
        st_probe = OrchestrationState.read(str(args.state_file))
        itnum = st_probe.iteration
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        iter_log = args.logdir / branch_target.replace('/', '-') / "ralph" / f"iter-{itnum:05d}_{ts}_{args.prompt}.log"

        # Resume mode: if a local stamped handoff exists but isn't pushed yet, publish and skip work
        st_local = OrchestrationState.read(str(args.state_file))
        if (st_local.expected_actor != "ralph" or st_local.status in {"complete", "failed"}) and has_unpushed_commits():
            def logp(msg: str) -> None:
                iter_log.parent.mkdir(parents=True, exist_ok=True)
                with open(iter_log, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            logp("[sync] Detected local stamped handoff with unpushed commits; attempting push-only reconciliation.")
            if not push_with_rebase(branch_target, logp):
                print("[sync] ERROR: failed to push local stamped handoff; resolve and retry.")
                return 1
            continue

        if args.sync_via_git:
            args.state_file.parent.mkdir(parents=True, exist_ok=True)
            # Logger bound to this iteration
            def logp(msg: str) -> None:
                iter_log.parent.mkdir(parents=True, exist_ok=True)
                with open(iter_log, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")

            logp("[SYNC] Waiting for expected_actor=ralph...")
            start = time.time()
            while True:
                _prepull_reports_autocommit(logp)
                if not _pull_with_error(logp, "polling"):
                    # Error line already printed
                    return 1
                st = OrchestrationState.read(str(args.state_file))
                if st.expected_actor == "ralph":
                    break
                if args.max_wait_sec and (time.time() - start) > args.max_wait_sec:
                    logp("[SYNC] Timeout waiting for turn; exiting")
                    return 1
                time.sleep(args.poll_interval)

            # Mark running
            st.status = "running-ralph"
            st.write(str(args.state_file))
            from .git_bus import git_lock as _git_lock
            with _git_lock(logp):
                add([str(args.state_file)])
                commit(f"[SYNC i={st.iteration}] actor=ralph status=running")
                push_to(branch_target, logp)

        # Execute one engineer loop
        prompt_path = Path("prompts") / f"{args.prompt}.md"
        if not prompt_path.exists():
            logp(f"ERROR: prompt file not found: {prompt_path}")
            return 2
        rc = tee_run([args.claude_cmd, "-p", "--dangerously-skip-permissions", "--verbose", "--output-format", "stream-json"], prompt_path, iter_log)

        # Auto-commit reports evidence (before stamping) — constrained by extension and size caps
        if args.auto_commit_reports:
            allowed = {e.strip().lower() for e in args.report_extensions.split(',') if e.strip()}
            autocommit_reports(
                allowed_extensions=allowed,
                max_file_bytes=args.max_report_file_bytes,
                max_total_bytes=args.max_report_total_bytes,
                force_add=args.force_add_reports,
                logger=logp,
                commit_message_prefix="RALPH AUTO: reports evidence — tests: not run",
                skip_predicate=_skip_reports,
                allowed_path_globs=report_path_globs,
            )

        # Complete handoff (stamp-first, idempotent)
        sha = short_head()
        st = OrchestrationState.read(str(args.state_file))

        if args.sync_via_git:
            # STAMP FIRST (idempotent)
            from .git_bus import git_lock as _git_lock
            if rc == 0:
                st.stamp(expected_actor="galph", status="complete", increment=True, ralph_commit=sha)
                st.write(str(args.state_file))
                with _git_lock(logp):
                    add([str(args.state_file)])
                    commit(f"[SYNC i={st.iteration}] actor=ralph → next=galph status=ok ralph_commit={sha}")
            else:
                st.stamp(expected_actor="ralph", status="failed", increment=False, ralph_commit=sha)
                st.write(str(args.state_file))
                with _git_lock(logp):
                    add([str(args.state_file)])
                    commit(f"[SYNC i={st.iteration}] actor=ralph status=fail ralph_commit={sha}")

            # Publish stamped state. If push fails, exit; restart resumes push.
            with _git_lock(logp):
                push_ok = push_with_rebase(branch_target, logp)
            if not push_ok:
                print("[sync] ERROR: failed to push stamped state; resolve and relaunch to resume push.")
                return 1
            if rc != 0:
                logp(f"Loop failed rc={rc}. Stamped failure and pushed; exiting.")
                return rc

        # Optional: push local commits from the loop (async hygiene)
        if rc == 0 and has_unpushed_commits():
            try:
                from .git_bus import git_lock as _git_lock
                with _git_lock(logp):
                    push_to(branch_target, logp)
            except Exception as e:
                logp(f"WARNING: git push failed: {e}")
                return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
