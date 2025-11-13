# Orchestration Guide (galph ↔ ralph)

This README documents the cross‑machine orchestration for the supervisor (galph) and engineer (ralph) loops. It lives alongside the orchestration code (not in `docs/`) to keep project docs and automation harness separate.

## Overview
- Two actors:
  - `supervisor.sh` → `scripts/orchestration/supervisor.py` (galph)
  - `loop.sh` → `scripts/orchestration/loop.py` (ralph)
- Modes:
  - Async: local, back‑to‑back iterations.
  - Sync via Git: strict turn‑taking using `sync/state.json` committed and pushed between machines.
- Wrappers call Python by default; set `ORCHESTRATION_PYTHON=0` to force legacy bash logic.

### Interpreter selection (important)

- Always run orchestration modules with the same interpreter as your active environment.
- Resolve the interpreter once and reuse it in commands:
  ```bash
  PY="$(python - <<'PY'
  import sys; print(sys.executable)
  PY
  )"
  "$PY" -m scripts.orchestration.supervisor --sync-via-git --branch feature/spec-based-2
  "$PY" -m scripts.orchestration.tail_interleave_logs feature-spec-based-2 -n 3
  ```
  Set `PYTHON_BIN` to override in advanced cases; orchestrators respect it for subprocesses.

## State File
- Path: `sync/state.json` (tracked and pushed so both machines see updates)
- Fields:
  - `iteration` (int)
  - `expected_actor` ("galph" | "ralph")
  - `status` ("idle" | "running-galph" | "waiting-ralph" | "running-ralph" | "complete" | "failed")
  - `last_update`, `lease_expires_at` (ISO8601)
  - `galph_commit`, `ralph_commit` (short SHAs)

## Branch Safety (important)
- Always operate on the intended branch; pass `--branch <name>` to both runners.
- The orchestrators will abort if the current branch is not the specified one.
- Pushes use explicit refspecs: `git push origin HEAD:<branch>` to avoid cross‑branch mistakes.

## Defaults
- Iterations: 20 (`--sync-loops N` to change)
- Poll interval: 5s (`--poll-interval S`)
- No max wait by default (`--max-wait-sec S` to enable)
- Loop prompt: `main` (feature work). Switch with `--prompt debug` for parity/trace loops.
- Per‑iteration logs under `logs/` (see Logging).

## Sync via Git (two machines)
1) Preconditions:
   - Both machines share the same remote and branch (e.g., `feature/spec-based-2`).
   - Ensure `sync/state.json` exists; set `expected_actor` to the actor who should start.
2) Start supervisor (galph machine):
   ```bash
   ORCHESTRATION_BRANCH=feature/spec-based-2    ./supervisor.sh --sync-via-git --branch feature/spec-based-2      --sync-loops 20 --logdir logs --verbose --heartbeat-secs 10
   ```
3) Start loop (ralph machine):
   ```bash
   ORCHESTRATION_BRANCH=feature/spec-based-2    ./loop.sh --sync-via-git --branch feature/spec-based-2      --sync-loops 20 --prompt main --logdir logs
   ```
- Handshake:
  - Galph writes: `expected_actor=ralph`, `status=waiting-ralph`.
  - Ralph writes: `status=running-ralph`, then on success sets `expected_actor=galph`, `status=complete`, and increments `iteration`.
  - Supervisor advances when it observes `expected_actor=galph` and `iteration` increased.

## Async (single machine)
- Run without `--sync-via-git` to execute N iterations locally (still writes per‑iteration logs):
  ```bash
  ./supervisor.sh --sync-loops 5 --logdir logs
  ./loop.sh --sync-loops 5 --prompt main --logdir logs
  ```

## Logging
- Descriptive per‑iteration logs:
  - Supervisor: `logs/<branch>/galph/iter-00017_YYYYMMDD_HHMMSS.log`
  - Loop: `logs/<branch>/ralph/iter-00017_YYYYMMDD_HHMMSS_<prompt>.log`
- Configure base directory with `--logdir PATH` (default `logs/`).
- Markdown summaries accompany the raw logs:
  - Supervisor summaries: `logs/<branch>/galph-summaries/iter-00017_YYYYMMDD_HHMMSS-summary.md`
  - Loop summaries: `logs/<branch>/ralph-summaries/iter-00017_YYYYMMDD_HHMMSS-summary.md`
  - See `docs/logging/log_summary_conventions.md` for the naming contract used by automation.
- Supervisor console options:
  - `--verbose`: print state changes to console and log
  - `--heartbeat-secs N`: periodic heartbeat lines while polling
- `logs/` is ignored by Git.

### Viewing recent interleaved logs

Use the helper script to interleave the last N galph/ralph logs (or markdown summaries) for a branch prefix. Entries are matched on iteration number and wrapped in an XML-like tag with CDATA. The tool annotates each log with the post-state commit that stamped the handoff and can optionally snapshot selected directories from that commit:

```bash
"$PY" -m scripts.orchestration.tail_interleave_logs feature-spec-based-2 -n 3
# Summaries instead of raw logs:
"$PY" -m scripts.orchestration.tail_interleave_logs feature-spec-based-2 -n 3 --source summaries
```

Output structure:

```xml
<logs prefix="feature-spec-based-2" count="3" source="logs">
  <log role="galph" iter="141" path="logs/feature-spec-based-2/galph/iter-00141_....log" source="log" format="text" commit="abc1234" commit_subject="[SYNC i=141] actor=galph → next=ralph status=ok ...">
    <![CDATA[
    ...
    ]]>
    <ls path="docs" commit="abc1234">
      <![CDATA[
      docs/workflows/pytorch.md
      ...
      ]]>
    </ls>
  </log>
  <log role="ralph" iter="141" path="logs/feature-spec-based-2/ralph/iter-00141_....log" source="log" format="text" commit="def5678" commit_subject="[SYNC i=142] actor=ralph → next=galph status=ok ...">
    <![CDATA[
    ...
    ]]>
    <!-- Optional ls-tree snapshots repeat for each requested root -->
  </log>
  ...
</logs>
```

Flags of note:

- `-n/--count` tail length (0 = all matching iterations)
- `--min-iter/--max-iter` numeric bounds on iteration selection
- `--no-ls` disables the commit `ls-tree` snapshots
- `--ls-paths docs,plans,reports` overrides which repository roots are listed when `ls` output is enabled
- `--source {logs,summaries}` switches between raw log files and markdown summaries
- `--roles galph,ralph` narrows the interleaved output to a subset of actors (order preserved)

### Manual state handoff (without running a loop)

Use the stamper to flip sync/state.json to the next actor and publish it without executing a supervisor/loop body:

```bash
# Supervisor hands off to Ralph (success)
"$PY" -m scripts.orchestration.stamp_handoff galph ok --branch feature/spec-based-2

# Supervisor marks failure (no handoff)
"$PY" -m scripts.orchestration.stamp_handoff galph fail --branch feature/spec-based-2

# Ralph hands off to Supervisor (success; increments iteration)
"$PY" -m scripts.orchestration.stamp_handoff ralph ok --branch feature-spec-based-2

# Ralph marks failure (no increment)
"$PY" -m scripts.orchestration.stamp_handoff ralph fail --branch feature-spec-based-2
```

Flags:
- `--no-pull` to skip pre-stamp pull; `--no-push` to skip push (local-only)
- `--allow-dirty` to bypass dirty-tree guard (not recommended)

Notes:
- Messages and iteration semantics match the orchestrators: Supervisor stamps at the current iteration; Ralph stamps success at the next iteration.
- The tool updates `last_update`, `lease_expires_at`, and `galph_commit`/`ralph_commit` using the current HEAD.

## Flag Reference
- Supervisor
  - `--sync-via-git` · `--sync-loops N` · `--poll-interval S` · `--max-wait-sec S`
  - `--branch NAME` (abort if not on this branch)
  - `--logdir PATH` (per‑iteration logs)
  - `--verbose` · `--heartbeat-secs N`
  - `--auto-commit-docs` / `--no-auto-commit-docs` (default: on)
    - When enabled, supervisor will auto‑stage+commit changes limited to a doc/meta whitelist after a successful run and before handing off:
      - Whitelist (globs): `input.md`, `galph_memory.md`, `docs/fix_plan.md`, `plans/**/*.md`, `prompts/**/*.md`
      - Files must be ≤ `--max-autocommit-bytes` (default 1,048,576 bytes)
      - Any dirty tracked changes outside the whitelist cause a clear error and the handoff is aborted (no state flip)
    - Configure whitelist via `--autocommit-whitelist a,b,c` and size via `--max-autocommit-bytes N`
  - Reports auto-commit (publishes Galph's evidence by file type)
    - `--auto-commit-reports` / `--no-auto-commit-reports` (default: on)
    - `--report-extensions ".png,.jpeg,.npy,.txt,.md,.json,.log,.py,.c,.h,.sh"` — allowed file types (logs + source files/scripts)
    - `--report-path-globs "glob1,glob2"` — optional glob allowlist (default allows any path); logs/`tmp/` are always skipped
    - `--max-report-file-bytes N` (default 5 MiB) · `--max-report-total-bytes N` (default 20 MiB)
    - `--force-add-reports` (default: on) — force-add files even if ignored by .gitignore
    - Notes: stamp-first handoff ensures reports + state publish together; adjust caps/extension list / path globs as needed for your workflow.
  - Tracked outputs auto-commit (publishes modified tracked artifacts like fixtures)
    - `--auto-commit-tracked-outputs` / `--no-auto-commit-tracked-outputs` (default: on)
    - `--tracked-output-globs "tests/fixtures/**/*.npy,tests/fixtures/**/*.npz,tests/fixtures/**/*.json,tests/fixtures/**/*.pkl"` — path allowlist (glob); only tracked modifications are considered
    - `--tracked-output-extensions ".npy,.npz,.json,.pkl"` — allowed extensions
    - `--max-tracked-output-file-bytes N` (default 32 MiB) · `--max-tracked-output-total-bytes N` (default 100 MiB)
    - Notes: runs before doc/meta hygiene; keeps repo clean when fixture‑like binaries are legitimately regenerated during a supervisor loop. Files exceeding caps remain dirty and will trigger the whitelist guard (handoff abort).
  - `--prepull-auto-commit-docs` / `--no-prepull-auto-commit-docs` (default: on)
    - If the initial git pull fails (e.g., due to local modified files), supervisor now follows a three-step recovery:
      1) Submodule scrub: `git submodule sync --recursive` then `git submodule update --init --recursive --checkout --force` (with manual gitlink align fallback)  
      2) Tracked outputs auto-commit: stage+commit modified fixture-like files within limits (default globs `tests/fixtures/**/*.npy,*.npz,*.json,*.pkl`)  
      3) Doc/meta whitelist auto-commit: stage+commit changes to `input.md`, `galph_memory.md`, `docs/fix_plan.md`, `plans/**/*.md`, `prompts/**/*.md` within size caps
    - The pull is retried after each step; if dirty paths remain outside these guards, the supervisor exits with a clear error.
- Loop
  - `--sync-via-git` · `--sync-loops N` · `--poll-interval S` · `--max-wait-sec S`
  - `--branch NAME` · `--logdir PATH` · `--prompt {main,debug}`
  - Reports auto-commit (publishes Ralph's evidence by file type)
    - `--auto-commit-reports` / `--no-auto-commit-reports` (default: on)
    - `--report-extensions ".png,.jpeg,.npy,.log,.txt,.md,.json,.py,.c,.h,.sh"` — allowed file types (including code diffs/scripts)
    - `--report-path-globs "glob1,glob2"` — optional glob allowlist (default allows any path); logs/`tmp/` are always skipped
    - `--max-report-file-bytes N` (default 5 MiB) · `--max-report-total-bytes N` (default 20 MiB)
    - `--force-add-reports` (default: on) — force-add files even if ignored by .gitignore
    - Notes: stamp-first handoff ensures reports + state publish together; adjust caps/extension list / path globs as needed for your workflow.

## Troubleshooting
- Pull failures: both orchestrators now fail fast on git pull errors (including untracked‑file or local‑modification collisions). Read the console/log message, resolve locally (commit/stash/move), and rerun.
- Index lock robustness: supervisor and loop operations acquire a coarse Git mutex (`.git/orchestrator.lock`) around add/commit/rebase/merge/push to serialize mutations and avoid `.git/index.lock` contention between concurrent processes. A bounded backoff still applies inside commands as a second line of defense.
- Submodule pointer drift: if `.claude/` or other gitlinks appear dirty, the supervisor auto-scrubs submodules (sync + update with `--checkout --force`) before retries. This is idempotent and does not commit pointer bumps; it aligns worktrees to the recorded superproject commits.
- Push rejected / rebase in progress: orchestrators auto‑abort in‑progress rebase before pulling. If conflicts arise, fix them locally, commit, and rerun.
- Branch mismatch: checkout the correct branch or adjust `--branch`.
- Missing prompt: ensure `prompts/<name>.md` exists (default is `main`).

## Notes
- `loop.sh`, `supervisor.sh`, and `input.md` are treated as protected entrypoints elsewhere in the project. Keep wrappers; they manage env and call Python modules by default.
