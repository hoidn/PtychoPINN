# Pyrefly Static Analysis Loop (Run & Fix)

Purpose
- Execute a single, fast loop to run configured static analysis (pyrefly) over `src/`, fix a small set of high‑confidence issues, validate with targeted tests, and commit. Keeps changes minimal and portable.

Autonomy & Messaging
- Do not ask what to do; act on this prompt.
- Only two messages: a short preamble, then the final Loop Output checklist.
- One item per loop; keep the scope minimal and surgical.

Preconditions & Inputs
- Tooling must be configured in‑repo. If `pyrefly` is not installed/available or `[tool.pyrefly]` is missing in `pyproject.toml`, do not introduce new tools in this loop. Instead, record a TODO in `docs/fix_plan.md` with a proposed dev/CI integration and exit after docs update.
- Read these before executing:
  - `./docs/index.md`
  - `./docs/development/testing_strategy.md` (see §1.5 Loop Execution Notes)
  - `./docs/fix_plan.md` (active item must be set `in_progress`)
  - `./input.md` (if present) — prefer its Do Now while maintaining one‑item execution

Ground Rules
- No new tools: use `pyrefly` only if already configured/available.
- Keep changes small: address a focused subset of high‑confidence findings (e.g., undefined names, obvious import errors, dead or unreachable code with zero behavior impact). Avoid API‑breaking refactors.
- Tests: run targeted pytest first; run the full test suite at most once at the end if code changed. For docs‑only loops, use `pytest --collect-only -q`.
- Protected Assets Rule: Do not modify or remove any file referenced in `docs/index.md`.
- Version control hygiene: never commit runtime artifacts (logs, reports); store under `reports/pyrefly/<date>/` locally.
- Subagents: many for search/summarization; at most 1 for running tests/build.

Instructions
<step 0>
- Sync: `git pull --rebase` (timeout 30s; fallback to `--no-rebase` on timeout; resolve conflicts and document in `galph_memory.md`).
- Ensure exactly one active item in `docs/fix_plan.md` is `in_progress` (create/update if needed). Record the exact commands you plan to run.
- Verify preconditions: `rg -n "^\[tool\.pyrefly\]" pyproject.toml` and `command -v pyrefly`.
  - If missing/unavailable: add a TODO entry to `docs/fix_plan.md` (proposed install, CI job, and prompt integration), then proceed to Finalize (docs‑only loop).
</step 0>

<step 1>
- Reproduce: run static analysis and capture artifacts.
  - Primary: `pyrefly check src` (or the configured command) and save stdout/stderr to `reports/pyrefly/<date>/pyrefly.log`.
  - If pyrefly is unavailable but repo linters exist, you MAY run configured linters (e.g., `ruff`, `flake8`, `mypy`) for context only; do not add new tools.
- Parse findings; group by severity (high → low) and type (undefined name, import error, unreachable, dead code, style‑only).
</step 1>

<step 2>
- Selection: choose a minimal, high‑confidence subset to fix in this loop (e.g., 1–5 items) that are unlikely to change public API or behavior.
- Explicitly defer style‑only or large‑scope refactor findings by appending a TODO to `docs/fix_plan.md` with file:line pointers.
</step 2>

<step 3>
- Fixes: implement the minimal changes. Keep patches small and scoped. Do not weaken tests or thresholds.
- If a finding requires a test, author the minimal targeted test first and run it.
</step 3>

<step 4>
- Validation:
  - Run targeted pytest for impacted areas (derive nodes from file paths or `docs/development/testing_strategy.md`).
  - If those pass and code changed, run the full test suite once (`pytest -v`). For docs‑only, run `pytest --collect-only -q`.
</step 4>

<step 5>
- Finalize:
  - Update `docs/fix_plan.md` Attempts History: include a short summary of findings fixed, the exact command(s) run, and artifact paths under `reports/pyrefly/<date>/`.
  - Commit: `git add -A && git commit -m "static-analysis: pyrefly fixes (suite: pass)"` (or `docs-only` when no code changed). Then `git push` (resolve conflicts if needed).
</step 5>

Output
- Problem statement: “Ran pyrefly; fixed high‑confidence findings.”
- Mapped commands executed (pyrefly + pytest nodes).
- Findings summary with file:line pointers (fixed vs. deferred).
- Diff/file list of changes.
- Test results (targeted; full suite once if code changed).
- `docs/fix_plan.md` delta (Attempts History entry, any TODOs added).
- Next actions (e.g., integrate pyrefly in CI; follow‑up items).

Completion Checklist
- Preconditions satisfied or recorded as TODO (no new tools introduced).
- Exactly one small scope executed (1–5 fixes).
- Targeted pytest passed; full suite run at most once if code changed.
- No Protected Assets modified; no runtime artifacts committed.
- Plan updated with artifacts and next actions.
