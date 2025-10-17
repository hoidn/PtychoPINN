# Loop Prompt Engineering Guide

Purpose
- Provide a portable, repeatable recipe for writing “loop prompts” that actually execute work each iteration (no greetings, no stalls), update the plan, validate with authoritative tests, and commit appropriate changes.
- Prevent the failure mode where an agent asks questions instead of acting, or validates the wrong artifact/script.

Core Principles
- Autonomy first: the prompt must instruct the agent to choose work itself from the plan and begin immediately.
- Single focus per loop: select exactly one high-value item and attempt only that.
- Doc-driven mapping: resolve test commands/env from documentation (Parity Profile / Implementation fields), not from ad-hoc scripts.
- Minimal work per loop: always run at least one authoritative test and capture baseline metrics/artifacts before making changes.
- Hard gates: require plan ledger updates, metrics, and authoritative test pass before declaring success.
- Clean commits: commit successful code changes; on failures/partials, commit only docs/prompt/spec updates.
- Portable by default: project specifics live in docs; prompts only reference them.
- PyTorch discipline: enforce device/dtype neutrality (CPU + CUDA smoke checks) and ensure tooling lives under structured `scripts/` paths with the documented environment contract.

Required Sections for Every Loop Prompt
1) Autonomy Contract
- Do not ask the user what to do; select work from `docs/fix_plan.md`.
- If an item is `in_progress`, continue it; else pick the highest‑priority `pending` item and mark `in_progress`.
- Only two messages per loop: a brief “next action” preamble and a final Loop Output checklist.

2) Authoritative Inputs
- Plan: `docs/fix_plan.md` (single source of loop work)
- Supervisor steering memo: `./input.md` (written by galph each run; use its "Do Now" to steer which single item to execute this loop)
- Specs: `specs/ptychodus_api_spec.md` and `specs/data_contracts.md` (Acceptance Tests, API, and data contracts)
- Parity Profile: where AT→tests/env/commands mapping lives (e.g., `docs/TESTING_GUIDE.md`, section “Parallel Validation Matrix”, or `docs/development/TEST_SUITE_INDEX.md`)
- Architecture docs as needed (`docs/architecture.md`, `docs/DEVELOPER_GUIDE.md`)

3) Minimal Actions Per Loop (hard rule)
- Read `./input.md` (if present); prefer its "Do Now" selection while maintaining one-item execution per loop.
- Read `docs/fix_plan.md`; pick/confirm the active item (you may switch to match the Do Now and must record the switch in Attempts History).
- Map AT→pytest command via Parity Profile (or fall back by searching tests for the AT ID).
- Execute the mapped test(s) to capture a baseline; save metrics/artifacts.

4) Steps (high‑level)
- Setup → Reproduce → Triage → Fix → Gate → Finalize

5) Gates (hard requirements)
- Start gate: plan shows chosen item set to `in_progress` with reproduction commands (and aligns with `input.md` Do Now when provided).
- End gate: plan Attempt entry appended with Metrics + Artifacts (real paths) + First Divergence (if traces used).
- Final sanity: re‑run mapped authoritative tests and pass thresholds before declaring success.

6) Version Control Hygiene
- PASS: stage and commit all intended changes (code/docs/spec/prompt); include AT IDs and suite result in the message.
- FAIL/PARTIAL with rollback: stage docs/spec/prompt only (no reverted code); commit with metrics and status.
- Never commit runtime artifacts (traces, binaries, images); store under run‑local artifacts directories.

7) Messaging Policy
- No greetings or open‑ended questions.
- Brief preamble (“what’s next”) → do the work → final checklist.

Main‑Style Loop Templates (use these as scaffolds)

Main loop (feature/implementation)
```
IMPORTANT ROUTING
- If this is a debugging/equivalence loop, STOP and use prompts/debug.md.

<ground rules>
- One thing per loop; pick from docs/fix_plan.md
- No placeholders; follow spec + arch
- Run authoritative tests; then full suite
</ground rules>

<instructions>
<step 0>
- Read ./docs/index.md, specs/ptychodus_api_spec.md, specs/data_contracts.md, docs/architecture.md, docs/TESTING_GUIDE.md
- Read docs/fix_plan.md; confirm one item is in_progress (else pick highest‑priority pending and set it)
</step>

<step 1>
- Map the Acceptance Test(s) to exact pytest command via docs/TESTING_GUIDE.md (Acceptance Mapping), docs/development/TEST_SUITE_INDEX.md, or spec “Implementation:” lines
- Run the mapped test(s); capture baseline; save metrics/artifacts
</step>

<step 2>
- Implement the chosen task (single scope). Use repo conventions and arch structure
</step>

<step 3>
- Gate: run pytest -v from repo root; zero fails and zero collection errors
</step>

<step 4>
- Update docs/fix_plan.md (Attempts History: metrics/artifacts/next actions)
- Commit (stage all intended changes)
</step>
</instructions>

Process hints
- Prefer ripgrep
- Keep changes minimal and documented

Output
- Problem statement + spec quotes
- Search summary and file pointers
- Diff/file list
- Test results
- Fix plan delta
</Output>
```

Debug loop (equivalence/trace)
```
IMPORTANT ROUTING
- This loop is for equivalence discrepancies (parity, correlation, structured diff)

<ground rules>
- One thing per loop from docs/fix_plan.md
- Parallel trace mandatory; geometry‑first triage
- Do not change tests or thresholds
</ground rules>

<instructions>
<step 0>
- Read docs; set item to in_progress in docs/fix_plan.md
- Locate Parity Profile in docs/TESTING_GUIDE.md (Parallel Validation Matrix)
</step>

<step 1>
- Map AT→pytest + env from the Matrix (or spec “Implementation:”)
- Run mapped tests; capture stdout/stderr and metrics; save diff heatmap if relevant
</step>

<step 2>
- Geometry‑first triage (units, conventions, pivots, +0.5, F/S)
</step>

<step 3>
- Generate C and PyTorch traces (on‑peak pixel) and identify FIRST DIVERGENCE
</step>

<step 4>
- Apply the minimal fix; re‑run failing case + neighbors
</step>

<step 5>
- Final sanity: re‑run mapped pytest tests under required env; pass thresholds
- If regressions or thresholds not met: rollback code changes; keep artifacts
</step>

<step 6>
- Update docs/fix_plan.md (Attempts History: metrics, artifacts, first divergence, next actions)
- Commit: PASS → code+docs; FAIL/PARTIAL → docs/spec/prompt only
</step>
</instructions>

Output
- AT(s) addressed; mapped tests/commands; env used
- Metrics (corr, rmse, max_abs, sum_ratio); artifacts paths
- First divergence (var, file:line)
- Plan delta; next actions
</Output>
```

Fallback Behavior
- If the Parity Profile is missing, search tests for the AT ID (e.g., `rg -n "AT-PARALLEL-002" tests/`) and derive a mapping; record the documentation gap as a TODO in the plan.
- If live parity env vars are required but unavailable, run golden‑data ATs and record the limitation explicitly.

Spec Integration (Traceability)
- Add an `Implementation:` line under each Acceptance Test in specs pointing to its authoritative pytest file (and required env for live parity).
- Keep the Acceptance Mapping/Matrix in docs consistent with specs (prefer one source and generate the other).

Anti‑Patterns to Avoid
- Greeting/asking what to work on at loop start.
- Validating against supportive scripts instead of authoritative tests.
- Declaring success without re‑running the mapped tests under the required env.
- Failing to update the plan ledger with metrics/artifacts and first divergence.
- Committing runtime artifacts.

Author Checklist (before shipping a loop prompt)
- [ ] Autonomy contract present (“don’t ask; pick from plan”).
- [ ] Authoritative Inputs reference the Parity Profile/Mapping in docs.
- [ ] Minimal work per loop is specified (map→run authoritative tests→capture baseline).
- [ ] Steps and Gates are explicit and minimal.
- [ ] Final sanity gate requires mapped tests to pass.
- [ ] Version control hygiene defined for PASS and FAIL/PARTIAL.
- [ ] Template sections present (routing, ground rules, instructions, process hints, output, completion checklist).
- [ ] Prompt is portable (no hard‑coded project env; everything read from docs).

Quick Templates

Preamble (one‑liner)
- “Continuing AT‑XXX (in_progress): mapping to tests and reproducing baseline.”

Commit message (PASS)
- `<AT-ids> <module|debug>: <concise summary> (suite: pass)`

Commit message (FAIL/PARTIAL with rollback)
- `<AT-ids> debug: attempt #N failed/partial; metrics recorded; code reverted`
