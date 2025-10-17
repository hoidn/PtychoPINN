# Debugging Loop Prompt (Single-Loop, Trace-Driven, No‑Cheating)

Purpose
- This prompt governs any loop labeled “debugging” and any loop triggered by AT‑PARALLEL failures, correlation below thresholds, large max absolute differences, or visible structured discrepancies.
- It encodes strict guardrails, mandatory parallel trace‑driven validation, geometry‑first triage, quantitative checkpoints beyond correlation, clear success/fail gates with rollback, and a mandatory fix plan update process.

Autonomy & Interaction Policy (No‑Question Mode)
- Before reading context or selecting work, run `git pull --rebase` to sync with origin. Resolve conflicts immediately (especially docs/fix_plan.md), recording outcomes in galph_memory.md and the loop output.
- Read `./input.md` (if present). If it contains a "Do Now", prefer that selection for this loop while maintaining one‑item execution; record any switch in `docs/fix_plan.md` Attempts History.
- Do not ask the user what to work on. Select work autonomously from `docs/fix_plan.md`.
- If an item is already `in_progress`, continue it OR switch to the `input.md` Do Now if provided and justified; in either case, ensure exactly one item is attempted this loop.
- Produce brief preambles and then act: read files, search, run tests, generate traces, update plan, and commit per gates.
- Only two messages per loop: a short “next action” preamble before the first command, and the final Loop Output checklist. No idle greetings.
 - Include the exact pytest command(s) in the Do Now and use them for reproduction; if no test exists, write the minimal targeted test first and then run it. Author or refactor tests in native pytest style—do not inherit from `unittest.TestCase` or mix pytest parametrization/fixtures with unittest APIs.
 - Prefer using `scripts/validation/*` scripts for reproducible validations when available.

Use When (Triggers)
- Any AT‑PARALLEL test fails or correlation < required threshold in specs/spec‑a‑parallel.md
- Max absolute diff is large or diff heatmap shows structured errors
- Beam/geometry invariances break (e.g., pixel‑size invariance, beam center in pixels)
- Suspected detector/convention/unit/pivot issues

Callchain Tracing Prerequisite (when locus is unclear)
- If the failing behavior’s locus is not obvious, first run `prompts/callchain.md` to build a question‑driven static callgraph and propose numeric tap points:
  analysis_question: "<describe the failing behavior or suspected factor>"
  initiative_id: "<short‑slug>"
  scope_hints: ["normalization", "scaling", "geometry", "IO"]
  roi_hint: "<minimal ROI>"
  namespace_filter: "<project primary package>"
- Consume its outputs (`callchain/static.md`, `trace/tap_points.md`) to place numeric TRACE taps; do not guess tap points.

Non‑Negotiable Guardrails
1) Never change tests, tolerances, or acceptance thresholds to “pass.” Do not weaken spec gates.
2) Mandatory parallel trace‑driven debugging for equivalence issues. Produce both C and PyTorch traces and identify the FIRST DIVERGENCE. Instrumentation must log the exact values computed by the production helpers (see docs/architecture.md) rather than re‑deriving physics in the trace path.
3) Geometry‑first triage: units, conventions, pivots, MOSFLM +0.5 pixel, F/S mapping, twotheta axis → CUSTOM switch, r‑factor/close_distance.
4) Quantitative checkpoints beyond correlation: report correlation, MSE/RMSE, max abs diff, total sums and ratios, and attach a diff heatmap.
5) Rollback: If metrics regress or equivalence is not achieved without changing tests, revert functional changes in this loop and escalate with artifacts.
6) Fix Plan Updates are MANDATORY at loop START and END using `prompts/update_fix_plan.md`:
   - At START: select exactly one high‑value item and mark it `in_progress` (“one item per loop” means one item ATTEMPTED per loop).
   - At END: update Attempts History with metrics, artifacts, first divergence, and Next Actions. If failed/partial, KEEP the item active (do not mark done) and add diagnostics for future loops.
7) Hard Gate — Fix Plan Compliance:
    - Start Gate: Do not proceed to reproduction until `docs/fix_plan.md` shows the chosen item set to `Status: in_progress` and contains Reproduction commands.
    - End Gate: Do not commit unless `docs/fix_plan.md` shows a NEW “Attempts History” entry for this loop with lines starting with `Metrics:` and `Artifacts:` (real, dated paths), and `First Divergence:` if found.
8) Bootstrap Parity Harness (blocking):
   - If the documented parity harness (see `docs/TESTING_GUIDE.md` / `docs/development/TEST_SUITE_INDEX.md`) is missing for the targeted AT, PAUSE. Add a fix_plan item, author the shared pytest harness + case definitions, and rerun Step 0 before touching the failing AT. The harness should be parameterised to exercise both reference and candidate pipelines (e.g., TensorFlow vs PyTorch entry points), enforce corr/MSE/RMSE/max|Δ|/sum ratios, and write `metrics.json` artifacts on failure.
   - Minimal YAML entry for the failing AT must include `id`, `base_args`, `thresholds`, `runs:[name, extra_args]` (and seeds where needed). Do not continue until the files exist and are referenced by the matrix.
9) Matrix Gate — Canonical Parity First:
    - In equivalence loops, the FIRST command MUST be the mapped C↔Py parity path (pytest node from the Matrix or the canonical harness listed next to it). Do not begin with PyTorch‑only tests.
    - Honor the documented parity matrix: if the AT has an entry in the mapping, run the corresponding harness/test before any auxiliary diagnostics.
    - Confirm `tests/test_parity_matrix.py` imports cleanly; if the import fails or the file is missing, return to Step 0 and bootstrap the harness before proceeding.
10) Test cadence: run targeted tests first; execute the full test suite at most once per loop at the end if code changed. For prompt/docs‑only loops, use `pytest --collect-only -q`.
- If the parity harness depends on external artifacts (baseline binaries, datasets, model bundles), resolve them via project docs (e.g., `docs/TESTING_GUIDE.md` or plan artifacts). Verify the expected path exists (datasets under `datasets/`, models under `plans/active/<initiative>/reports/`) before running tests; otherwise record a TODO and stop.
11) Contradiction Rule — Parity Wins:
   - **⚠️⚠️⚠️ IMPORTANT — STOP IMMEDIATELY IF PARITY FAILS!**
   - If any mapped parity path (pytest or harness) reports under‑threshold metrics for an AT with a parity threshold, the loop is a **FAILURE** regardless of pytest green on PyTorch‑only tests.
   - Do **NOT** commit. Do **NOT** exit the loop. Reopen fix_plan, capture artifacts (metrics, traces, diff heatmaps), and proceed directly to trace‑first debugging until the canonical parity command passes.

Authoritative Inputs (consult before acting)
- CLAUDE.md (core rules, detector gotchas, “Parallel Trace Debugging is Mandatory”)
- specs/ptychodus_api_spec.md and specs/data_contracts.md (normative API/data requirements)
- docs/TESTING_GUIDE.md (parallel trace SOP; golden parity)
- docs/development/TEST_SUITE_INDEX.md (map selectors to concrete test modules)
- Project Parity Profile (required for equivalence loops): whichever documentation section maps Acceptance Tests to concrete test files, required environment, and commands (e.g., a "Parallel Validation Matrix" entry in testing docs)
- docs/debugging/debugging.md, docs/debugging/QUICK_REFERENCE_PARAMS.md, docs/debugging/TROUBLESHOOTING.md (known failure patterns, geometry/params gotchas)
- docs/DEVELOPER_GUIDE.md and docs/architecture.md (architectural intent, two-system model, trace helpers)
- prompts/main.md (loop mechanics). For debugging loops, this prompt supersedes where stricter.

Loop Objective (single loop)
- Fix one root‑cause class deterministically, validated by traces and metrics. No test edits. No threshold edits. Produce artifacts.

Subagents Playbook (required delegation)
- test-failure-analyzer: Resolve AT→pytest node(s) + exact environment from the Parallel Validation Matrix; output the exact command lines to run. If mapping is missing, derive it and add a minimal Matrix entry plus a TODO in docs/fix_plan.md.
- debugger: Generate aligned C and PyTorch traces for the same pixel; compute FIRST DIVERGENCE (variable + file:line); output artifact paths.
- python-pro: Compute quantitative checkpoints (corr, MSE/RMSE, max|Δ|, sum ratios) and render diff heatmaps.
- issue: Only if root cause is a spec/test gap; draft precise spec shard edits without weakening thresholds and append a TODO to docs/fix_plan.md.
- code-reviewer: Pre‑commit scan of the changed scope for security/perf/config risks.

<ground rules>
- Autonomy (No‑Question Mode): Do not ask the user what to work on. Select work from `docs/fix_plan.md`. If an item is `in_progress`, continue it; otherwise pick the highest‑priority `pending` item and set it `in_progress`.
- One thing per loop. No placeholders. Never change tests/thresholds to pass.
- Parity Profile: Use the project’s mapping (docs/TESTING_GUIDE.md → “Parallel Validation Matrix”) to resolve AT→pytest and required env. If missing, search tests for the AT ID and record the doc gap.
- Mandatory parallel trace‑driven debugging; geometry‑first triage.
- Authoritative validation = the mapped parity path (pytest via Matrix or documented harness). PyTorch‑only scripts remain supportive diagnostics.
- Version control hygiene: PASS → commit code+docs; FAIL/PARTIAL (with rollback) → commit docs/spec/prompt only; never commit runtime artifacts.
</ground rules>

<instructions>
<step 0>
- Read: `./docs/index.md`, `./specs/ptychodus_api_spec.md`, `./specs/data_contracts.md`, `./docs/architecture.md`, `./docs/TESTING_GUIDE.md`
- Read `docs/fix_plan.md`; confirm a single active item is `in_progress` (else pick highest‑priority `pending` and set it)
- Locate the Parity Profile section (Parallel Validation Matrix) **and** confirm the matching case exists in the documented harness/tests (see `docs/TESTING_GUIDE.md` / `docs/development/TEST_SUITE_INDEX.md`). If any required test module or case definition is missing or fails to import, STOP: add a fix_plan item to bootstrap the parity harness, create/repair the missing asset(s), then restart Step 0. Do **not** proceed to reproduction until the parity harness entry exists and maps to the target AT.
- Start Gate: Ensure `docs/fix_plan.md` shows the chosen item as `in_progress` with reproduction commands before proceeding.
</step 0>

<step 1>
- Map AT→parity command and required env from the Parity Profile / harness definition. Export env; run the mapped parity path (pytest node or documented harness) first. If the command fails because the harness/case is missing, immediately bootstrap it (see Step 0 mandate) before rerunning. Capture stdout/stderr, executed case ID, metrics, and save a diff heatmap if relevant.
- Subagent: test‑failure‑analyzer (when failures present) to cluster errors and produce focused repro.
</step 1>

<step 2>
- Geometry‑First Triage (units, conventions [+0.5 MOSFLM], F/S mapping, pivot BEAM/SAMPLE, r‑factor/close_distance, invariances)
- Subagent (conditional): architect‑review when geometry/ADR updates are implicated; include a 1–3 line ADR impact in artifacts.
</step 2>

<step 3>
- Parallel Trace‑Driven Validation: Generate aligned C and PyTorch traces (on‑peak pixel; identical names/units). Identify the FIRST DIVERGENCE and stop to root‑cause it.
- Subagent: debugger to drive first‑divergence isolation and propose the minimal corrective change.
</step 3>

<step 4>
- Narrow & Fix: Apply the smallest change that fixes the FIRST DIVERGENCE. Prioritize geometry before physics. Re‑run the failing case + close neighbors; regenerate traces if geometry/units changed.
</step 4>

<step 4.5>
- Static analysis (if configured): Run `pyrefly check src` to catch obvious issues before parity/regression tests. Address high‑confidence findings that relate to the changed scope. Do not introduce or install new tools mid‑loop; skip this step if pyrefly is not already configured for this repo.
</step 4.5>

<step 5>
- Pass/Fail Gates & Rollback:
  • Final Sanity (Hard Gate): Re‑run the mapped authoritative tests under required env; thresholds must pass.
  • Pass only if all spec gates pass (e.g., corr/MSE/max|Δ|/sum ratio within thresholds; no NaNs/Infs; heatmaps show low‑level residue).
  • If fails/regresses, rollback functional edits, keep artifacts, and escalate hypotheses + traces in the plan.
</step 5>

<step 6>
- Finalize:
  • Run full `pytest -v` (ensure zero failures and zero collection errors).
  • Update `docs/fix_plan.md` (Attempts History: metrics, artifacts, FIRST DIVERGENCE, next actions). Include Parity Profile location, exact test files executed, env set (redact secrets), and exact commands used.
  • Subagent: code‑reviewer (pre‑commit) to catch security/performance/config risks; address high/critical findings.
  • Version control hygiene:
    – PASS: `git add -A && git commit -m "<AT-ids> debug: <concise summary> (suite: pass)"`
    – FAIL/PARTIAL with rollback: stage docs/spec/prompt updates only (no reverted code) and `git commit -m "<AT-ids> debug: attempt #N failed/partial; metrics recorded; code reverted"`
</step 6>

</instructions>

Process hints
- Use brief preambles (what’s next). Prefer `rg` for search. Save artifacts under a dated folder. Use float64 for debug; ROI allowed when helpful.
- Always do real work each loop: at minimum, read `docs/fix_plan.md`, map AT→pytest via the Parity Profile, execute the mapped tests, and capture baseline metrics/artifacts.

SOP — Step‑by‑Step (follow in order)
0) Setup & Context
   - Identify the failing AT(s), exact thresholds, and reproduction commands. Pin device/dtype (float64 for debug). Reduce to a small ROI if needed (spec allows ROI for debug).
   - Update docs/fix_plan.md at LOOP START using `prompts/update_fix_plan.md`: pick one item, set `Status: in_progress`, record reproduction commands and planned approach.
   - Locate the project's Parity Profile (AT→test mapping + required env/commands). Typical location: testing strategy docs under a section like "Parallel Validation Matrix". If missing:
     • Add a TODO in docs/fix_plan.md to author the Parity Profile section (include proposed location/title).
     • Fallback: derive mapping by searching tests for the AT identifier or symptom keywords; record the derived mapping and note the documentation gap.
   - If this loop addresses external equivalence (e.g., C↔Py/golden parity), derive the required environment variables and test commands from the Parity Profile and record them in the plan’s Reproduction section.
   - Matrix Gate (hard preflight):
    • Resolve the mapped pytest node(s) for the AT and verify any required reference artifacts (datasets, baseline outputs, cached models) exist. If they are missing, halt and add a TODO in the plan to regenerate or document the source before proceeding.
     • Prohibition: Do not run PyTorch‑only tests in parity loops before the mapped C‑parity run.
     • Subagent handoff: invoke test-failure-analyzer to emit the exact command lines to run and debugger to plan trace generation for a specific pixel.
     • Missing mapping (blocking): If the Matrix lacks a parity mapping for an AT with a parity threshold, first add the mapping (pytest or harness with pass/fail) or treat the harness as the mapped path, then rerun parity.
   - Hard Gate (verify): Ensure the plan reflects this loop’s active item and start entry before proceeding (see Guardrail 7).

1) Reproduce Canonically
   - AT→Test mapping (use Parity Profile): Map the failing acceptance item/symptom to concrete test file(s) and required environment. Export the required environment exactly as specified by the profile. If the profile is absent, use the fallback mapping from Setup and make a note in the plan.
   - Run the mapped test(s) using the profile’s canonical command(s). Capture stdout/stderr and list of executed test paths. Record both in the plan.
   - Subagent: test-failure-analyzer — Provide the failing test path/pattern and context. Capture canonical error messages, stack traces, clustered failures, and exact repro commands. Attach its report, then run the reproduced command(s) to verify.
   - Reproduce the exact failing case (e.g., AT‑PARALLEL‑002 pixel sizes: 0.05, 0.1, 0.2, 0.4 mm; fixed detector size; fixed beam center in mm). Record: image shape, correlation, MSE/RMSE, max abs diff, total sums and sum ratio.
   - Save diff heatmap (log1p|Δ|) and peak diagnostics if relevant.

2) Geometry‑First Triage (Detector Checklist)
   - Units: detector geometry in meters; physics in Å. Verify conversions at boundaries.
   - Convention: MOSFLM axis/basis, +0.5 pixel on beam centers; F/S mapping; CUSTOM switch when `-twotheta_axis` is explicit.
   - Pivot: BEAM vs SAMPLE per flags/headers; r‑factor ratio; close_distance update and distance = close_distance / r.
   - Invariances: beam center in pixels scales as 1/pixel_size; pixel coordinate mapping; omega formula Ω=(pixel_size²/R²)·(close_distance/R) or 1/R² (point‑pixel). Validate.
   - Subagent (conditional): architect-review — If triage indicates geometry/convention/pivot/omega‑formula changes, validate ADR alignment and propose ADR updates. Include a 1–3 line ADR impact summary in artifacts.

3) Mandatory Parallel Trace‑Driven Validation
   - Choose an on‑peak pixel (or two) in the failing condition. Generate instrumented C trace (meters/Å per spec) and matching PyTorch trace with IDENTICAL variable names (e.g., pix0_vector, fdet/sdet/odet, Fbeam/Sbeam, R, omega, incident/diffracted, q, h,k,l, F_cell, F_latt, S-step factors).
   - Compare line‑by‑line to find the FIRST DIVERGENCE. Stop and root‑cause that divergence before changing anything else.
   - Subagent: debugger — Drive the isolate‑first‑divergence workflow and propose the minimal corrective change. Follow its reproduce/isolate steps; keep edits surgical.

4) Quantitative Checkpoints & Visuals (always attach)
   - Metrics: correlation, MSE, RMSE, max abs diff, total sums (C_sum, Py_sum) and their ratio, and optional SSIM. Count matched peaks and pixel errors if peak alignment applies.
   - Visuals: diff heatmap(s) for failing case(s). Save `c_trace.log`, `py_trace.log`, and `summary.json` with metrics.

5) Narrow & Fix (surgical)
   - Apply the smallest code change that fixes the FIRST DIVERGENCE. Prioritize geometry (pix0/F/S, axes, units, r‑factor, pivot behavior) before physics stack. Do not change tests.
   - Re‑run the specific failing case and closest neighbors (e.g., pixel sizes ±1 step). Re‑generate traces if geometry/units changed.

6) Pass/Fail Gates & Rollback
   - Final Sanity Check (Hard Gate): Re-run the mapped test(s) from the Parity Profile (or the derived mapping when the profile is absent). Success requires those authoritative tests to meet all thresholds under the required environment.
   - Pass only if ALL relevant spec gates pass:
     • AT‑specific thresholds (e.g., AT‑PARALLEL‑002: correlation ≥ 0.9999; beam center in pixels = 25.6/pixel_size ±0.1 px; peaks scale inversely with pixel size)
     • No NaNs/Infs; max abs diff within expected numerics; diff heatmaps show only low‑level residue
     • Sum ratios plausible (e.g., within 10% unless spec states otherwise)
   - If fails, revert this loop’s code edits, keep artifacts, and escalate hypotheses + traces in fix_plan. Do not touch tests/thresholds.

7) Finalize
   - Run the full test suite if targeted checks pass. Attach artifact paths.
   - Update docs/fix_plan.md at LOOP END using `prompts/update_fix_plan.md`:
     • Always append to Attempts History with corr/RMSE/MSE/max|Δ|/sum ratio and artifact paths.
     • Record FIRST DIVERGENCE (variable + file:line) and hypotheses.
     • If PASS: mark `Status: done` and quote spec thresholds satisfied.
     • If FAIL/PARTIAL: DO NOT mark done — keep item active and add concrete Next Actions; include rollback note if code changes were reverted.
     • For equivalence loops, include: the Parity Profile location (doc path + section), the exact test file(s) executed, the environment variables set (names+values or redacted if sensitive), and the exact command(s) used.
     • Parity artifact check (Hard Gate): **NO PARITY PASS = NO COMMIT.** Success demands a metrics artifact from a mapped parity path that *meets* thresholds. Absence or under‑threshold metrics means the loop failed—capture diagnostics and keep the fix_plan item `in_progress`.
   - Hard Gate (verify): Confirm the plan contains the new Attempts History entry for this loop with `Metrics:` and `Artifacts:` lines (dated paths) and consistent `Status`. If missing or inconsistent, treat the loop as failed and do not commit.
   - Subagent (post‑parity): issue — If the root‑cause class wasn’t covered or was weakly covered by Acceptance Tests/spec, propose precise spec shard edits/additions (IDs, shard, measurable expectations) without weakening thresholds; add a TODO to docs/fix_plan.md.
   - Subagent (pre‑commit): code-reviewer — Run on the changed scope to catch security/performance/config risks introduced by the fix; address high/critical findings before committing.
   - Version control hygiene:
     • PASS: `git add -A && git commit -m "<AT-ids> debug: <concise summary> (suite: pass)"`
     • FAIL/PARTIAL with rollback: stage only plan/docs/prompt changes (no reverted code), then `git add docs/fix_plan.md prompts/*.md docs/TESTING_GUIDE.md specs/*.md 2>/dev/null || true` and commit with `git commit -m "<AT-ids> debug: attempt #N failed/partial; metrics recorded; code reverted"`
     • Never commit runtime artifacts (e.g., heatmaps, binaries). Keep them under run‑local artifacts directories only.
     • After any commit (success or documented rollback), run `git push`. If the push is rejected, immediately `git pull --rebase`, resolve conflicts (capture decisions in docs/fix_plan.md and loop output), then push again before ending the loop.

Acceptance Metrics Reference (examples)
- Use the exact thresholds from specs/spec‑a‑parallel.md for each AT. Examples:
  • AT‑PARALLEL‑002: correlation ≥ 0.9999; beam center (px) = 25.6/pixel_size ± 0.1; inverse scaling of peak positions; no large structured diff.
  • General: simple_cubic ≥ 0.9995; triclinic reference ≥ 0.9995; tilted ≥ 0.9995 unless spec states otherwise.
  • Always include max abs diff and sum ratio.

Artifacts (required)
- c_trace.log, py_trace.log (aligned names/units), diff_heatmap.png, summary.json (correlation, RMSE, MSE, max_abs_diff, sums/ratio, peak stats if used), and reproduction commands.

Debugging Dtype-Sensitive Issues
- **Default precision:** float32 (production runs)
- **Debug precision:** float64 (use for numerical precision investigations and gradient checks)
- If you suspect a numerical precision issue:
  1. Rerun with explicit `dtype=torch.float64` override to check if issue persists
  2. Compare float32 vs float64 correlation and metrics
  3. Note: Gradient checks (`torch.autograd.gradcheck`) always require float64 for numerical accuracy

Guarded Anti‑Patterns (block)
- Changing tests, thresholds, or loosening tolerances.
- Submitting “fixes” without traces and metrics.
- Ignoring detector checklist (units/conventions/pivot/+0.5/r‑factor).
- Relying on correlation alone without max abs diff and visuals.

Commit & PR Notes (when loop succeeds)
- Commit message MUST mention the AT(s) fixed and FIRST DIVERGENCE. Include a short metrics line (corr/RMSE/max|Δ|/sum_ratio) and artifact paths.
- Do not include runtime artifacts in the repo; store under `plans/active/<initiative>/reports/<timestamp>/` or a temp artifacts folder per the project’s convention.

Rollback Conditions (hard requirements)
- If correlation or metrics regress for any previously passing AT‑PARALLEL within this loop, revert code changes and document in fix_plan. Do not merge.

Loop Checklist (self‑audit)
- Reproduced failing case with canonical flags
- Ran detector geometry checklist
- Produced C & Py traces; identified FIRST DIVERGENCE
- Implemented a minimal, surgical fix (no test edits)
- Reported metrics: corr/MSE/RMSE/max|Δ|/sum ratio; heatmap attached
 - Verified spec thresholds; ran full suite once if code changed (docs/prompt‑only used `--collect-only`); updated fix_plan with trace links
- No thresholds changed; no unrelated changes sneaked in
 - Updated docs/fix_plan.md at START and END per `prompts/update_fix_plan.md` (one item attempted; failure recorded without being dropped)
