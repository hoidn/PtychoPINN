# Reviewer Prompt (integration test gate)

You are the reviewer. Your job is to validate the long integration test and report a clear success/failure result.

Required steps:
0) read index.md and identify all plans with recent activity
1) Create a fresh output directory for this run and execute the test command below.
2) If the test fails, run the same command one more time with a new output directory.
3) If it fails again, investigate the failure reason, focusing on the last review_every_n iterations (or the fallback window below).
4) Write a success or failure report (with explanation) as an output artifact.
5) 
- Do a deep analysis of code and documentation changes since the previous review. (git diff, filter by extension: just .md, .py, .yaml). Ignore 'noisy' files such as logs, reports and artifacts unless you discover specific reasons to analyze particular ones. Ignore prompts/.
- For every Markdown file changed since the last review, validate all outbound links/anchors (including repo‑root vs docs/ paths) and report any missing targets or misrooted links.
- For every behavior/format claim touched in changed docs/specs or core modules in the diff, open the referenced spec/implementation and verify the claim still holds; log any divergence as conceptual drift.
- Evaluate the design quality of plans worked on since the last review; report findings
- Evaluate the implementation quality of changes since the last review in context of associated plans, architectural conventions, developement guidelines, and any relevant specs and normative project documentation; report findings
- Evaluate project spec and architecture self-consistency and accuracy in light of recent changes; report findings
- Evaluate plan self-consistency with other plans and existing architectural conventions; report findings
- Evaluate whether tech debt increased or decreased since the last review; report findings
- identify the most important plan with progress since the last review. review / critique that plan in its entirety (not just the changes since the last review). investigate carefully and then report your findings. IMMPORTANT while doing the review and investigation, infer the INTENTION of the plan and clarify whether the plan
  approaches it correctly IMPORTANT
- Assess whether the agent is off-track, tunnel-visioned, or stuck; if so, recommend plan or approach revisions. perpetual plan changes, documentation, and artifact collection does NOT count as not being stuck. Actual progress means: real debugging progress OR implementation progress
- Verify `docs/index.md`’s authoritative map matches actual repo structure (spec root + architecture doc locations); report mismatches as actionable.
- Include prompts/ in the doc/spec/arch cross-reference scan; flag any references to deprecated roots like `docs/spec*`  `docs/architecture/` or missing files.

<guidelines>
IMPORTANT assume evidence void of genuine implementation progress is worthless, unless you can prove otherwise IMPORTANT 
</guidelines>

Test command (use a fresh UTC timestamp per run):
RUN_TS=$(date -u +%Y-%m-%dT%H%M%SZ) RUN_LONG_INTEGRATION=1 INTEGRATION_OUTPUT_DIR=.artifacts/integration_manual_1000_512/${RUN_TS}/output pytest tests/test_integration_manual_1000_512.py -v

Investigation guidance (only after 2 failures):
- Read review cadence from orchestration.yaml (router.review_every_n); if missing or 0, use a fallback window of the last 3 iterations.
- Read state_file and logs_dir from orchestration.yaml when present; otherwise default to sync/state.json and logs/.
- Use the configured state_file to determine the current iteration and the expected actor.
- Focus on the last review_every_n (or fallback) iterations: review logs and summaries under the configured logs_dir for those iterations.
- Identify the most likely change in that window that could cause the failure (tests, configs, scripts, or data paths).
- Capture the failing test output and the suspected change in the report.

Output requirement (conditional):
- If the review surfaces actionable findings not already captured in existing plans/roadmap, write a `user_input.md`
  in the repo root in addition to creating a .artifacts report. Actionable means new issues or plan gaps not reflected in
  `docs/fix_plan.md` or `plans/active/**`. Include:
  - A concise summary of the findings
  - Evidence pointers (files/logs/tests)
  - The plan update needed or the new plan you recommend
  - The exact next steps you want the supervisor to take
- Otherwise, create a new report directory under `.artifacts/reviewer/<timestamp>/` and write
  `.artifacts/reviewer/<timestamp>/reviewer_result.md` with:
  - Description of all issues identified
  - PASS or FAIL integration test outcome
  - The test command used
  - Key error excerpt (if failed)
  - The review_every_n (or fallback) window inspected and the files/logs referenced
  - The state_file and logs_dir values used

If the test passes on the first or second run, still write the report with a brief success explanation and the output
location (user_input.md if actionable; .artifacts ).
