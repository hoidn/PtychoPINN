Summary: Audit the dense Phase C→G run, harden the digest script for failures, and relaunch to capture final metrics evidence.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T190500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py::main (exit non-zero and mark digest when `n_failed > 0`)
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
- Execute: export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && python bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_cli.log
- Run: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/analysis/metrics_summary.json --highlights plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/analysis/aggregate_highlights.txt --output plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/analysis/metrics_digest.md | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/analysis/metrics_digest.log
- Capture: Update summary/summary.md with MS-SSIM/MAE deltas, pipeline exit status, and artifact references; refresh docs/fix_plan.md attempts ledger with exit codes + metrics.
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`
2. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/green/pytest_highlights_preview.log`
3. `tail -n 40 plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_cli.log` (confirm previous run stopped before relaunch)
4. `python bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_cli.log`
5. Wait for all eight commands to finish; verify `analysis/metrics_summary.json`, `analysis/aggregate_highlights.txt`, and `analysis/aggregate_report.md` exist (copy them from orchestrator outputs if needed).
6. Edit `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py` to fail-fast when `n_failed > 0`, add a "**FAILURES PRESENT**" banner to the digest, then rerun the script: `python plans/active/.../bin/analyze_dense_metrics.py --metrics ... --highlights ... --output ... | tee plans/active/.../analysis/metrics_digest.log`
7. Update `summary/summary.md` with metrics tables + command outcomes and note exit status; propagate the same data into docs/fix_plan.md Attempts History.

Pitfalls To Avoid:
- Never skip the AUTHORITATIVE_CMDS_DOC export (CONFIG-001 guard).
- Keep all outputs inside the 170500Z hub; do not spill artifacts elsewhere.
- Ensure the digest exits 0 only when `n_failed == 0`; log failures to stderr alongside the banner.
- Capture any pipeline error logs under `red/` with exit codes.
- Preserve Path operations in the script (TYPE-PATH-001 adherence).
- Confirm metrics files exist before running the digest; handle missing files as blockers.
- Avoid editing production modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Do not launch another pipeline run while one is active; check with `pgrep -fl run_phase_g_dense.py` if unsure.
- Budget 2–4 hours for pipeline completion and document partial progress if interrupted.
- Use `tee` for every long-running command so logs persist even on failure.

If Blocked:
- Save failing logs to `red/` (e.g., `red/run_phase_g_dense_cli.log`), capture exit codes, and detail the failure signature in docs/fix_plan.md + galph_memory.
- If the pipeline cannot finish, record which command failed, summarize diagnostics, and leave the hub ready for resume.
- Should the digest guard trigger, include the stderr banner and metrics snapshot in the summary, marking the attempt as blocked.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch dependency enforced; confirm torch import before long run.
- CONFIG-001 — AUTHORITATIVE_CMDS_DOC export precedes pytest/pipeline commands.
- DATA-001 — Treat any validator/schema mismatch as a hard stop.
- TYPE-PATH-001 — Continue using Path objects in scripts and artifact moves.
- OVERSAMPLING-001 — Verify metadata reports satisfy K ≥ C; flag violations in summary.

Pointers:
- docs/fix_plan.md:4 — Active ledger context.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/plan/plan.md:1 — Current execution plan.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py:1 — Script to harden.
- tests/study/test_phase_g_dense_orchestrator.py:724 — Highlights preview regression selector.
- docs/TESTING_GUIDE.md:268 — Phase G orchestrator workflow + AUTHORITATIVE guardrails.

Next Up (optional):
- After dense evidence lands, queue sparse-view parity run using the same digest tooling.

Doc Sync Plan (Conditional): Not needed — no new tests added this loop.

Mapped Tests Guardrail:
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv` must collect (>0); treat collection failures as blockers.

Hard Gate:
- Do not close the loop until the pipeline exits 0, `analysis/metrics_summary.json` + `aggregate_highlights.txt` exist, the digest exits 0 (no failure banner), and docs/fix_plan.md plus summary capture the final MS-SSIM/MAE deltas with artifact paths.
