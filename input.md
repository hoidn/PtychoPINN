Summary: Run the dense Phase C→G pipeline after the validator fix and capture a metrics digest for highlights evidence.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py::main (emit metrics_digest.md from metrics_summary.json + aggregate_highlights.txt)
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv
- Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_cli.log
- Run: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/analysis/metrics_summary.json --highlights plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/analysis/aggregate_highlights.txt --output plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/analysis/metrics_digest.md | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/analysis/metrics_digest.log
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/

How-To Map:
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`
2. `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/green/pytest_highlights_preview.log`
3. `python bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run --dose 1000 --view dense --splits train test --clobber | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_cli.log`
4. After the pipeline finishes, collect phase logs (`cli/phase_c_generation.log`, etc.) and copy `metrics_summary.json`, `aggregate_report.md`, and `aggregate_highlights.txt` into the hub's `analysis/` directory.
5. Author `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py` (T2 script) with argparse arguments `--metrics`, `--highlights`, and `--output` that loads JSON/txt inputs, summarizes MS-SSIM/MAE deltas, writes Markdown to `--output`, and prints the digest to stdout.
6. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/analysis/metrics_summary.json --highlights plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/analysis/aggregate_highlights.txt --output plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/analysis/metrics_digest.md | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/analysis/metrics_digest.log`
7. Update `summary/summary.md` with key Phase G metrics, command outcomes, and artifact links; sync docs/fix_plan.md Attempts History accordingly.

Pitfalls To Avoid:
- Do not run the pipeline without exporting AUTHORITATIVE_CMDS_DOC (CONFIG-001 guard).
- Preserve Path usage in the new analysis script (TYPE-PATH-001) to avoid string/Path regressions.
- Ensure the script verifies both MS-SSIM (phase/amplitude) and MAE deltas; missing sections should fail fast.
- Monitor pipeline stages; if any command exits non-zero, stop immediately and capture the failure in `red/` logs.
- Keep artifacts inside the 170500Z hub; no outputs should land outside `plans/active/.../reports/...`.
- Avoid modifying production modules beyond the planned script; supervisor loop is evidence-focused.
- Respect long runtime: budget 2–4 hours and note partial progress if stopped early.
- Use `tee` so logs persist even if the command fails mid-run.
- Delete transient scratch files before finishing; only keep reproducible artifacts.
- Do not alter stable cores (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).

If Blocked:
- Capture failing command output under `red/` (e.g., `red/pytest_highlights_preview.log`, `red/run_phase_g_dense_cli.log`) with exit codes.
- Note the precise failure signature in docs/fix_plan.md and galph_memory; include command and env guard state.
- If pipeline cannot finish (time/resource), record completed stages and plan follow-up run, leaving hub intact for continuation.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch remains required for Phase F/G; confirm imports succeed before long run.
- CONFIG-001 — Always export AUTHORITATIVE_CMDS_DOC to satisfy orchestrator/test checks.
- DATA-001 — Validator evidence must stay GREEN; treat any schema mismatch as a blocker.
- TYPE-PATH-001 — Normalize filesystem operations via Path in scripts and when moving artifacts.
- OVERSAMPLING-001 — Ensure metadata/analysis respects K ≥ C when reviewing outputs.

Pointers:
- docs/fix_plan.md:4 — Active focus status and attempt ledger.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-08T170500Z/phase_g_dense_full_execution_real_run/plan/plan.md:1 — Current plan for this run.
- tests/study/test_phase_g_dense_orchestrator.py:724 — Highlights preview regression selector.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/report_phase_g_dense_metrics.py:1 — Reference implementation for metrics formatting helpers.
- docs/TESTING_GUIDE.md:268 — Phase G orchestrator workflow, AUTHORITATIVE_CMDS_DOC guard.

Next Up (optional):
- Sparse view dense pipeline execution once dense evidence is archived.

Doc Sync Plan (Conditional): Not needed — no new tests added this loop.

Mapped Tests Guardrail:
- `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_prints_highlights_preview -vv` must collect and pass; investigate immediately if collection fails.

Hard Gate:
- Do not close the loop until the dense Phase C→G pipeline exits 0 and `analysis/metrics_digest.md` is generated alongside updated summary/doc entries; otherwise record the block with artifacts.
