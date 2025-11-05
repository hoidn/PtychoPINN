# Phase G Dense Evidence Loop — 2025-11-09T070500Z

## Objective
Deliver real-run evidence for the dense Phase C→G pipeline after digest automation landed, while tightening success messaging so MS-SSIM/MAE digests and logs are traceable from stdout.

## Scope
- Extend `run_phase_g_dense.py::main` success banner to surface both the Markdown digest path and the new digest CLI log (currently silent).
- Add pytest coverage confirming the success banner mentions the digest artifacts in order (RED→GREEN).
- Execute the dense pipeline (`--clobber`) end-to-end to generate real MS-SSIM/MAE metrics, aggregate highlights, and the new metrics digest.
- Archive CLI logs, digest output, and metrics deltas under this loop hub; summarize findings for docs/fix_plan.md.

## Tasks
1. **TDD — Success banner guard**
   - Update `tests/study/test_phase_g_dense_orchestrator.py` with a new assertion (or helper test) that fails until the banner references both `metrics_digest.md` and `cli/metrics_digest_cli.log`.
   - Capture RED evidence (`pytest ...::test_run_phase_g_dense_exec_runs_analyze_digest`).
2. **Implementation — Success banner & log surfacing**
   - Modify `plans/active/.../bin/run_phase_g_dense.py::main` to print `Metrics digest (Markdown): …` and `Metrics digest log: …` using pathlib for TYPE-PATH-001 compliance.
   - Ensure collect-only output remains unchanged aside from appended message ordering.
   - Run GREEN pytest for the updated selector(s) + smoke the highlights preview guard.
3. **Evidence Run — Dense Phase C→G pipeline**
   - Export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` before any command (CONFIG-001 guard).
   - Execute `run_phase_g_dense.py --clobber` with HUB rooted at this loop path.
   - Verify artifacts include `metrics_summary.json`, `aggregate_report.md`, `aggregate_highlights.txt`, and `metrics_digest.md`.
   - Capture CLI log tail, produce artifact inventory, and stash metrics digest preview under `analysis/`.
4. **Documentation & Ledger Sync**
   - Record MS-SSIM/MAE deltas, exit codes, and digest pointers in `summary/summary.md` (prepend Turn Summary per instructions).
   - Update `docs/fix_plan.md` Attempts History (Attempt 2025-11-09T070500Z) with status/outcomes + artifact path.
   - If tests are renamed/added, refresh `docs/TESTING_GUIDE.md` §2.5 & `docs/development/TEST_SUITE_INDEX.md` references after GREEN.

## Success Criteria
- New pytest assertion(s) pass and collect (selector returns ≥1 test).
- `run_phase_g_dense.py` stdout explicitly lists digest Markdown + CLI log paths.
- Dense pipeline command exits 0; digest and highlights regenerated under this hub.
- `summary.md` captures MS-SSIM/MAE deltas and artifact links; docs/fix_plan.md updated accordingly.

## Risks / Mitigations
- **Long runtime (2–4h)**: Kick off pipeline immediately after GREEN tests; monitor CLI log; abort early if `run_command` fails (capture blocker log).
- **Digest missing**: Retain CLI log + artifact inventory, diagnose via `cli/metrics_digest_cli.log`, document in Attempt history.
- **Env drift**: Stick to existing scripts; no new dependencies; ensure `AUTHORITATIVE_CMDS_DOC` exported.

## Artifacts
All evidence stored under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-09T070500Z/phase_g_dense_full_execution_real_run/`.
