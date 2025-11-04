Summary: Surface sparse selection metadata in Phase F CLI outputs and execute sparse/train + sparse/test LSQML runs with deterministic evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F3 — Sparse LSQML execution telemetry
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv; pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/

Do Now:
- Implement: studies/fly64_dose_overlap/reconstruction.py::main — surface sparse `selection_strategy` + acceptance metrics in manifest/summary; extend tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs so it RED-fails until metadata appears. Capture RED log under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/red/pytest_phase_f_sparse_red.log`.
- Validate: pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/green/pytest_phase_f_sparse_green.log
- Collect: pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -k ptychi -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/collect/pytest_phase_f_sparse_collect.log
- CLI sparse/train: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.reconstruction --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/real_run --dose 1000 --view sparse --split train --allow-missing-phase-d 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/cli/sparse_train.log
- CLI sparse/test: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.reconstruction --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/real_run --dose 1000 --view sparse --split test --allow-missing-phase-d 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/cli/sparse_test.log
- Artifacts: Update `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/docs/summary.md` with selection strategy metrics + run outcomes, sync `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md`, and append Attempt #87 results to docs/fix_plan.md with manifest + log references.

Priorities & Rationale:
- docs/fix_plan.md:34 — Phase F status notes sparse LSQML evidence outstanding; completing F3 closes remaining baseline gap.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md — new F3.1–F3.4 tasks require manifest metadata + sparse train/test runs.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:212-260 — Phase F section expects RED→GREEN selector proving selection_strategy surfacing and sparse CLI evidence archived.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T034500Z/phase_d_sparse_downsampling_fix/docs/summary.md — confirms greedy fallback emits sparse NPZs; CLI must now record strategy usage.
- docs/TESTING_GUIDE.md:184-208 — authoritative CLI commands & deterministic knobs for Phase F runs.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- RED setup: pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv --maxfail=1 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/red/pytest_phase_f_sparse_red.log
- Implement metadata surfacing in reconstruction.py and rerun GREEN command `pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv` 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/green/pytest_phase_f_sparse_green.log; rerun regression suite `pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv` 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/green/pytest_phase_f_sparse_suite_green.log
- Collect proof: pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -k ptychi -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/collect/pytest_phase_f_sparse_collect.log
- Sparse LSQML runs: ensure `tmp/phase_c_f2_cli` / `tmp/phase_d_f2_cli` exist (regenerate via Phase C/D scripts if missing); run train/test commands above, verify manifests in `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/real_run/reconstruction_manifest.json`, skip summary, and per-job logs.
- Post-run: summarize selection_strategy & acceptance metrics in `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/docs/summary.md`, update implementation/test_strategy docs, refresh ledger Attempt entry, and archive CLI + reconstruction logs under new hub.

Pitfalls To Avoid:
- Do not mutate params.cfg inside reconstruction helpers (CONFIG-001).
- Keep manifest JSON deterministic (sorted keys, stable ordering) so tests remain stable.
- Ensure sparse runs write outputs under the new artifact root, not tmp/ defaults.
- Avoid deleting or relocating prior dense evidence directories.
- Guard against missing `tmp/phase_*` roots—document minimal reproduction if regeneration needed.
- Maintain DATA-001 contract when inspecting NPZ metadata (read only).
- Preserve existing skip summary semantics (allow_missing-phase_d continues to report missing jobs).
- No environment/package changes; rely on existing torch/tike setup.
- Capture both stdout and stderr in CLI logs; do not truncate long outputs.
- Ensure tests still pass under `pytest -k "ptychi"` after metadata assertions.

If Blocked:
- If `tmp/phase_c_f2_cli` or `tmp/phase_d_f2_cli` missing, rerun Phase C/D generation per docs/TESTING_GUIDE.md §Phase C/D and record regeneration steps + logs in docs/summary.md; if regeneration fails, preserve error signature and mark Attempt #87 blocked in docs/fix_plan.md with hypothesis + next actions.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch remains required; CLI relies on torch>=2.2, no dependency changes.
- CONFIG-001 — Reconstruction CLI stays pure; selection metadata derived from NPZ headers only.
- DATA-001 — Sparse NPZs validated via Phase B contract; metadata reads must not mutate contents.
- OVERSAMPLING-001 — Greedy selection must respect 102.4 px threshold; document acceptance percentages in summary.

Pointers:
- docs/fix_plan.md:34
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:212
- studies/fly64_dose_overlap/reconstruction.py:200
- tests/study/test_dose_overlap_reconstruction.py:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/plan/plan.md:1
- docs/TESTING_GUIDE.md:184

Next Up (optional):
- Phase G comparisons once sparse LSQML evidence captured.

Doc Sync Plan:
- After GREEN tests + sparse runs, update docs/TESTING_GUIDE.md §Phase F and docs/development/TEST_SUITE_INDEX.md with sparse run selector + CLI commands, attach new collect-only log, and cite artifact hub in summary. Update implementation.md Phase F section with F3 completion details.
