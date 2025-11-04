Summary: Kick off Phase G by adding the comparison job builder + CLI harness so we can generate three-way metrics for every dose/view/split.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.G1 — Comparison job orchestration
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_comparison.py::test_build_comparison_jobs_creates_all_conditions -vv; pytest tests/study/test_dose_overlap_comparison.py -k comparison -vv; pytest tests/study/test_dose_overlap_comparison.py --collect-only -k comparison -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/

Do Now:
- Implement: tests/study/test_dose_overlap_comparison.py::test_build_comparison_jobs_creates_all_conditions — add RED pytest asserting `build_comparison_jobs` raises `NotImplementedError`, create fixture scaffolding for fake Phase C/E/F artifacts, and capture failure with `pytest tests/study/test_dose_overlap_comparison.py::test_build_comparison_jobs_creates_all_conditions -vv --maxfail=1` teeing to `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/red/pytest_phase_g_red.log`.
- Implement: studies/fly64_dose_overlap/comparison.py::build_comparison_jobs — replace stub with dataclass-based job builder + `main()` CLI supporting `--dose/--view/--split/--dry-run`, ensure `update_legacy_dict` runs before loaders, emit manifest/summary JSON into artifact root, and keep deterministic job order; log GREEN proof under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/green/pytest_phase_g_green.log`.
- Validate: pytest tests/study/test_dose_overlap_comparison.py::test_build_comparison_jobs_creates_all_conditions -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/green/pytest_phase_g_target_green.log; pytest tests/study/test_dose_overlap_comparison.py -k comparison -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/green/pytest_phase_g_suite_green.log.
- Collect: pytest tests/study/test_dose_overlap_comparison.py --collect-only -k comparison -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/collect/pytest_phase_g_collect.log.
- CLI dry-run: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.comparison --phase-c-root tmp/phase_c_f2_cli --phase-e-root tmp/phase_e_training_gs2 --phase-f-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/cli --dose 1000 --view dense --split train --dry-run 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/cli/phase_g_cli_dry_run.log.
- Artifacts: Summarize RED→GREEN + CLI dry-run outcomes in plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/docs/summary.md, update `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/analysis/inventory.md` with resolved paths, and record Attempt #90 execution details + selector outcomes in docs/fix_plan.md/test_strategy.md/galph_memory.md.

Priorities & Rationale:
- docs/fix_plan.md:4 — Active focus moved to Phase G comparisons; this loop must advance that item with implementation-ready work.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/plan/plan.md:20 — G1 tasks require implementing the comparison job builder + CLI after the RED harness is in place.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/plan/plan.md:27 — G1.1 explicitly calls for `build_comparison_jobs` producing pointers to Phase C/E/F artifacts with CONFIG-001 compliance.
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:248 — Phase G section defines the planned selectors and artifact expectations that this loop must realize.
- docs/findings.md:8 — POLICY-001 mandates PyTorch availability; pty-chi comparisons must assume torch is present and avoid torch-optional fallbacks.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- Author RED test & stub: pytest tests/study/test_dose_overlap_comparison.py::test_build_comparison_jobs_creates_all_conditions -vv --maxfail=1 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/red/pytest_phase_g_red.log
- Implement builder + CLI, then rerun targeted selector with tee to `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/green/pytest_phase_g_target_green.log`
- Run suite selector: pytest tests/study/test_dose_overlap_comparison.py -k comparison -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/green/pytest_phase_g_suite_green.log
- Collect-only proof: pytest tests/study/test_dose_overlap_comparison.py --collect-only -k comparison -vv 2>&1 | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/collect/pytest_phase_g_collect.log
- CLI dry-run (dense/train) command above; ensure artifact-root subdirs created before run
- After GREEN, document inputs/outputs in plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/analysis/inventory.md and docs/summary.md; update docs/test registries per Doc Sync Plan.

Pitfalls To Avoid:
- Do not hardcode absolute repo paths; use Path operations relative to provided roots.
- Keep orchestrator pure — no writes to `params.cfg`, per CONFIG-001.
- Ensure job ordering is deterministic to prevent flaky tests.
- Treat Phase F manifests as read-only; never mutate prior evidence.
- Guard against missing artifacts by validating paths and failing fast with descriptive errors.
- Maintain device-neutral logic (CPU default); avoid requiring GPUs.
- Capture stdout/stderr via tee for all pytest and CLI runs; logs must live under artifact hub.
- Avoid dropping sparse acceptance metadata when forwarding pty-chi manifests.
- Respect `--dry-run` flag: no external scripts invoked during dry-run tests.
- No environment/package changes; rely on existing torch/tike installations.

If Blocked:
- If test fixtures cannot locate Phase F manifests, document missing path in plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/analysis/inventory.md, capture traceback in summary, and note block in docs/fix_plan.md Attempt log.
- If CLI dry-run fails before emitting manifest, keep full log + partial outputs, move artifacts into `cli/failed/`, and mark attempt blocked with error signature in summary + ledger.
- If `build_comparison_jobs` cannot compute all 18 combinations due to absent checkpoints, log which conditions missing and pivot to regenerating prerequisites per docs/TESTING_GUIDE.md §4 before continuing.

Findings Applied (Mandatory):
- POLICY-001 — Assume torch>=2.2 present; do not add torch-optional branches.
- CONFIG-001 — Invoke `update_legacy_dict` before legacy loaders; keep orchestrator side-effect free.
- DATA-001 — Validate Phase C/D NPZ contract when loading comparison inputs; no schema deviations.
- OVERSAMPLING-001 — Preserve sparse acceptance metadata (`selection_strategy`, `acceptance_rate`) in manifests and tests.

Pointers:
- docs/fix_plan.md:31
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/plan/plan.md:20
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/plan/plan.md:27
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:248
- docs/findings.md:8
- docs/TESTING_GUIDE.md:110

Doc Sync Plan: After GREEN, update docs/TESTING_GUIDE.md (Phase G section) and docs/development/TEST_SUITE_INDEX.md with new selectors + CLI commands, referencing logs stored under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/collect/` and `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/cli/`; capture `pytest tests/study/test_dose_overlap_comparison.py --collect-only -k comparison -vv` output and archive under the artifact hub before editing docs.

Next Up (optional): G2 dense/sparse real comparison runs once job builder + CLI validated.
