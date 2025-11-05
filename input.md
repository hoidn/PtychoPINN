Summary: Add a Phase C metadata guard to the dense orchestrator and prove it with TDD plus a fresh dense pipeline run.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_metadata -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T150500Z/phase_c_metadata_guard/

Do Now (hard validity contract):
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_metadata — add failing test covering missing `_metadata` on Phase C outputs and capture RED log.
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::validate_phase_c_metadata — introduce metadata guard using MetadataManager with actionable RuntimeError.
  - Implement (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — invoke the guard immediately after Phase C command succeeds (skip when --collect-only).
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_metadata -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs -vv
  - Validate (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001): AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T150500Z/phase_c_metadata_guard --dose 1000 --view dense --splits train test

How-To Map:
  1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
  2. mkdir -p plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T150500Z/phase_c_metadata_guard/{red,green,collect,cli,analysis}
  3. Author the new RED test in tests/study/test_phase_g_dense_orchestrator.py and run `pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_requires_metadata -vv` redirecting output to `red/pytest_guard_red.log` (expect RuntimeError signature).
  4. Update run_phase_g_dense.py with validate_phase_c_metadata helper + main() hook; ensure guard skips under --collect-only and uses MetadataManager.
  5. Re-run pytest selector, capturing GREEN log to `green/pytest_guard_green.log`, and collect-only proof via `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv > collect/pytest_phase_g_orchestrator_collect.log`.
  6. Execute orchestrator command (dose 1000 dense train/test) with `--hub` pointing to this loop; tee stdout/stderr to `cli/phase_g_orchestrator.log` and ensure guard prints metadata confirmation; store any blocker details under `analysis/blocker.log` if command fails.
  7. If command succeeds, archive resulting `analysis/metrics_summary.*` and note pass/fail in summary/summary.md; if failure before Phase G, document guard message and stop.
  8. Update docs/fix_plan.md Attempts + summary artifacts; append new insights to docs/findings.md only if we crystallize a metadata policy gap.

Pitfalls To Avoid:
  - Guard must not mutate datasets or delete outputs; only inspect metadata.
  - Ensure guard runs after Phase C only when not in --collect-only mode.
  - Keep RuntimeError text stable for pytest assertion (mention `_metadata`).
  - Avoid hardcoding hub paths; rely on Path resolves for TYPE-PATH-001.
  - Do not touch stable physics modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
  - Stop pipeline immediately on guard failure and log the culprit NPZ in blocker note.
  - Maintain DATA-001 shapes; no squeezing of Y beyond canonical rules.
  - Capture pytest logs and orchestrator transcript under the hub; no stray artifacts at repo root.
  - Keep orchestrator compatible with TensorFlow-only environments (no PyTorch-only calls in guard).
  - Ensure mapped selectors still collect ≥1 test after code changes.

If Blocked:
  - Record failure signature in analysis/blocker.log with offending path, keep CLI log, update summary.md, and notify supervisor via docs/fix_plan.md / galph_memory.md entry.

Findings Applied (Mandatory):
  - POLICY-001 — PyTorch remains required; guard must coexist with TF pipeline.
  - CONFIG-001 — Guard cannot bypass update_legacy_dict sequencing; pure metadata check only.
  - DATA-001 — Preserve canonical NPZ structure while inspecting metadata.
  - OVERSAMPLING-001 — Do not alter dense overlap thresholds or grouping parameters.
  - TYPE-PATH-001 — Normalize all filesystem paths inside new helper.

Pointers:
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:44 (orchestrator helpers)
  - plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:303 (main command wiring)
  - tests/study/test_phase_g_dense_orchestrator.py:1 (existing orchestrator TDD harness)
  - scripts/tools/transpose_rename_convert_tool.py:1 (metadata-preserving canonicalizer reference)
  - ptycho/metadata.py:200 (MetadataManager.add_transformation_record API)

Next Up (optional):
  - Phase F reconstruction metadata validation once dense guard is green.

Doc Sync Plan (Conditional):
  - After GREEN, run `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv > plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T150500Z/phase_c_metadata_guard/collect/pytest_phase_g_orchestrator_collect.log` and update docs/TESTING_GUIDE.md §2 plus docs/development/TEST_SUITE_INDEX.md with the new selector.
