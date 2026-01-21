Summary: Keep Phase D4 dataset-derived intensity scaling CPU-bound by updating `calculate_intensity_scale()` to consume lazy-container NumPy arrays and proving `_tensor_cache` stays empty.
Focus: DEBUG-SIM-LINES-DOSE-001 — Phase D4d lazy-container dataset-scale fix
Branch: paper
Mapped tests:
  - pytest tests/test_train_pinn.py::TestIntensityScale -v
  - pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T004455Z/

Do Now — DEBUG-SIM-LINES-DOSE-001:
- Implement: ptycho/train_pinn.py::calculate_intensity_scale — prefer `_X_np` (or a streaming reducer) to sum `|X|^2` in float64 without touching `.X`, fall back to the existing TensorFlow path only when the container lacks NumPy storage, and keep the dataset/fallback guard identical to spec.
- Implement: tests/test_train_pinn.py::TestIntensityScale::test_lazy_container_does_not_materialize — add a stub mimicking `PtychoDataContainer` (`_X_np`, `_tensor_cache`, lazy `.X`) and assert the dataset-derived path returns the correct scalar while `_tensor_cache` stays empty.
- Validate: Rerun `run_phase_c2_scenario.py --scenario gs2_ideal` + `analyze_intensity_bias.py` under the new hub and keep the CLI smoke selector green so telemetry proves the CPU-only reducer behaves the same at scale.

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export ARTIFACTS=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T004455Z; mkdir -p "$ARTIFACTS/logs" "$ARTIFACTS/gs2_ideal".
2. Update `ptycho/train_pinn.py::calculate_intensity_scale()` to detect `_X_np` (or iterable batches) and compute `sum_xy |X|^2` via NumPy float64 reduction, only dispatching to TensorFlow when `_X_np` is missing; make sure `_tensor_cache` is untouched in the NumPy path.
3. Extend `tests/test_train_pinn.py::TestIntensityScale` with a lazy-container stub and the new regression that checks `_tensor_cache` length stays zero after invoking `calculate_intensity_scale()` while still matching the spec value.
4. AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest --collect-only tests/test_train_pinn.py::TestIntensityScale -q | tee "$ARTIFACTS/logs/pytest_test_train_pinn_collect.log".
5. AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/test_train_pinn.py::TestIntensityScale -v | tee "$ARTIFACTS/logs/pytest_test_train_pinn.log".
6. AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs2_ideal --group-limit 64 --prediction-scale-source least_squares --output-dir "$ARTIFACTS/gs2_ideal" | tee "$ARTIFACTS/logs/gs2_ideal_runner.log".
7. AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py --scenario gs2_ideal=$ARTIFACTS/gs2_ideal --output-dir $ARTIFACTS | tee "$ARTIFACTS/logs/analyze_intensity_bias.log".
8. AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee "$ARTIFACTS/logs/pytest_cli_smoke.log".

Pitfalls To Avoid:
- Do not call `ptycho_data_container.X` anywhere inside the new reducer; touching `.X` will repopulate `_tensor_cache` and reintroduce GPU allocations.
- Preserve CONFIG-001/SIM-LINES-CONFIG-001 sequencing — no edits to runner update_legacy_dict calls or params.cfg mutation order.
- Keep the TensorFlow fallback path intact for containers that lack `_X_np`; the new code must not break legacy stubs/tests.
- Use float64 for intermediate reductions to match spec tolerances; do not downcast to float32 midstream.
- Never train models on CPU; gs2 rerun remains inference/telemetry only.
- Respect PLAN-LOCAL scope: do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or other protected physics modules.
- Capture every command’s stdout/stderr into $ARTIFACTS/logs per TEST-CLI-001; missing logs will block sign-off.
- Avoid introducing new dependencies or environment changes; CPU-only reducer must rely on NumPy already in the environment.
- Keep edits ASCII-only and adhere to repo formatting conventions.
- Restore any modified `params.cfg` values inside tests to prevent bleed-over.

If Blocked:
- Write the failing command + stderr to "$ARTIFACTS/logs/blocked.log", describe the issue in docs/fix_plan.md Attempts History, and ping Galph via summary + galph_memory so we can decide whether to pause Phase D4 or open a new initiative.

Findings Applied (Mandatory):
- PINN-CHUNKED-001 — lazy containers must stay on CPU; confirm `_tensor_cache` remains empty.
- SIM-LINES-CONFIG-001 — all runner/analyzer invocations must continue calling `update_legacy_dict(params.cfg, config)` before legacy modules execute.
- NORMALIZATION-001 — follow `specs/spec-ptycho-core.md §Normalization Invariants` exactly when recomputing intensity scales.
- TEST-CLI-001 — archive pytest/analyzer logs under the artifacts hub for every CLI invoked.

Pointers:
- specs/spec-ptycho-core.md:80 — Normative dataset-derived intensity scale equation and fallback conditions.
- ptycho/train_pinn.py:165 — Current dataset-derived implementation that still touches `.X`.
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:360 — D4 checklist with D4d scope and required evidence.
- docs/fix_plan.md:18 — Initiative ledger + latest Attempts History entry for Phase D4.
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T002114Z/summary.md — Evidence showing D4c outcome and remaining gap.

Next Up (optional): If D4d lands cleanly, resume D4e by instrumenting the loss graph (mae/nll composition) to chase the remaining amplitude bias.

Doc Sync Plan:
- None — selectors stay under `tests/test_train_pinn.py::TestIntensityScale` and `tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke`, both already documented.

Mapped Tests Guardrail:
- Run `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest --collect-only tests/test_train_pinn.py::TestIntensityScale -q` first; abort and report if it collects 0 tests.

Normative Math/Physics:
- Reference `specs/spec-ptycho-core.md §Normalization Invariants` for the exact dataset-derived formula (`s = sqrt(nphotons / E_batch[Σ_xy |X|^2])`) and fallback guard; do not paraphrase or invent alternate equations.
