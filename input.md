Summary: Implement the Phase D4c dataset-derived intensity scale fix in `train_pinn.calculate_intensity_scale()` and prove it collapses the gs2_ideal normalization gap with fresh analyzer + pytest logs.
Focus: DEBUG-SIM-LINES-DOSE-001 — Phase D4c dataset-derived intensity scale fix
Branch: paper
Mapped tests:
  - pytest tests/test_train_pinn.py::TestIntensityScale -v
  - pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T002114Z/

Do Now — DEBUG-SIM-LINES-DOSE-001:
- Implement: ptycho/train_pinn.py::calculate_intensity_scale — replace the fallback-only math with the dataset-derived equation `s = sqrt(nphotons / E_batch[Σ_xy |Ψ|²])`, summing across (1,2,3) axes for multi-channel tensors, guarding against zero/NaN means before falling back to `sqrt(nphotons)/(N/2)`.
- Implement: tests/test_train_pinn.py::TestIntensityScale::test_dataset_and_fallback_branches — introduce a regression suite using a stub container so the dataset-derived path matches the spec equation and the fallback engages when batch_mean→0.
- Validate: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v (plus rerun the plan-local gs2_ideal scenario + analyzer so intensity_stats.json now shows dataset/fallback ratio ≈1).

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md and ARTIFACTS=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T002114Z; mkdir -p "$ARTIFACTS/logs" for runner/analyzer/pytest logs.
2. Update `ptycho/train_pinn.py::calculate_intensity_scale` to cast `ptycho_data_container.X` to tf.float64, reduce_sum squares over axes (1,2,3) (handle rank-3 tensors by computing axes dynamically), compute `batch_mean = tf.reduce_mean(sum_intensity)` and `dataset_scale = tf.sqrt(p.get('nphotons') / batch_mean)` when `batch_mean > 1e-12`, else fall back to `sqrt(nphotons)/(p.get('N')/2)`; return a Python float.
3. Create `tests/test_train_pinn.py` (or extend it if it already exists) with a `TestIntensityScale` class that:
   • builds a stub container exposing `.X` backed by a tf.Tensor created from deterministic NumPy data;
   • sets `params.cfg['nphotons']`/`params.cfg['N']` via `params.set` and restores previous values with `try/finally`;
   • verifies the dataset-derived path equals `sqrt(nphotons / mean(sum(X**2)))` and a zeroed tensor triggers the fallback; keep tensors small (e.g., shape (2,4,4,1)).
4. Run `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest --collect-only tests/test_train_pinn.py::TestIntensityScale -q | tee "$ARTIFACTS/logs/pytest_test_train_pinn_collect.log"` to prove the selector exists.
5. Run `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/test_train_pinn.py::TestIntensityScale -v | tee "$ARTIFACTS/logs/pytest_test_train_pinn.log"` to exercise the new regression.
6. Rebuild gs2_ideal evidence with the fixed scale: `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs2_ideal --group-limit 64 --prediction-scale-source least_squares --output-dir "$ARTIFACTS/gs2_ideal" | tee "$ARTIFACTS/logs/gs2_ideal_runner.log"`.
7. Regenerate analyzer output: `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py --scenario gs2_ideal=$ARTIFACTS/gs2_ideal --output-dir $ARTIFACTS | tee "$ARTIFACTS/logs/analyze_intensity_bias.log"` and confirm `intensity_stats.json` reports dataset vs fallback ratio within ≈1 %.
8. Re-run the CLI guard: `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee "$ARTIFACTS/logs/pytest_cli_smoke.log"`.
9. Archive updated `bias_summary.json/.md`, `intensity_stats.*`, runner/analyzer logs, and pytest logs under $ARTIFACTS so Phase D4c evidence is auditable.

Pitfalls To Avoid:
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or other physics-stable modules; D4c is limited to `train_pinn.py` + tests.
- Preserve CONFIG-001/SIM-LINES-CONFIG-001 behavior—no changes to `update_legacy_dict` call sites or params.cfg initialization order.
- Handle multi-channel tensors generically; do not hard-code channel=1 or rely on static shapes.
- Avoid pulling large tensors back to NumPy; use TensorFlow ops for reductions and only convert the final scalar to Python float.
- Keep the fallback path intact for legacy datasets where container.X might already be normalized or zeroed and guard division-by-zero to prevent NaNs.
- Ensure the new pytest fixture restores any mutated `params.cfg` values to avoid cross-test contamination.
- Re-run the plan-local runner/analyzer with the fixed code before touching docs—Stage D exit criteria require runtime evidence.
- Capture every command output under $ARTIFACTS/logs per TEST-CLI-001.
- Use ASCII-only edits; no unicode symbols in comments or docs.
- PYTHON-ENV-001: invoke Python/pytest via PATH binaries, no repo-specific wrappers.

If Blocked:
- Write the failing command + stderr to "$ARTIFACTS/logs/blocked.log", summarize the blocker (e.g., tf OOM, analyzer missing files) in docs/fix_plan.md Attempts History, and ping Galph with whether the issue is in train_pinn, the new tests, or the runner so we can decide whether to pause D4c or adjust scope.

Findings Applied (Mandatory):
- CONFIG-001 — maintain the existing `update_legacy_dict(params.cfg, config)` workflow and avoid altering params.cfg globals outside sanctioned entry points.
- SIM-LINES-CONFIG-001 — runner/analyzer invocations must continue syncing params.cfg before legacy modules execute; keep the guard command intact.
- NORMALIZATION-001 — enforce the physics normalization contract by implementing the dataset-derived intensity scale exactly as specified and verifying symmetry via analyzer outputs.
- H-NEPOCHS-001 — epoch-count tweaks were ruled out, so all verification must focus on the normalization fix rather than retraining for longer.

Pointers:
- specs/spec-ptycho-core.md:86 — Normative normalization invariant equation that D4c must implement.
- ptycho/train_pinn.py:165 — Current fallback-only `calculate_intensity_scale` implementation to replace.
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:359 — D4 checklist detailing D4a–D4c requirements.
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py:1179 — Dataset-scale helper already used by the runner (reference for expected math/logging).
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T231745Z/bias_summary.md — Baseline evidence showing the 0.585 dataset/fallback ratio to beat.

Next Up (optional): If D4c lands cleanly, plan D4d around auditing loader normalization (`ptycho/loader.py::normalize_data`) so production workflows stop double-applying the fallback.

Doc Sync Plan: Not needed unless new pytest selectors beyond `tests/test_train_pinn.py::TestIntensityScale` are introduced; existing docs already list the CLI smoke test.

Mapped Tests Guardrail: Run `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest --collect-only tests/test_train_pinn.py::TestIntensityScale -q` before editing other modules so the new selector registers (>0); abort if collection fails.

Normative Math/Physics: See `specs/spec-ptycho-core.md §Normalization Invariants` for the exact dataset-derived intensity scale formula used in D4c.
