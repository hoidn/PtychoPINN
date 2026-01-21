Summary: Align Phase D4 normalization with `specs/spec-ptycho-core.md §Normalization Invariants` by making `normalize_data()` use the dataset-derived scale and proving grouped→normalized ratios stay symmetric.
Focus: DEBUG-SIM-LINES-DOSE-001 — Phase D4e normalize_data dataset-scale parity
Branch: paper
Mapped tests:
  - pytest tests/test_loader_normalization.py -v
  - pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T005723Z/

Do Now — DEBUG-SIM-LINES-DOSE-001:
- Implement: ptycho/raw_data.py::normalize_data and ptycho/loader.py::normalize_data — compute `s = sqrt(nphotons / E_batch[Σ_xy |X|^2])` via float64 NumPy reduction (with a guarded fallback when `batch_mean <= 1e-12`), keep all work CPU-bound, and share a helper so loader + RawData apply the same dataset-derived scale.
- Implement: tests/test_loader_normalization.py::TestNormalizeData::test_dataset_scale + related cases — add a stub container that surfaces `_X_np` and `_tensor_cache`, assert that the helper returns the expected dataset-derived value, respects the fallback, and leaves `_tensor_cache` untouched.
- Verify: Rerun `run_phase_c2_scenario.py --scenario gs2_ideal` with the new normalization, regenerate `bias_summary.*` via `analyze_intensity_bias.py`, and keep the CLI smoke selector green so artifacts in `.../2026-01-21T005723Z/` prove grouped→normalized ratios and normalization-invariant checks converge.

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md; export HUB=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T005723Z; mkdir -p "$HUB"/logs "$HUB"/gs2_ideal.
2. Update `ptycho/raw_data.py::normalize_data()` + `ptycho/loader.py::normalize_data()` to import `ptycho.params`, compute `batch_mean_sum_intensity = np.mean(np.sum(diffraction**2, axis=(1,2,3)))`, derive `dataset_scale = sqrt(nphotons / batch_mean)`, use the fallback `(N/2)` path only when `batch_mean <= 1e-12`, and persist the scalar into Markdown telemetry via the existing helpers.
3. Add `tests/test_loader_normalization.py` with a `LazyStubContainer` (mirrors `_X_np`, `_tensor_cache`, lazy `.X`) and regression tests covering dataset-derived, fallback, and TensorFlow-only paths; reset params after each test to avoid bleed-over.
4. AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest --collect-only tests/test_loader_normalization.py -q | tee "$HUB"/logs/pytest_collect_loader_norm.log.
5. AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/test_loader_normalization.py -v | tee "$HUB"/logs/pytest_loader_norm.log.
6. AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs2_ideal --group-limit 64 --prediction-scale-source least_squares --output-dir "$HUB"/gs2_ideal | tee "$HUB"/logs/gs2_ideal_runner.log.
7. AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py --scenario gs2_ideal=$HUB/gs2_ideal --output-dir $HUB | tee "$HUB"/logs/analyze_intensity_bias.log.
8. AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee "$HUB"/logs/pytest_cli_smoke.log.

Pitfalls To Avoid:
- Keep reducers CPU-only; never call `.X` or populate `_tensor_cache` while computing dataset scale (PINN-CHUNKED-001).
- Preserve CONFIG-001/SIM-LINES-CONFIG-001 sequencing — do not reorder `update_legacy_dict()` calls in the runner.
- Maintain TensorFlow fallback behavior so legacy containers without `_X_np` still work; add coverage for this path.
- Use float64 accumulators for `sum_xy |X|^2`; downcasting will reintroduce precision drift flagged in D4c.
- Do not hand-edit analyzer outputs; stage ratios must derive from the new math.
- Keep `nphotons` reads in sync with params.cfg and reset params in tests to avoid cross-test contamination.
- Store every command log under `$HUB/logs/` per TEST-CLI-001.
- Avoid touching protected modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- No environment tweaks; rely on existing NumPy/TensorFlow deps.
- Ensure new test modules remain ASCII and follow repo linting expectations.

If Blocked:
- Capture the failing command + stderr in `$HUB/logs/blocked.log`, add the blocker (with signature) to docs/fix_plan.md Attempts History, and ping Galph so we can decide whether to pause Phase D4 or open a new initiative.

Findings Applied (Mandatory):
- NORMALIZATION-001 — respect the three-system separation; dataset-derived scaling must operate symmetrically with the physics scaler.
- CONFIG-001 — keep params.cfg synchronized before invoking legacy pipelines, especially during runner/analyzer reruns.
- SIM-LINES-CONFIG-001 — continue calling `update_legacy_dict(params.cfg, config)` in the runner before any training/inference stage.
- PINN-CHUNKED-001 — lazy containers must remain CPU-bound; new reducers cannot populate `_tensor_cache`.

Pointers:
- specs/spec-ptycho-core.md:80 — Normative dataset-derived vs fallback intensity-scale definition.
- docs/DEVELOPER_GUIDE.md:157 — Explanation of the three normalization systems and why they must not mix.
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:372 — D4d/D4e checklist and acceptance criteria.
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T004455Z/summary.md — Evidence from D4d that we must not regress lazy-container safety.
- docs/fix_plan.md:320 — Latest Attempts History covering D4c/D4d context and the new D4e plan.

Next Up (optional): If normalization symmetry holds, proceed to D4f by instrumenting loss wiring (MAE/NLL composition) to chase the remaining prediction→truth gain.

Doc Sync Plan (tests added): After code passes, run `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest --collect-only tests/test_loader_normalization.py -q | tee "$HUB"/logs/pytest_collect_loader_norm.log`, then update `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` with the new selector/reference.

Mapped Tests Guardrail:
- First run `pytest --collect-only tests/test_loader_normalization.py -q`; if it collects 0 cases, stop and report before editing code further.

Normative Math/Physics:
- Reference `specs/spec-ptycho-core.md §Normalization Invariants` verbatim for `s = sqrt(nphotons / E_batch[Σ_xy |X|^2])` and the `sqrt(nphotons)/(N/2)` fallback; quote the clause instead of paraphrasing.
