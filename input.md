# Ralph Input — DEBUG-SIM-LINES-DOSE-001 Phase D6 (Training label telemetry)

**Summary:** Capture and persist training-label statistics so we can compare the labels the model sees with the ground-truth tensors used during inference.

**Focus:** DEBUG-SIM-LINES-DOSE-001 — D6: Training target formulation analysis

**Branch:** paper

**Mapped tests:** `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`

**Artifacts:** `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T021500Z/`

---

## Do Now
- Implement: `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py::record_training_label_stats` to emit per-label stats (Y_amp, Y_I, optional Y_phi/Y) and write them into `run_metadata.json::training_labels` plus a `label_vs_truth_analysis` block.
- Verify: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` (guards the runner); capture both run log and pytest log under the artifacts directory.
- Archive: rerun the gs1_ideal scenario with the new telemetry and store `run_metadata.json`, analyzer output, and logs inside the artifacts hub.

## How-To Map
```bash
export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
ARTIFACTS=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T021500Z
mkdir -p "$ARTIFACTS"/logs

# collect-only to prove the selector is live
pytest --collect-only tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -q \
  2>&1 | tee "$ARTIFACTS"/logs/pytest_cli_smoke_collect.log

# run gs1_ideal with new telemetry
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py \
  --scenario gs1_ideal \
  --output-dir "$ARTIFACTS"/gs1_ideal \
  --group-limit 64 --nepochs 5 --prediction-scale-source least_squares \
  2>&1 | tee "$ARTIFACTS"/logs/gs1_ideal_runner.log

# analyzer (optional sanity check)
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py \
  --scenario gs1_ideal="$ARTIFACTS"/gs1_ideal --output-dir "$ARTIFACTS" \
  2>&1 | tee "$ARTIFACTS"/logs/analyzer.log

# pytest guard
pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v \
  2>&1 | tee "$ARTIFACTS"/logs/pytest_cli_smoke.log
```

## Pitfalls To Avoid
1. Do **not** change any loss weights (CLAUDE.md constraint) — only log data.
2. Avoid touching core modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
3. Ensure label stats are computed on NumPy buffers to avoid forcing tensor materialization unnecessarily.
4. Capture telemetry right after container construction so training labels reflect pre-training values.
5. Guard against missing attributes: check `hasattr` before dereferencing Y_amp/Y_I/Y_phi.
6. Keep the artifacts directory clean (JSON, Markdown, logs) — no large NPZ copies.
7. Ensure pytest selector collects >0 tests (collect-only command above) before running.
8. Maintain device/dtype neutrality; no GPU-only hacks.
9. Record environment versions in logs if unexpected behavior appears.
10. Refrain from modifying analyzer tolerances until we have evidence.

## If Blocked
- If container lacks the expected label attributes, log which ones are missing inside the artifacts README and capture whatever stats are available.
- If `record_training_label_stats` uncovers TensorFlow tensors that cannot be converted lazily, reuse `_ensure_numpy()` from the runner helpers and document any gaps.
- Add a note in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T021500Z/README.md` plus galph_memory if telemetry cannot be captured; leave the code changes in place for the available signals.

## Findings Applied (Mandatory)
- **CONFIG-001 / SIM-LINES-CONFIG-001:** Runner already syncs `params.cfg`; keep it that way when adding new calls.
- **NORMALIZATION-001:** Reference `specs/spec-ptycho-core.md §Normalization Invariants` when interpreting label vs truth scales; the telemetry should make those comparisons explicit.

## Pointers
- `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:408-458` — D6 scope + constraints (no loss-weight edits).
- `plans/active/DEBUG-SIM-LINES-DOSE-001/plan/parity_logging_spec.md` — Telemetry fields all new logs must follow.
- `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py` — Runner instrumentation hook and helper definitions.
- `specs/spec-ptycho-core.md §Normalization Invariants` — Ground-truth definitions for label scaling.
- `docs/TESTING_GUIDE.md §2` — pytest selector expectations for CLI smoke tests.

## Next Up (optional)
- Compare the captured `training_labels` stats with inference ground-truth metrics; summarize deltas in `implementation.md` D6 notes.

