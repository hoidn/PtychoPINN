Summary: Add forward-pass telemetry to trace where the ~6.5x amplitude shrinkage occurs between model predictions and ground truth.
Focus: DEBUG-SIM-LINES-DOSE-001 — Phase D5b forward-pass IntensityScaler tracing
Branch: paper
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T030000Z/
Do Now (DEBUG-SIM-LINES-DOSE-001 / D5b — see implementation.md §Phase D, D5b entry):
- Debug: Instrument `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py::run_inference_and_reassemble` to capture forward-pass scale diagnostics:
  1. Before calling `nbutils.reconstruct_image()`, log the external intensity_scale being used (`params.params()['intensity_scale']`).
  2. After `nbutils.reconstruct_image()` returns `obj_tensor_full`, compute and log:
     - `obj_mean = np.mean(np.abs(obj_tensor_full))` — model output amplitude mean
     - `input_mean = np.mean(container.X)` — normalized input mean
     - `amplification_ratio = obj_mean / input_mean` — forward-pass amplification
  3. Capture the IntensityScaler layer's `exp(log_scale)` value at inference time (already captured in `intensity_scaler_state`, but verify it matches the external scale).
  4. Log comparison: `external_scale` vs `model_exp_log_scale` and flag any discrepancy >1%.
  5. Persist these diagnostics to `run_metadata.json` under a new `forward_pass_diagnostics` block.
- Debug: Extend `run_metadata.json` output to include:
  ```json
  "forward_pass_diagnostics": {
    "external_intensity_scale": <float>,
    "model_exp_log_scale": <float>,
    "scale_match_pct": <float>,
    "input_mean": <float>,
    "output_mean": <float>,
    "amplification_ratio": <float>,
    "ground_truth_mean": <float>,
    "output_vs_truth_ratio": <float>
  }
  ```
- Validate via: Rerun `gs2_ideal` scenario with the new diagnostics:
  `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs2_ideal --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T030000Z/gs2_ideal --prediction-scale-source least_squares`
  Then `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T030000Z/logs/pytest_cli_smoke.log`.
How-To Map:
1. `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`.
2. Edit `bin/run_phase_c2_scenario.py::run_inference_and_reassemble` to add the diagnostic logging (print statements + dict for metadata).
3. In `main()`, after calling `run_inference_and_reassemble`, merge the forward_pass_diagnostics into the metadata dict before writing `run_metadata.json`.
4. Rerun gs2_ideal and capture the runner log under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T030000Z/logs/gs2_ideal_runner.log`.
5. Inspect the `run_metadata.json` for the new diagnostics block.
6. Run CLI smoke test and capture the log.
Pitfalls To Avoid:
- Don't touch `ptycho/model.py`, `ptycho/nbutils.py`, or other core modules (per CLAUDE.md directive #6); keep instrumentation in plan-local scripts only.
- The `obj_tensor_full` returned by `reconstruct_image` is a TensorFlow tensor — convert to NumPy with `.numpy()` before computing means.
- Use Python floats (not np.float64) in JSON to avoid serialization issues.
- Don't modify the inference logic, only add observability.
- If `_get_log_scale()` returns a tf.Variable, call `.numpy()` to get the float value.
- Keep print statements concise with `[runner][D5b]` prefix for easy filtering.
If Blocked: If IntensityScaler state cannot be retrieved (e.g., model structure changed), document the obstacle in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T030000Z/blocked.md` and update docs/fix_plan.md.
Findings Applied (Mandatory):
- PINN-CHUNKED-001 — any tensor conversions for diagnostics should be done after inference completes; do not add GPU materializations during the forward pass.
- SIM-LINES-CONFIG-001 — the external intensity_scale should match `params.cfg['intensity_scale']` which was synced via CONFIG-001 bridging earlier.
Pointers:
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:389 (D5b scope)
- ptycho/nbutils.py:186 (reconstruct_image call with external scaling)
- ptycho/model.py:257 (log_scale initialization from params.cfg)
- ptycho/model.py:524 (IntensityScaler layer in model)
- docs/fix_plan.md:439 (D5b planning entry)
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T024500Z/bias_summary.md (D5 evidence showing 6.5x gap)
Next Up (optional): Once forward-pass diagnostics are captured, analyze whether the gap is in model output scale vs ground truth scale, which would point to a training target mismatch (labels were scaled but ground truth comparison isn't).
