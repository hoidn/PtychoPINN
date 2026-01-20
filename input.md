# DEBUG-SIM-LINES-DOSE-001: Phase C4e Rescale Prototype

Summary:
Prototype an amplitude-rescaling hook so we can test whether a deterministic scalar (bundle `intensity_scale` or least-squares fit) removes the ~12x amplitude drop before touching shared workflows.

Focus:
Isolate the sim_lines_4x vs dose_experiments discrepancy by rescaling inference outputs in the Phase C4e pipeline.

Branch:
paper

Mapped test:
`pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`

Artifacts hub:
`plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/`

## Do Now (Implementation)

1. Add a `--prediction-scale-source` flag to:
   - `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py::main`
   - `scripts/studies/sim_lines_4x/pipeline.py::run_inference`

   Allowed values: `none`, `recorded`, `least_squares` (default: `none`).

2. Compute the scalar:
   - `recorded`: use the bundle/legacy `intensity_scale` stored under `intensity_info`.
   - `least_squares`: compute `scale = sum(pred_amp * truth_amp) / sum(pred_amp ** 2)`.
     - Use ground-truth amplitude only (display/evaluation layer).
     - Guard against division by zero; if `sum(pred_amp ** 2) == 0`, skip rescaling and warn.

3. Apply scaling:
   - Apply the scalar to the reconstruction amplitude (or the complex field before splitting), so phase stays unchanged.
   - Save both scaled and unscaled arrays (e.g., `amplitude_unscaled.npy`).

4. Persist metadata:
   - Store the scalar and mode under `run_metadata['prediction_scale']` and `stats.json`.
   - Ensure values are JSON-serializable (cast numpy scalars to float).

5. Keep CONFIG-001 hygiene: call `update_legacy_dict(params.cfg, config)` before touching legacy modules.

## Validate (Commands)

Prefix each command with the authoritative testing doc:
`AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`

```bash
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py \
  --scenario gs1_ideal \
  --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/gs1_ideal \
  --prediction-scale-source least_squares \
  --group-limit 64 \
  --nepochs 5 \
  > plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/gs1_ideal_runner.log

python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py \
  --scenario gs2_ideal \
  --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/gs2_ideal \
  --prediction-scale-source least_squares \
  --group-limit 64 \
  --nepochs 5 \
  > plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/gs2_ideal_runner.log

python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py \
  --scenario gs1_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/gs1_ideal \
  --scenario gs2_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/gs2_ideal \
  --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z \
  > plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/analyze_intensity_bias.log

pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v \
  | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/pytest_cli_smoke.log
```

## Pitfalls to Avoid

- Do not overwrite existing amplitude files without saving the unscaled version.
- Keep rescaling strictly in the display/evaluation layer; never feed scaled tensors back into training.
- Ensure warnings are emitted instead of crashing on zero-division.
- Avoid modifying stable physics modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Keep logs lightweight; write CLI output to the artifacts hub.

## Findings Applied (Mandatory)

- CONFIG-001 (docs/findings.md:14): sync legacy params before legacy module access.
- NORMALIZATION-001 (docs/findings.md:22): keep physics/statistical/display scaling separate.
- POLICY-001 (docs/findings.md:12): stay compatible with torch>=2.2 and avoid interpreter wrappers.

## References

- `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:199` (C4e checklist)
- `specs/spec-ptycho-core.md:86` (Normalization invariant)
- `docs/DATA_NORMALIZATION_GUIDE.md:9` (scaling boundaries)
- `docs/TESTING_GUIDE.md:55` (CLI smoke selector)

## If Blocked

Capture the failing command/traceback in:
`plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/blocker.log`

Then update `docs/fix_plan.md` Attempts + initiative summary, and stop after recording the block in galph_memory with `<status>blocked</status>`.

## Doc Sync Plan (Conditional)

Not needed (selectors unchanged).

## Next Up (Optional)

1. If rescaling is insufficient, plan Phase C4f to inspect loader normalization math before touching shared workflows.
2. If gs2_ideal still emits NaNs after rescaling, dump per-epoch loss tables for NaN channels and compare against gs1_ideal.
