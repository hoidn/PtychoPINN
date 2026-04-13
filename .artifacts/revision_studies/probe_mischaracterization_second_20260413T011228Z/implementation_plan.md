# Probe Mischaracterization Second Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development if explicitly authorized by the user, otherwise use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved second-pass probe-mischaracterization revision study with study-local child-process isolation, claim-safe evidence gates, gated paper export, and a documented text-only pivot path.

**Architecture:** Keep ownership in `scripts/studies/probe_mischaracterization_stress_test.py` and `tests/studies/test_probe_mischaracterization_stress_test.py`. The parent process builds the canonical true-probe simulation once, writes a complete run-local child input bundle with explicit normalization metadata, then launches an isolated smoke/baseline child before launching one sequential child process per reviewer-facing condition to train/infer/evaluate from the bundle in a fresh TensorFlow process. Manuscript and paper-side assets are updated only after data/artifact inspection and all claim gates pass.

**Tech Stack:** Python via PATH `python`, TensorFlow/Keras grid-lines workflow, NumPy NPZ/JSON artifacts, scikit-image metrics, matplotlib figures, pytest.

**Execution Context:** Execute in the current checkout. Do not create a worktree. Do not edit `revision_designs/probe_mischaracterization_stress_test.md`.

---

## Consumed Inputs

- `.artifacts/revision_studies/probe_mischaracterization_second_20260413T011228Z/approved_design.md`
- `.artifacts/revision_studies/probe_mischaracterization_second_20260413T011228Z/plan_review.json`
- `state/revision-study-probe-second-20260413T011228Z/revision_context.md`

Supporting context read for this plan:

- `.artifacts/revision_studies/probe_mischaracterization_second_20260413T011228Z/design_review.json`
- `docs/index.md`
- `docs/findings.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/development/INVOCATION_LOGGING_GUIDE.md`
- `docs/plans/templates/implementation_plan.md`
- `scripts/studies/probe_mischaracterization_stress_test.py`
- `tests/studies/test_probe_mischaracterization_stress_test.py`
- `scripts/studies/invocation_logging.py`
- `tests/test_grid_lines_invocation_logging.py`
- `ptycho/loader.py`
- `ptycho/workflows/grid_lines_workflow.py`
- `ptycho/data_preprocessing.py`
- `ptycho/evaluation.py`
- `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`

## Compliance Matrix

- [ ] **Approved design:** Use a study-local process-isolation design; do not promote perturbation utilities to shared APIs.
- [ ] **Seed provenance:** Do not edit `revision_designs/probe_mischaracterization_stress_test.md`.
- [ ] **Stable core:** Do not edit `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- [ ] **Shared contracts:** Do not change `specs/data_contracts.md` or the project NPZ data contract.
- [ ] **CONFIG-001:** Every legacy TensorFlow training child must call `configure_legacy_params(...)`, which runs `update_legacy_dict(params.cfg, config)`, before training/inference touches legacy modules.
- [ ] **NORMALIZATION-001:** Keep `norm_Y_I_train_container`, `norm_Y_I_test_container`, `norm_Y_I_test_stitch`, and `intensity_scale_model` separately named and checksummed. Do not write a bare `norm_Y_I` in the child bundle.
- [ ] **PYTHON-ENV-001:** Use PATH `python` in persisted commands and Python subprocesses.
- [ ] **TF-REPEATED-MODEL-OOM-001:** Use sequential per-condition subprocess isolation because in-process cleanup was insufficient for the first-pass full grid.
- [ ] **NEW-PLAN-PROBE-SECOND-001:** The probe-consumption smoke/baseline TensorFlow training path must run in a child process before condition children, so a process-isolated failure cannot be blamed on parent allocator state left by smoke training.
- [ ] **NEW-PLAN-PROBE-SECOND-002:** Child-mode CLI dispatch must happen before parent output-root preparation; child output paths must come from the child request JSON, not from parent `--output-root` parsing.
- [ ] **Invocation logging:** Study entrypoints and parent/child runs must emit deterministic `invocation.json`/`invocation.sh` or child-equivalent runtime metadata artifacts.
- [ ] **Source-provenance enforcement:** Do not add parent/child source-fingerprint checks, child Git commit matching gates, `expected_source_provenance` request fields, or paper-export blocks based on child source provenance. If code changes while a diagnostic run is active, rerun from the settled implementation before relying on reviewer-facing metrics instead of adding a source-provenance subsystem.
- [ ] **Reviewer scope:** Do not imply trainable-probe, joint-probe, or position-refinement variants.

## Compatibility Boundaries and Non-Goals

- Boundary: all process-isolation input bundles are run-local study artifacts under `.artifacts/revision_studies/probe_mischaracterization/<run_id>/`; they are not reusable dataset schemas.
- Boundary: parent may simulate canonical true-probe data once; children may only rebuild train/test `PtychoDataContainer` objects from persisted canonical data plus the per-condition assumed probe.
- Boundary: parent must not call `train_pinn_model(...)`, `train_baseline_for_smoke(...)`, or `run_probe_consumption_smoke(...)` after bundle creation; smoke/baseline TensorFlow work is a child responsibility.
- Boundary: children must run sequentially: smoke/baseline child first, then condition children. No concurrent child runs.
- Boundary: true measurement arrays and constructor inputs must stay fixed across all conditions; only the reconstruction-side assumed probe changes.
- Non-goal: no PyTorch port.
- Non-goal: no new external dependency or solver.
- Non-goal: no reusable workflow/API refactor in `ptycho/workflows/grid_lines_workflow.py`.
- Non-goal: no broad manuscript rewrite beyond the fixed-probe limitation/result text needed for this reviewer issue.
- Manuscript claim limit: if numeric evidence is claim-safe, report sensitivity/degradation only for the fixed supplied-probe setting. If the gate requires sensitivity language, do not call the method robust to probe error.
- Manuscript claim limit: if the full grid fails or evidence is incomplete, use a text-only limitation response and do not include partial diagnostic metrics as reviewer-facing evidence.

## File Map

- Modify: `scripts/studies/probe_mischaracterization_stress_test.py`
  - Add child CLI mode, child request parsing, canonical bundle serialization, manifest validation, child container reconstruction, isolated smoke/baseline child runner, sequential condition child launcher, child status harvesting, and export gating updates.
- Modify: `tests/studies/test_probe_mischaracterization_stress_test.py`
  - Add tests for bundle schema, distinct normalization fields, presence/absence markers, child reconstruction, child-mode output-root parsing, isolated smoke/baseline child execution, child launch command/runtime metadata, gate behavior, and dry-run artifacts.
- Read/reuse only unless explicitly needed by tests:
  - `scripts/studies/invocation_logging.py`
  - `ptycho/loader.py`
  - `ptycho/workflows/grid_lines_workflow.py`
  - `ptycho/evaluation.py`
- Modify only after numeric or text-only pivot decision:
  - `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`
  - `/home/ollie/Documents/ptychopinnpaper2/changelog.txt`
  - `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
- Create/generated at runtime:
  - `.artifacts/revision_studies/probe_mischaracterization/<run_id>/canonical_condition_inputs.npz`
  - `.artifacts/revision_studies/probe_mischaracterization/<run_id>/canonical_condition_inputs_manifest.json`
  - `.artifacts/revision_studies/probe_mischaracterization/<run_id>/conditions/smoke/child_request.json`
  - `.artifacts/revision_studies/probe_mischaracterization/<run_id>/conditions/smoke/child_invocation.json`
  - `.artifacts/revision_studies/probe_mischaracterization/<run_id>/conditions/smoke/child_stdout.log`
  - `.artifacts/revision_studies/probe_mischaracterization/<run_id>/conditions/smoke/child_stderr.log`
  - `.artifacts/revision_studies/probe_mischaracterization/<run_id>/conditions/<condition_id>/child_request.json`
  - `.artifacts/revision_studies/probe_mischaracterization/<run_id>/conditions/<condition_id>/child_invocation.json`
  - `.artifacts/revision_studies/probe_mischaracterization/<run_id>/conditions/<condition_id>/metrics.json`
  - `.artifacts/revision_studies/probe_mischaracterization/<run_id>/conditions/<condition_id>/recon/recon.npz`

## Tranche A - Preflight and Discovery

- [ ] A1: Confirm the target files still match the approved shape with `rg -n "def run_condition|def run_full_or_smoke|def persist_true_measurements|def train_baseline_for_smoke|def run_probe_consumption_smoke|def parse_args|def main|prepare_output_root|norm_Y_I|PtychoDataContainer" scripts/studies/probe_mischaracterization_stress_test.py`.
- [ ] A2: Confirm paper revision context before any manuscript edits by reading `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`.
- [ ] A3: Confirm prior diagnostic artifact is treated as diagnostic only by inspecting `.artifacts/revision_studies/probe_mischaracterization/full_tranched_20260413T003740Z/manifest.json` if it exists.
- [ ] A4: Confirm no duplicate full run is writing to the intended output root before launching any run: `test ! -e "$output_root"`.
- [ ] A5: Record in the implementation notes that Table 2 comparability data comes from `TABLE2_AMP_SSIM`, `TABLE2_AMP_PSNR`, and `/home/ollie/Documents/ptychopinnpaper2/data/sim_lines_4x_metrics.json`.

Verification for Tranche A:

```bash
test -f .artifacts/revision_studies/probe_mischaracterization_second_20260413T011228Z/approved_design.md
test -f state/revision-study-probe-second-20260413T011228Z/revision_context.md
test -f /home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md
git diff -- ptycho/model.py ptycho/diffsim.py ptycho/tf_helper.py
```

Stop if `git diff -- ptycho/model.py ptycho/diffsim.py ptycho/tf_helper.py` is non-empty for this task or if the intended run output root already exists and is non-empty.

## Tranche B - Child Bundle Contract Tests

- [ ] B1: In `tests/studies/test_probe_mischaracterization_stress_test.py`, add a toy container fixture with deliberately distinct values for:
  - `train_container.norm_Y_I`
  - `test_container.norm_Y_I`
  - `sim["test"]["norm_Y_I"]` as stitching/display normalization
  - `sim["intensity_scale"]` as model/physics scaling
- [ ] B2: Add a failing test for a new bundle writer, for example `write_canonical_condition_bundle(output_root, sim, true_probe, cfg)`, that asserts:
  - `canonical_condition_inputs.npz` exists.
  - `canonical_condition_inputs_manifest.json` exists.
  - NPZ keys include `X_train`, `X_test`, `Y_I_train`, `Y_I_test`, `Y_phi_train`, `Y_phi_test`, `coords_nominal_train`, `coords_nominal_test`, `coords_true_train`, `coords_true_test`, `YY_full_train`, `YY_full_test`, `YY_ground_truth_test`, `probe_true`, `norm_Y_I_train_container`, `norm_Y_I_test_container`, `norm_Y_I_test_stitch`, and `intensity_scale_model`.
  - NPZ keys do not include bare `norm_Y_I`.
- [ ] B3: Add a failing test that `canonical_condition_inputs_manifest.json` records per-field dtype, shape, checksum, source split, required/present/absent status, constructor mapping, and normalization alias metadata.
- [ ] B4: Add failing tests for optional grouping metadata:
  - If `nn_indices`, `global_offsets`, `local_offsets`, or `coords_offsets` are present, the manifest records presence and checksums for train/test.
  - If absent, the manifest records explicit absence with a source-container reason.
- [ ] B5: Add a failing test that a bundle validator rejects a manifest or NPZ containing bare `norm_Y_I` or missing any distinct normalization field.

Verification for Tranche B:

```bash
pytest tests/studies/test_probe_mischaracterization_stress_test.py -k "bundle or normalization or absence" -v
```

Expected before implementation: new tests fail for missing bundle helpers. Expected after Tranche C: these tests pass.

## Tranche C - Implement Canonical Bundle Serialization

- [ ] C1: In `scripts/studies/probe_mischaracterization_stress_test.py`, add constants for bundle filenames:
  - `CANONICAL_BUNDLE_NPZ = "canonical_condition_inputs.npz"`
  - `CANONICAL_BUNDLE_MANIFEST = "canonical_condition_inputs_manifest.json"`
- [ ] C2: Add a helper such as `_array_record(name, value, source_split, required, absent_reason=None)` that returns dtype, shape, checksum, presence, and absence metadata without converting absent optional fields into empty arrays.
- [ ] C3: Add `build_canonical_condition_bundle_payload(sim, true_probe, cfg)` that extracts arrays from `sim["train"]["container"]` and `sim["test"]["container"]`, preferring container private NumPy arrays (`_X_np`, `_Y_I_np`, `_Y_phi_np`, `_coords_nominal_np`, `_coords_true_np`) over display copies.
- [ ] C4: Add `write_canonical_condition_bundle(output_root, sim, true_probe, cfg)` that writes the NPZ and JSON manifest.
- [ ] C5: Preserve distinct normalization fields:
  - `norm_Y_I_train_container = sim["train"]["container"].norm_Y_I`
  - `norm_Y_I_test_container = sim["test"]["container"].norm_Y_I`
  - `norm_Y_I_test_stitch = sim["test"]["norm_Y_I"]`
  - `intensity_scale_model = sim["intensity_scale"]`
- [ ] C6: Include `PtychoDataContainer(...)` constructor mapping in the manifest:
  - train `X <- X_train`, `Y_I <- Y_I_train`, `Y_phi <- Y_phi_train`, `norm_Y_I <- norm_Y_I_train_container`, `YY_full <- YY_full_train`, `coords_nominal <- coords_nominal_train`, `coords_true <- coords_true_train`, `nn_indices <- nn_indices_train`, `global_offsets <- global_offsets_train`, `local_offsets <- local_offsets_train`, `probeGuess <- condition assumed_probe`.
  - test mapping with corresponding test fields and `norm_Y_I_test_container`.
- [ ] C7: Add `validate_canonical_condition_bundle(bundle_path, manifest_path)` that checks required field presence, checksums, constructor mapping, no bare `norm_Y_I`, normalization aliases, and optional field absence reasons before any training starts.
- [ ] C8: Update `role_for_artifact()` so the bundle NPZ and manifest have clear roles in `artifact_manifest.json`.
- [ ] C9: Call `write_canonical_condition_bundle(...)` in `run_full_or_smoke(...)` immediately after `simulate_grid_data(...)` and before any condition loop or child launch.

Verification for Tranche C:

```bash
pytest tests/studies/test_probe_mischaracterization_stress_test.py -k "bundle or normalization or absence" -v
python -m py_compile scripts/studies/probe_mischaracterization_stress_test.py
```

Stop if the implementation needs changes in `ptycho/loader.py`, `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` to express the bundle contract.

## Tranche D - Child Reconstruction Contract

- [ ] D1: In `tests/studies/test_probe_mischaracterization_stress_test.py`, add a failing test for `load_condition_inputs_for_child(bundle_path, manifest_path, assumed_probe_path)` that returns train/test `PtychoDataContainer` objects with the assumed probe and all constructor arrays matching manifest checksums.
- [ ] D2: Add a failing test that the child preflight fails before training when:
  - true measurement checksums differ,
  - a normalization field is missing or collapsed,
  - assumed-probe checksums differ,
  - an optional field is silently omitted without an absence reason.
- [ ] D3: Implement `load_condition_inputs_for_child(...)` in the study script. It must call `validate_canonical_condition_bundle(...)`, load the condition assumed probe from NPZ, construct train/test `PtychoDataContainer` objects, and return a preflight record.
- [ ] D4: Update `assert_condition_preflight(...)` or add a child-specific preflight helper so the preflight record includes:
  - canonical bundle path and manifest path,
  - constructor-input checksums,
  - true-measurement checksums,
  - train/test container checksums,
  - assumed-probe checksums,
  - distinct normalization field checksums and alias metadata.
- [ ] D5: Keep `clone_container_with_probe(...)` only as a child/smoke reconstruction helper, or mark it with a short code comment explaining that the parent must not call it as part of TensorFlow training after bundle creation.

Verification for Tranche D:

```bash
pytest tests/studies/test_probe_mischaracterization_stress_test.py -k "child or preflight or constructor" -v
python -m py_compile scripts/studies/probe_mischaracterization_stress_test.py
```

Stop if a child cannot rebuild train/test containers solely from the run-local bundle plus an assumed-probe NPZ.

## Tranche E - Child CLI Mode, Isolated Smoke Runner, and Sequential Parent Launcher

- [ ] E1: In `tests/studies/test_probe_mischaracterization_stress_test.py`, add failing tests proving child mode is executable without parent `--output-root`:
  - `main(["--child-smoke-runner", "--child-request-json", "<request>"])` dispatches before `select_conditions(...)`, `prepare_output_root(...)`, and parent-run `write_invocation_artifacts(...)`.
  - `main(["--child-condition-runner", "--child-request-json", "<request>"])` dispatches before `select_conditions(...)`, `prepare_output_root(...)`, and parent-run `write_invocation_artifacts(...)`.
  - A child request whose `run_root` already exists and is non-empty is accepted because child output paths are derived from the request JSON.
- [ ] E2: Extend `parse_args(...)` in `scripts/studies/probe_mischaracterization_stress_test.py` with internal child-mode flags:
  - `--child-smoke-runner`
  - `--child-condition-runner`
  - `--child-request-json`
  After parsing, require `--output-root` only when neither child flag is set. In `main(...)`, dispatch child modes before parent-only work: `select_conditions(...)`, `prepare_output_root(...)`, parent invocation logging, dry-run handling, and `run_full_or_smoke(...)`.
- [ ] E3: Add `build_smoke_child_request(...)` that writes `conditions/smoke/child_request.json` with the canonical bundle paths, true-probe path/checksum, large-perturbed assumed-probe path/checksum, output paths for `probe_consumption_smoke.json`, `conditions/smoke/child_invocation.json`, stdout/stderr logs, grid config payload, and canonical checksum policy.
- [ ] E4: Add `run_smoke_child_from_request(request_path)` that:
  - writes child invocation/runtime metadata before training,
  - loads and validates the canonical bundle,
  - rebuilds train/test containers with the true probe for baseline training,
  - calls `configure_legacy_params(cfg, true_probe)`,
  - trains the smoke baseline model in the child process only,
  - runs the probe-consumption smoke against the large perturbed assumed probe,
  - writes root `probe_consumption_smoke.json` and child-local smoke status,
  - returns `0` only when the smoke gate completes and records a valid branch decision.
- [ ] E5: Add `build_condition_child_request(...)` that writes `conditions/<condition_id>/child_request.json` with condition metadata, bundle paths, assumed-probe path, output paths, branch decision, grid config payload, and canonical checksum policy.
- [ ] E6: Add `run_condition_child_from_request(request_path)` that:
  - writes child invocation/runtime metadata before training,
  - loads and validates the canonical bundle,
  - rebuilds train/test containers with the assumed probe,
  - calls `configure_legacy_params(cfg, assumed_probe)`,
  - trains/infer/stitches/evaluates this one condition,
  - writes `conditions/<condition_id>/metrics.json`,
  - writes optional `train_history.json` and `recon/recon.npz`,
  - returns `0` only when the condition status is `ok`.
- [ ] E7: Add a shared `launch_child_process(...)` helper that uses PATH `python`, derives child output paths from the request, and records the exact child PID. Use a `subprocess.Popen([...], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=child_env)` pattern followed by `communicate()`/`wait()` so the parent records command, cwd, PID, start/end timestamps, return code, stdout/stderr log paths, request path, and request checksum.
- [ ] E8: In `child_env`, preserve PATH interpreter behavior and set `PYTHONPATH` to `str(REPO_ROOT)` so the child imports the current checkout rather than ambient paths.
- [ ] E9: In `run_full_or_smoke(...)`, after canonical bundle creation and before condition launches, write and launch the smoke child, then harvest `probe_consumption_smoke.json` from the child output. The parent must not call `train_baseline_for_smoke(...)`, `train_pinn_model(...)`, or `run_probe_consumption_smoke(...)` itself.
- [ ] E10: If `args.smoke_only` or the smoke child records `pivot_text_only`, stop after writing parent manifest and artifact manifest. Otherwise, replace the in-process per-condition `run_condition(...)` call in the fixed-wrong-probe branch with sequential condition child launches.
- [ ] E11: Ensure parent classifies child return codes by reading the child `metrics.json`; if metrics are missing, parent writes a failed condition payload with the child return code and stderr excerpt.

Verification for Tranche E:

```bash
pytest tests/studies/test_probe_mischaracterization_stress_test.py -k "child or subprocess or request or output_root or smoke" -v
pytest tests/test_grid_lines_invocation_logging.py -v
python -m py_compile scripts/studies/probe_mischaracterization_stress_test.py
```

Stop if child launches require a hardcoded interpreter path, parent `--output-root`, parent run-root preparation, a new dependency, concurrent GPU use, or parent-process smoke/baseline TensorFlow training.

## Tranche F - Parent Gates, Manifests, and Dry/Smoke Runs

- [ ] F1: Update `base_manifest(...)` documents_read paths to the second-pass approved design and revision context paths.
- [ ] F2: Update `base_manifest(...)` documents_read paths to include `.artifacts/revision_studies/probe_mischaracterization_second_20260413T011228Z/plan_review.json` and the two plan-review finding IDs addressed by this revision.
- [ ] F3: Update `manifest["tf_memory_mitigation"]` to state `smoke_baseline_subprocess_isolation=True`, `per_condition_subprocess_isolation=True`, `child_launch_policy="smoke_then_conditions_sequential"`, `parent_trains_tensorflow_after_bundle_write=False`, and `finding="TF-REPEATED-MODEL-OOM-001"`.
- [ ] F4: Ensure `manifest.json` records smoke child and condition child return codes, request paths, child invocation paths, stdout/stderr log paths, exact PIDs, bundle paths, baseline gate, mild gate, infrastructure gate, and export policy.
- [ ] F5: Ensure dry-run mode writes condition manifests and bundle metadata without launching training. Dry-run should not claim reviewer-facing metrics.
- [ ] F6: Ensure smoke-only mode launches only the isolated smoke child, validates the probe-consumption branch, records child runtime metadata, and does not claim publishable metrics.
- [ ] F7: Ensure `artifact_manifest.json` includes canonical bundle files, smoke child request/invocation/log files, condition child request/invocation files, per-condition metrics, logs, recon files, probe images, final metrics, and figure when present.
- [ ] F8: Preserve the infrastructure gate: stop with exit code `2` if more than two non-baseline conditions fail for the same infrastructure reason after process isolation.
- [ ] F9: Preserve the baseline gate: compare rerun baseline to Table 2 tolerances unless `--adopt-rerun-baseline` is explicitly used.
- [ ] F10: Preserve the mild-perturbation gate: require `phase_curvature_scale_0p75`, `amplitude_blur_sigma_px_0p5`, and `phase_noise_sigma_rad_0p1pi_seed11`; force sensitivity language when drops exceed thresholds.
- [ ] F11: Preserve export gate: never write paper-side stress metrics or figures unless the stress figure exists and all claim gates permit export.

Verification for Tranche F:

```bash
pytest tests/studies/test_probe_mischaracterization_stress_test.py -v
python scripts/studies/probe_mischaracterization_stress_test.py \
  --output-root .artifacts/revision_studies/probe_mischaracterization/dry_run_process_isolated_YYYYMMDDTHHMMSSZ \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --dry-run
python scripts/studies/probe_mischaracterization_stress_test.py \
  --output-root .artifacts/revision_studies/probe_mischaracterization/smoke_process_isolated_YYYYMMDDTHHMMSSZ \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --smoke-only
```

Inspect after dry/smoke:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path(".artifacts/revision_studies/probe_mischaracterization/dry_run_process_isolated_YYYYMMDDTHHMMSSZ")
manifest = json.loads((root / "manifest.json").read_text())
artifact_manifest = json.loads((root / "artifact_manifest.json").read_text())
print(manifest["measurement_arrays_fixed_across_conditions"])
print(manifest["scope"])
print(manifest["tf_memory_mitigation"]["smoke_baseline_subprocess_isolation"])
print(manifest["tf_memory_mitigation"]["per_condition_subprocess_isolation"])
print(len(artifact_manifest["artifacts"]))
PY
```

Replace `YYYYMMDDTHHMMSSZ` with the actual run timestamp before executing.

## Tranche G - Full Process-Isolated Numeric Attempt

- [ ] G1: Choose a fresh run root under `.artifacts/revision_studies/probe_mischaracterization/full_process_isolated_YYYYMMDDTHHMMSSZ`.
- [ ] G2: Before launching, ensure the output root does not exist:

```bash
output_root=.artifacts/revision_studies/probe_mischaracterization/full_process_isolated_YYYYMMDDTHHMMSSZ
test ! -e "$output_root"
```

- [ ] G3: Launch the full run in tmux using the project long-run guardrail:

```bash
tmux new -s probe-mischar-process-isolated
conda activate ptycho311
output_root=.artifacts/revision_studies/probe_mischaracterization/full_process_isolated_YYYYMMDDTHHMMSSZ
test ! -e "$output_root"
python scripts/studies/probe_mischaracterization_stress_test.py \
  --output-root "$output_root" \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --N 64 \
  --gridsize 1 \
  --nimgs-train 2 \
  --nimgs-test 2 \
  --nepochs 60 \
  --batch-size 16 \
  --probe-scale-mode pad_preserve \
  --probe-smoothing-sigma 0.5 \
  & pid=$!; wait "$pid"; status=$?; test "$status" -eq 0
```

- [ ] G4: Do not relaunch to the same `--output-root` if the run is active or already wrote artifacts.
- [ ] G5: Consider the run complete only when the tracked PID exits `0` and required artifacts are fresh.

Required post-run artifacts:

- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/manifest.json`
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/artifact_manifest.json`
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/invocation.json`
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/canonical_condition_inputs.npz`
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/canonical_condition_inputs_manifest.json`
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/probe_consumption_smoke.json`
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/conditions/smoke/child_request.json`
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/conditions/smoke/child_invocation.json` or equivalent smoke-child runtime metadata
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/conditions/smoke/child_stdout.log`
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/conditions/smoke/child_stderr.log`
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/metrics.csv`
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/metrics.json`
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/figures/probe_mischaracterization_stress.png`
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/conditions/<condition_id>/metrics.json` for every requested condition
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/conditions/<condition_id>/child_request.json` for every requested condition
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/conditions/<condition_id>/child_invocation.json` or equivalent child runtime metadata for every requested condition
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/conditions/<condition_id>/recon/recon.npz` for every successful condition

Verification for Tranche G:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path(".artifacts/revision_studies/probe_mischaracterization/full_process_isolated_YYYYMMDDTHHMMSSZ")
manifest = json.loads((root / "manifest.json").read_text())
bundle_manifest = json.loads((root / "canonical_condition_inputs_manifest.json").read_text())
metrics = json.loads((root / "metrics.json").read_text())
smoke_request = root / "conditions" / "smoke" / "child_request.json"
smoke_invocation = root / "conditions" / "smoke" / "child_invocation.json"
assert smoke_request.exists()
assert smoke_invocation.exists()
assert manifest["tf_memory_mitigation"]["smoke_baseline_subprocess_isolation"] is True
assert manifest["tf_memory_mitigation"]["per_condition_subprocess_isolation"] is True
assert manifest["tf_memory_mitigation"]["parent_trains_tensorflow_after_bundle_write"] is False
assert manifest["tf_memory_mitigation"]["child_launch_policy"] == "smoke_then_conditions_sequential"
assert manifest["child_runs"]["smoke"]["return_code"] == 0
assert "pid" in manifest["child_runs"]["smoke"]
assert manifest["infrastructure_failure_gate"]["status"] != "stop_full_grid"
assert manifest["baseline_comparability_gate"]["claim_safe"] is True
assert manifest["mild_perturbation_gate"]["export_allowed"] is True
assert "norm_Y_I" not in bundle_manifest.get("fields", {})
for name in ("norm_Y_I_train_container", "norm_Y_I_test_container", "norm_Y_I_test_stitch", "intensity_scale_model"):
    assert name in bundle_manifest["fields"], name
assert len(metrics["conditions"]) == 10
assert (root / "figures" / "probe_mischaracterization_stress.png").exists()
PY
git diff -- ptycho/model.py ptycho/diffsim.py ptycho/tf_helper.py
```

Stop and pivot to text-only if the full run exits nonzero from an approved pivot gate, if the manifest records an unsafe gate, or if required artifacts are missing/stale.

## Tranche H - Paper-Side Assets and Manuscript Decision

Data/artifact inspection must happen before manuscript or claim updates.

- [ ] H1: Inspect `manifest.json`, `canonical_condition_inputs_manifest.json`, `metrics.json`, `metrics.csv`, and `artifact_manifest.json` from the full run.
- [ ] H2: Confirm every reviewer-facing condition succeeded or that the manifest records an approved narrowed condition set before launch.
- [ ] H3: Confirm the baseline policy is either `comparable_to_table2` or `adopt_rerun_baseline_no_old_numeric_comparison`.
- [ ] H4: Confirm the mild gate either permits a bounded robustness statement or requires sensitivity language.
- [ ] H5: Only after H1-H4 pass, run paper export with a fresh output root or the completed run root if the script supports export-only safely. If export requires re-running, do not overwrite the completed run root.
- [ ] H6: Expected paper-side assets after successful numeric export:
  - `/home/ollie/Documents/ptychopinnpaper2/data/probe_mischaracterization_metrics.json`
  - `/home/ollie/Documents/ptychopinnpaper2/figures/probe_mischaracterization_stress.png`
  - Optional `/home/ollie/Documents/ptychopinnpaper2/tables/probe_mischaracterization_metrics.tex`
- [ ] H7: Edit `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex` only after the numeric-export gate passes. Candidate locations:
  - Methods: state the probe is supplied/pre-estimated and held fixed.
  - Results near Table 2: add the stress-test result if claim-safe.
  - Discussion: state the fixed-probe limitation and no joint probe/position refinement.
- [ ] H8: If mild perturbations degrade strongly, use sensitivity language and avoid robustness claims.
- [ ] H9: If the study pivots text-only, do not create or reference paper-side stress metrics/figure. Add only fixed-probe limitation text and reviewer-response language explaining the bounded attempt and stop condition.
- [ ] H10: Update `/home/ollie/Documents/ptychopinnpaper2/changelog.txt` with a reviewer-facing entry that cites the chosen evidence path: either the claim-safe run root plus paper-side metrics/figure paths, or the text-only pivot artifact. Do not cite diagnostic first-pass metrics as reviewer-facing evidence.
- [ ] H11: Update `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md` under the R3 probe-mischaracterization item with the numeric result or text-only pivot resolution, the same run root/pivot artifact cited in `changelog.txt`, and the fixed-probe/no-trainable-probe scope. Leave the item unchecked until the manuscript, changelog, checklist, and PDF inspection all agree; mark it `[x]` only after that reviewer-facing update is complete.

Manual PDF/manuscript inspection:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
latexmk -pdf ptychopinn_2025.tex
```

Then inspect the compiled PDF manually for:

- fixed-probe wording in Methods and Discussion,
- no trainable-probe or joint-refinement implication,
- stress figure/table placement and caption clarity if numeric assets were added,
- Results text near Table 2 does not overstate robustness,
- no stale placeholders, TODOs, or unsupported partial metrics,
- changelog and reviewer checklist agree with the manuscript state.

Stop if the PDF cannot compile, the figure/table is missing, or the manuscript wording implies a method variant that was not studied.

## Tranche I - Final Verification and Handoff

- [ ] I1: Run focused tests:

```bash
pytest tests/studies/test_probe_mischaracterization_stress_test.py -v
pytest tests/test_grid_lines_invocation_logging.py -v
pytest tests/test_grid_lines_workflow.py -k "probe or pipeline or invocation" -v
python -m py_compile scripts/studies/probe_mischaracterization_stress_test.py
```

- [ ] I2: If production workflow behavior changed, run the integration marker per `docs/TESTING_GUIDE.md`:

```bash
pytest -v -m integration
```

- [ ] I3: Archive logs under the active run artifact root or link to them from the run manifest. Do not commit bulky artifacts.
- [ ] I4: Verify artifact freshness:

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path(".artifacts/revision_studies/probe_mischaracterization/full_process_isolated_YYYYMMDDTHHMMSSZ")
artifact_manifest = json.loads((root / "artifact_manifest.json").read_text())
missing = [item["path"] for item in artifact_manifest["artifacts"] if not Path(item["path"]).exists()]
if missing:
    raise SystemExit(f"missing artifact_manifest paths: {missing}")
print(len(artifact_manifest["artifacts"]))
PY
```

- [ ] I5: Verify protected files remain untouched:

```bash
git diff -- ptycho/model.py ptycho/diffsim.py ptycho/tf_helper.py specs/data_contracts.md revision_designs/probe_mischaracterization_stress_test.md
```

- [ ] I6: Verify final changed-file scope:

```bash
git status --short scripts/studies/probe_mischaracterization_stress_test.py \
  tests/studies/test_probe_mischaracterization_stress_test.py
git -C /home/ollie/Documents/ptychopinnpaper2 status --short \
  ptychopinn_2025.tex \
  changelog.txt \
  reviewer_revision_checklist.md \
  data/probe_mischaracterization_metrics.json \
  figures/probe_mischaracterization_stress.png
```

## Decision Gates and Pivot Criteria

Proceed to numeric manuscript updates only if all are true:

- Child-bundle preflight passes before training for every condition.
- The isolated smoke/baseline child succeeds before any condition child launches.
- The parent proves fixed true-measurement checksums across all conditions.
- The parent proves equivalent train/test constructor inputs across all conditions.
- The parent records distinct normalization checksums and alias metadata.
- The baseline condition succeeds.
- Baseline comparability is claim-safe, or the run explicitly adopts the rerun baseline and avoids old-number comparison.
- The three mild conditions succeed.
- The stress figure exists before export.
- `artifact_manifest.json` is fresh and complete.

Pivot to text-only if any occur:

- Process isolation still trips the infrastructure gate.
- The smoke/baseline TensorFlow training path cannot be moved out of the parent process without stable-core edits.
- Any child cannot rebuild containers from canonical true data without stable-core edits.
- The child bundle cannot preserve `YY_full`/grouping-offset presence and distinct normalization fields.
- Preflight cannot prove fixed true-measurement checksums and equivalent constructor inputs before training.
- The full run cannot produce a successful unperturbed baseline.
- The rerun baseline cannot be made Table 2 comparable and the project does not explicitly adopt the rerun baseline.
- A perturbation axis is a no-op for reported object metrics or changes data generation rather than only the assumed probe.
- The run would require a new dependency, solver, or broad workflow API change.

Scientifically unsafe stop conditions:

- Do not report diagnostic first-pass amplitude SSIM `0.9140266098362918` or PSNR `69.11115271659475` as reviewer-facing evidence.
- Do not launch dry/smoke/full process-isolated runs if child mode still requires parent `--output-root` or parent run-root preparation; fix Tranche E first.
- Do not export paper-side metrics without a complete claim-safe grid and stress figure.
- Do not state or imply robustness if mild perturbations exceed the sensitivity thresholds.
- Do not compare against old Table 2 numbers if the baseline gate records `rerun_baseline_not_comparable`.
- Do not present phase metrics as primary evidence; keep phase SSIM/PSNR/MSE/MAE diagnostic unless a separate approved design authorizes otherwise.
- Do not claim trainable-probe, joint-probe, or position-refinement capability.

## Completion Criteria

- [ ] Study-local process-isolated child mode is implemented and tested.
- [ ] Child-mode parser/dispatch can run from request JSON without parent `--output-root` or parent run-root preparation.
- [ ] Canonical child bundle NPZ/manifest records all required arrays, optional presence/absence markers, constructor mapping, and distinct normalization fields.
- [ ] Parent launches the smoke child and condition children sequentially with PATH `python`, records child runtime metadata/PIDs, and harvests smoke decisions plus condition metrics/return codes.
- [ ] Dry-run and smoke-only commands produce expected artifacts without reviewer-facing metric claims.
- [ ] Full run either produces claim-safe metrics/figure/export artifacts or records a text-only pivot artifact explaining the stop condition.
- [ ] Manuscript, changelog, and reviewer checklist updates match the evidence path chosen.
- [ ] Focused pytest selectors and required manual PDF inspection are complete.
- [ ] Protected stable-core and seed-design files are unchanged.
