<plan_update version="1.0">
  <trigger>Resume the paused PyTorch↔Ptychodus parity initiative so Ralph can work the Phase 1 bridge/persistence gaps again.</trigger>
  <focus_id>INTEGRATE-PYTORCH-PARITY-001</focus_id>
  <documents_read>docs/index.md, docs/specs/spec-ptychopinn.md, docs/specs/spec-ptycho-core.md, docs/specs/spec-ptycho-runtime.md, docs/specs/spec-ptycho-interfaces.md, docs/specs/spec-ptycho-workflow.md, docs/specs/spec-ptycho-tracing.md, docs/specs/spec-ptycho-config-bridge.md, docs/specs/spec-ptycho-conformance.md, docs/specs/overlap_metrics.md, docs/architecture.md, docs/workflows/pytorch.md, docs/findings.md, plans/ptychodus_pytorch_integration_plan.md, plans/pytorch_integration_test_plan.md, PYTORCH_INVENTORY_SUMMARY.txt, docs/fix_plan.md</documents_read>
  <current_plan_path>plans/ptychodus_pytorch_integration_plan.md</current_plan_path>
  <proposed_changes>- Document the reactivation scope with a Phase R immediate-focus checklist (config bridge invocation + persistence shim + regression test gate).
- Record the new reports hub path and testing guard so downstream loops land evidence consistently.
- Align the Do Now with the inventory quick wins (update_legacy_dict wiring, n_groups default, targeted pytest).</proposed_changes>
  <impacts>Re-enabling parity work pulls Ralph away from export/docs tasks and reintroduces PyTorch regression risk; requires pytest parity guard and hub evidence on every loop; future attempts must honor POLICY-001 torch requirements.</impacts>
  <ledger_updates>Add a high-priority focus row to docs/fix_plan.md plus a matching input.md brief and galph_memory entry pointing at this plan + hub.</ledger_updates>
  <status>approved</status>
</plan_update>

<plan_update version="1.0">
  <trigger>Phase R quick wins (config bridge invocation, CLI guardrails, persistence shim, parity pytest) are complete; now we must route the supported backend flag through the production training/inference CLIs so PyTorch becomes reachable from the primary workflow entry points.</trigger>
  <focus_id>INTEGRATE-PYTORCH-PARITY-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/workflows/pytorch.md, docs/specs/spec-ptycho-config-bridge.md, docs/DEVELOPER_GUIDE.md, docs/fix_plan.md, plans/ptychodus_pytorch_integration_plan.md, plans/pytorch_integration_test_plan.md, plans/active/INTEGRATE-PYTORCH-001/summary.md, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/analysis/artifact_inventory.txt, scripts/training/train.py, scripts/inference/inference.py, ptycho/workflows/backend_selector.py, ptycho/workflows/components.py, tests/torch/test_backend_selection.py</documents_read>
  <current_plan_path>plans/ptychodus_pytorch_integration_plan.md</current_plan_path>
  <proposed_changes>- Record that the Phase R checklist is satisfied with evidence references.
- Add an immediate-focus subsection that targets backend_selector adoption inside `scripts/training/train.py` and `scripts/inference/inference.py`.
- Spell out implementation + testing expectations (guard TensorFlow-only persistence paths, add unit tests that spy on backend dispatch, and run the backend-selector pytest node).</proposed_changes>
  <impacts>Routing the production CLIs through backend_selector touches user-facing scripts and risks regressions in TensorFlow training/inference flows; requires new mocks/tests to prove dispatch correctness and guard double-persistence side effects.</impacts>
  <ledger_updates>Update docs/fix_plan.md Do Now to describe backend-selector integration, cite the GREEN Phase R log (`green/pytest_config_bridge.log`), and point Ralph at the new CLI work plus pytest selector.</ledger_updates>
  <status>approved</status>
</plan_update>

<plan_update version="1.0">
  <trigger>Backend-selector wiring in the training/inference CLIs landed (commit a53f897b with GREEN pytest logs), so we now need to expose the backend flag on the inference CLI and prove the new route via a real PyTorch smoke run.</trigger>
  <focus_id>INTEGRATE-PYTORCH-PARITY-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/workflows/pytorch.md, docs/fix_plan.md, galph_memory.md, input.md, plans/ptychodus_pytorch_integration_plan.md, plans/pytorch_integration_test_plan.md, plans/active/INTEGRATE-PYTORCH-001/summary.md, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/analysis/artifact_inventory.txt, scripts/training/train.py, scripts/inference/inference.py, ptycho/workflows/backend_selector.py, tests/scripts/test_training_backend_selector.py, tests/scripts/test_inference_backend_selector.py</documents_read>
  <current_plan_path>plans/ptychodus_pytorch_integration_plan.md</current_plan_path>
  <proposed_changes>- Mark the backend-selector checklist items complete with evidence references.
- Add a new immediate-focus checklist covering (a) adding `--backend` support to the inference CLI, (b) running a PyTorch CLI smoke test with the minimal fixture, and (c) logging the new pytest selector + CLI evidence under the active hub.</proposed_changes>
  <impacts>Smoke-running the production CLIs exercises Lightning training and inference, so Ralph must reserve time for the run (~5–7 minutes CPU) and capture logs. Adding the inference backend flag changes CLI UX, so help text and tests must be updated in lockstep.</impacts>
  <ledger_updates>Update docs/fix_plan.md Do Now, rewrite input.md with the smoke-test commands, and prepend the initiative summary with today’s Turn Summary so Ralph has a ready_for_implementation brief.</ledger_updates>
  <status>approved</status>
</plan_update>

<plan_update version="1.0">
  <trigger>Ralph delivered the inference CLI backend flag, reran the backend-dispatch selectors, and produced PyTorch CLI smoke logs; inference still fails because `scripts/inference/inference.py` always executes the TensorFlow-only `perform_inference` path (`'probe'` KeyError in `$HUB/cli/pytorch_cli_smoke/inference.log`).</trigger>
  <focus_id>INTEGRATE-PYTORCH-PARITY-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/workflows/pytorch.md, docs/fix_plan.md, plans/ptychodus_pytorch_integration_plan.md, plans/pytorch_integration_test_plan.md, plans/active/INTEGRATE-PYTORCH-001/summary.md, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/analysis/artifact_inventory.txt, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/cli/pytorch_cli_smoke/inference.log, scripts/inference/inference.py, ptycho/workflows/backend_selector.py, ptycho_torch/workflows/components.py, ptycho_torch/inference.py, ptycho_torch/config_factory.py, tests/scripts/test_inference_backend_selector.py</documents_read>
  <current_plan_path>plans/ptychodus_pytorch_integration_plan.md</current_plan_path>
  <proposed_changes>- Mark the PyTorch CLI smoke checklist complete and record the partial inference result.
- Add a new immediate-focus checklist that targets the missing PyTorch inference execution path: branch the CLI when backend='pytorch', reuse `ptycho_torch.inference._run_inference_and_reconstruct`, and prove it via unit tests + CLI rerun with the minimal fixture.
- Call out the need to update hub inventory/summary with the new inference logs once the path runs clean.</proposed_changes>
  <impacts>Touching `scripts/inference/inference.py` risks regressions for TensorFlow users and requires new tests to guard PyTorch routing; CLI refactor must respect POLICY-001/CONFIG-001 and capture fresh hub evidence.</impacts>
  <ledger_updates>Update docs/fix_plan.md Do Now + Attempts History with the PyTorch inference gap, refresh input.md, and prepend the initiative summary with this decision so Ralph has a clear Do Now.</ledger_updates>
  <status>approved</status>
</plan_update>

<plan_update version="1.0">
  <trigger>PyTorch inference execution path shipped (commit 12fa29dd + GREEN CLI smoke), so we must record completion and pivot the Do Now toward exposing execution-config flags on the canonical inference CLI.</trigger>
  <focus_id>INTEGRATE-PYTORCH-PARITY-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/workflows/pytorch.md, docs/fix_plan.md, plans/ptychodus_pytorch_integration_plan.md, plans/active/INTEGRATE-PYTORCH-001/summary.md, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/analysis/artifact_inventory.txt, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/summary/summary.md, input.md, galph_memory.md, scripts/inference/inference.py, tests/scripts/test_inference_backend_selector.py</documents_read>
  <current_plan_path>plans/ptychodus_pytorch_integration_plan.md</current_plan_path>
  <proposed_changes>- Mark the PyTorch inference execution-path checklist complete (with references to _run_inference_and_reconstruct, GREEN pytest log, and CLI smoke artifacts).
- Add a new immediate-focus checklist instructing Ralph to expose PyTorch execution-config flags in `scripts/inference/inference.py`, reuse `ptycho_torch.cli.shared.build_execution_config_from_args`, and document the new testing/CLI evidence expectations.</proposed_changes>
  <impacts>Adding CLI flags touches the public UX and requires new tests plus another CLI smoke run; execution-config plumbing must honor POLICY-001 / CONFIG-LOGGER-001 so downstream PyTorch workflows get validated knobs.</impacts>
  <ledger_updates>Update docs/fix_plan.md Do Now + Attempts History, rewrite input.md, and prepend the initiative summary with this Turn Summary.</ledger_updates>
  <status>approved</status>
</plan_update>

## Ptychodus ↔ PtychoPINN (PyTorch) Integration Plan

### Immediate Focus — Phase R (Bridge Reactivation, 2025-11-13)

1. **Wire the configuration bridge in runtime entry points.** In `ptycho_torch/train.py` and `ptycho_torch/inference.py`, instantiate the canonical dataclasses via `config_bridge`, call `update_legacy_dict(params.cfg, config)` before touching `RawData`/loader modules, and raise actionable errors when required overrides (e.g., `train_data_file`, `n_groups`) are missing. See <doc-ref type="spec">docs/specs/spec-ptycho-config-bridge.md</doc-ref> for the exact mapping and CONFIG-001 flow requirements.
2. **Backfill spec-mandated defaults in `ptycho_torch/config_params.py`.** Ensure `n_groups`, `test_data_file`, `gaussian_smoothing_sigma`, and related knobs required by `specs/ptychodus_api_spec.md §§5.1-5.3` exist with TensorFlow-parity defaults so the bridge can populate legacy consumers.
3. **Provide a native persistence shim.** Until the full `.h5.zip` adapter lands, teach `ptycho_torch/api/base_api.py::PtychoModel.save_pytorch()` to emit a Lightning checkpoint + manifest bundle and document how `load_*` surfaces rehydrate configs; keep the implementation small but spec-compliant (§4.6).
4. **Regression gate.** Every loop must run `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -vv` (or a stricter subset) and upload logs under the active hub `plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/`.
5. **Exit criteria for reactivation:** (a) update_legacy_dict invoked in both CLI entry points, (b) config defaults + persistence shim merged, (c) targeted pytest selector green with evidence, (d) hub `analysis/artifact_inventory.txt` + `summary/summary.md` list the code/test paths touched.

#### Phase R status (2025-11-13)
- Config bridge usage in both PyTorch CLI entry points is already handled by `create_training_payload()` / `create_inference_payload()` → `populate_legacy_params()` (see `analysis/artifact_inventory.txt`).
- PyTorch persistence shim landed in `ptycho_torch/api/base_api.py::PtychoModel.save_pytorch()` via commit `ccff2f5c` with manifest + params snapshot, and the parity pytest selector (`tests/torch/test_config_bridge.py::TestConfigBridgeParity`) is green with log under `green/pytest_config_bridge.log`.
- Remaining gap: the primary TensorFlow-oriented CLIs (`scripts/training/train.py`, `scripts/inference/inference.py`) still call TensorFlow workflows directly even when `TrainingConfig.backend='pytorch'`, so PyTorch cannot yet be invoked from the canonical entry points.

#### Next Do Now — Backend selector adoption in production CLIs

- [x] **Routing layer:** Replace direct `run_cdi_example` / `load_inference_bundle` calls in `scripts/training/train.py` and `scripts/inference/inference.py` with `ptycho.workflows.backend_selector.{run_cdi_example_with_backend, load_inference_bundle_with_backend}` so the existing `--backend` flag actually dispatches PyTorch. (commit a53f897b; see `green/pytest_training_backend_dispatch.log` / `green/pytest_inference_backend_dispatch.log`).
- [x] **Persistence guardrails:** Ensure the training script only calls `model_manager.save()` / `save_outputs()` on the TensorFlow path to avoid double-saving when PyTorch workflows already emit bundles (`save_torch_bundle`); for PyTorch runs, rely on the backend workflow’s persistence and surface manifest/log locations via stdout. (Verified via artifact inventory + training CLI diff).
- [x] **Inference parity:** When backend=`'pytorch'`, skip TensorFlow-specific visualization helpers that assume `tf.keras.Model` objects; instead record the returned amplitude/phase outputs (if any) and log a warning if PyTorch inference does not yet supply visualization artifacts. (tests/scripts/test_inference_backend_selector.py covers the dispatch path).
- [x] **Tests:** Add a lightweight unit test (e.g., under `tests/scripts/test_training_backend_selector.py`) that monkeypatches `backend_selector.run_cdi_example_with_backend` to assert it receives the CLI-produced `TrainingConfig` with `backend='pytorch'` and that TensorFlow-specific helpers are skipped. Mirror the pattern for inference by patching `load_inference_bundle_with_backend`. Update `tests/torch/test_backend_selection.py` if additional coverage is needed. (5 tests now green; logs under `green/pytest_training_backend_dispatch.log` + `green/pytest_inference_backend_dispatch.log`.)
- [x] **Validation command:** `pytest tests/scripts/test_training_backend_selector.py::TestTrainingCliBackendDispatch::test_pytorch_backend_dispatch -vv` (plus the companion inference selector once added) with logs stored in the active hub’s `green/` directory. (Executed 2025-11-13; see hub `green/` logs.)

<plan_update version="1.0">
  <trigger>Execution-config flags for the inference CLI landed with commit b218696a and new pytest/CLI evidence, so we need to pivot the plan toward parity on the training CLI.</trigger>
  <focus_id>INTEGRATE-PYTORCH-PARITY-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/workflows/pytorch.md, docs/fix_plan.md, plans/ptychodus_pytorch_integration_plan.md, plans/pytorch_integration_test_plan.md, plans/active/INTEGRATE-PYTORCH-001/summary.md, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/analysis/artifact_inventory.txt, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/green/pytest_backend_selector_cli.log, scripts/inference/inference.py, scripts/training/train.py</documents_read>
  <current_plan_path>plans/ptychodus_pytorch_integration_plan.md</current_plan_path>
  <proposed_changes>Mark the inference execution-config checklist complete (evidence captured in the hub) and author a new Do Now for surfacing PyTorch execution-config flags on scripts/training/train.py, including helper delegation, pytest coverage, CLI reruns, and hub updates.</proposed_changes>
  <impacts>Training CLI edits touch a widely used entry point; we must preserve TensorFlow defaults, keep CONFIG-001 intact, and add tests/logs proving PyTorch execution-config knobs behave as documented.</impacts>
  <ledger_updates>Update docs/fix_plan.md and input.md with the completed inference work plus the new training-execution-config Do Now.</ledger_updates>
  <status>approved</status>
</plan_update>

#### Next Do Now — Inference backend flag + PyTorch CLI smoke (2025-11-13)

- [x] **Expose backend flag on inference CLI:** Extend `scripts/inference/inference.py::parse_arguments` / `setup_inference_configuration` to accept `--backend {tensorflow,pytorch}` (default `tensorflow`), plumb the value into `InferenceConfig.backend`, and update help text/docstring references so users can opt into PyTorch while respecting POLICY-001 / CONFIG-001.
- [x] **Unit-test the new CLI surface:** Update `tests/scripts/test_inference_backend_selector.py` with a case that exercises the CLI argument parsing (e.g., instantiate args namespace with `backend='pytorch'`, call `setup_inference_configuration`, and assert `config.backend` propagates). Re-run `pytest tests/scripts/test_inference_backend_selector.py::TestInferenceCliBackendDispatch::test_pytorch_backend_dispatch -vv tests/scripts/test_training_backend_selector.py::TestTrainingCliBackendDispatch::test_pytorch_backend_dispatch -vv` and log to `$HUB/green/`.
- [x] **PyTorch training CLI smoke:** From the repo root, set `HUB="$PWD/plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation"`, create `"$HUB"/cli/pytorch_cli_smoke`, and run  
  `python scripts/training/train.py --config configs/gridsize2_minimal.yaml --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --test_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --backend pytorch --output_dir "$HUB"/cli/pytorch_cli_smoke/train_outputs --n_groups 4 --n_subsample 16 --neighbor_count 7 --batch_size 4 --nepochs 1 |& tee "$HUB"/cli/pytorch_cli_smoke/train.log`.  
  Capture the emitted `bundle_path` (should be `$HUB/cli/pytorch_cli_smoke/train_outputs/wts.h5.zip`) and note any Lightning warnings in the log.
- [x] **PyTorch inference CLI smoke:** Using the new backend flag, run  
  `python scripts/inference/inference.py --model_path "$HUB"/cli/pytorch_cli_smoke/train_outputs --test_data tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --output_dir "$HUB"/cli/pytorch_cli_smoke/inference_outputs --backend pytorch --n_images 4 --n_subsample 16 |& tee "$HUB"/cli/pytorch_cli_smoke/inference.log`,  
  ensuring it loads the PyTorch bundle, restores `params.cfg`, and writes outputs under `inference_outputs/`.
- [x] **Hub + summary updates:** Append both CLI commands/logs to `analysis/artifact_inventory.txt`, note the training/inference output dirs + bundle path in `summary.md` and `summary/summary.md`, and drop any failures under `$HUB/red/blocked_<timestamp>.md`.

Outcome: Training CLI succeeded (bundle at `cli/pytorch_cli_smoke/train_outputs/wts.h5.zip`), but the inference CLI still exits with `Error during inference: 'probe'` because `perform_inference()` expects TensorFlow models. Logs live under `$HUB/cli/pytorch_cli_smoke/{train.log,inference.log}` and the artifact inventory documents the partial status.

#### Next Do Now — PyTorch inference execution path (2025-11-13)

- [x] **Branch the inference CLI:** Update `scripts/inference/inference.py` so when `config.backend == 'pytorch'` it bypasses `perform_inference()` and instead calls a PyTorch helper (e.g., `ptycho_torch.inference._run_inference_and_reconstruct`) that consumes the Lightning module returned by `load_inference_bundle_with_backend`. Ensure the helper receives the `RawData` object already loaded by the CLI, honors `n_subsample`/`n_images`, and reuses `save_reconstruction_images` so outputs match the TensorFlow UX. Guard TensorFlow-specific code paths (e.g., probe visualization, tf.keras cleanup) so they only run for `backend='tensorflow'`.
- [x] **Execution config plumbing:** Instantiate a `PyTorchExecutionConfig` (respecting POLICY-001 / CONFIG-LOGGER-001) when running the PyTorch branch so we can control inference batch size / accelerator if future flags are added. Default to CPU for the minimal smoke test but keep the structure ready for future CLI flags.
- [x] **Extend backend-dispatch tests:** Augment `tests/scripts/test_inference_backend_selector.py` with a case that patches `ptycho_torch.inference._run_inference_and_reconstruct` to assert it is called when `backend='pytorch'`, and that the TensorFlow `perform_inference()` path is skipped. Keep existing dispatch tests green. Re-run `pytest tests/scripts/test_training_backend_selector.py::TestTrainingCliBackendDispatch::test_pytorch_backend_dispatch tests/scripts/test_inference_backend_selector.py::TestInferenceCliBackendDispatch::test_pytorch_backend_dispatch -vv | tee "$HUB"/green/pytest_backend_selector_cli.log`.
- [x] **Smoke rerun:** Reuse the minimal dataset fixture to rerun the PyTorch training and inference CLI commands (same as above) so we capture a successful inference log plus the generated amplitude/phase PNGs under `$HUB/cli/pytorch_cli_smoke/inference_outputs/`. Record stdout/stderr to `cli/pytorch_cli_smoke/{train.log,inference.log}` and update `$HUB/analysis/artifact_inventory.txt`.
- [x] **Hub + summary refresh:** Document the new code paths, pytest selector, CLI commands, and artifact outputs in `$HUB/summary.md`, `$HUB/summary/summary.md`, and `analysis/artifact_inventory.txt`; drop any new failures under `$HUB/red/blocked_<timestamp>.md`.

#### Next Do Now — Canonical inference CLI execution-config flags (2025-11-13)

- [x] **Expose PyTorch execution flags:** Add `--torch-accelerator {auto,cpu,cuda,gpu,mps,tpu}`, `--torch-num-workers INT`, and `--torch-inference-batch-size INT` arguments to `scripts/inference/inference.py` (document that they only apply when `backend='pytorch'` and cite `docs/workflows/pytorch.md §12`). Defaults should match the PyTorch CLI (`auto`, `0`, `None`).
- [x] **Delegate to shared helpers:** Build a tiny namespace (or reuse argparse child parsers) so you can call `ptycho_torch.cli.shared.build_execution_config_from_args(..., mode='inference')` and obtain a validated `PyTorchExecutionConfig`. Pass the resulting `execution_config` plus resolved device into `_run_inference_and_reconstruct`; keep the TensorFlow path identical.
- [x] **Extend backend-dispatch tests:** Add a new test (e.g., `test_pytorch_execution_config_flags`) under `tests/scripts/test_inference_backend_selector.py` that patches `_run_inference_and_reconstruct` and asserts the `execution_config` respects the CLI flags (accelerator, num_workers, inference_batch_size). Keep `test_pytorch_inference_execution_path` green. Rerun `pytest tests/scripts/test_inference_backend_selector.py::TestInferenceCliBackendDispatch::test_pytorch_inference_execution_path tests/scripts/test_inference_backend_selector.py::TestInferenceCliBackendDispatch::test_pytorch_execution_config_flags -vv | tee "$HUB"/green/pytest_backend_selector_cli.log`.
- [x] **CLI smoke rerun:** Execute the minimal PyTorch training/inference commands again but include the new flags (e.g., `--torch-accelerator cpu --torch-inference-batch-size 2 --torch-num-workers 0`) so the logs prove the knobs are honored. Archive stdout/stderr and regenerated PNGs under `$HUB/cli/pytorch_cli_smoke/`.
- [x] **Hub + summary refresh:** Update `$HUB/analysis/artifact_inventory.txt`, `$HUB/summary.md`, and `$HUB/summary/summary.md` with the new CLI flags/tests, and capture any failures under `$HUB/red/blocked_<timestamp>.md`.

<plan_update version="1.0">
  <trigger>Training CLI execution-config flags landed in commit 04a016ad with GREEN pytest evidence, but the PyTorch training smoke now fails immediately (`train_debug.log:30621-30623` AttributeError: 'PtychoPINN_Lightning' has no `loss_name`).</trigger>
  <focus_id>INTEGRATE-PYTORCH-PARITY-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md (POLICY-001, CONFIG-001, CONFIG-LOGGER-001), docs/workflows/pytorch.md §12, docs/fix_plan.md, plans/ptychodus_pytorch_integration_plan.md, plans/pytorch_integration_test_plan.md, plans/active/INTEGRATE-PYTORCH-001/summary.md, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/{analysis/artifact_inventory.txt,red/blocked_20251113T183500Z_loss_name.md}, scripts/training/train.py, ptycho/workflows/backend_selector.py, train_debug.log</documents_read>
  <current_plan_path>plans/ptychodus_pytorch_integration_plan.md</current_plan_path>
  <proposed_changes>Mark the training execution-config checklist items complete (code/tests landed), record that the CLI smoke is blocked with evidence under `red/`, and add a new Do Now focusing on mapping supervised configs to a supported PyTorch `loss_function` + regression tests + rerun of the CLI smoke.</proposed_changes>
  <impacts>Until the supervised loss mapping is fixed, PyTorch cannot be invoked from the canonical training CLI, so we cannot declare parity. The fix touches config factory/bridge code as well as Lightning monitoring defaults, so regression tests are required.</impacts>
  <ledger_updates>Update docs/fix_plan.md Attempts/Do Now, refresh input.md, and prepend the hub/initiative summaries with the AttributeError findings plus the new supervised-loss tasking.</ledger_updates>
  <status>approved</status>
</plan_update>

#### Next Do Now — Training execution-config flags on canonical CLI (2025-11-13)

- [x] **Surface PyTorch execution knobs in scripts/training/train.py:** Extend the CLI surface to accept the training-specific flags documented in `docs/workflows/pytorch.md §12` (at minimum accelerator, deterministic toggle, num-workers, accumulate-grad-batches, learning-rate, scheduler, logger backend, checkpoint knobs). Ensure help text makes it clear these options only apply when `--backend pytorch` and that POLICY-001 / CONFIG-LOGGER-001 still hold. (commit 04a016ad)
- [x] **Delegate to build_execution_config_from_args(mode='training'):** After parsing, build a validated `PyTorchExecutionConfig` (reuse `ptycho_torch.cli.shared.build_execution_config_from_args`) and pass it through the backend selector. This required threading a `torch_execution_config` parameter down to the PyTorch path via `run_cdi_example_with_backend`. TensorFlow path remains untouched. (commit 04a016ad)
- [x] **Strengthen backend-selector tests:** Added `test_pytorch_execution_config_flags` under `tests/scripts/test_training_backend_selector.py` (and kept the inference execution-config test green) to assert the CLI flags propagate exactly once into `build_execution_config_from_args` and into the PyTorch workflow. GREEN log: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/green/pytest_backend_selector_cli.log`.
- [P] **PyTorch CLI smoke rerun with training flags:** Minimal dataset command now uses the new training flags. Backend selector reaches PyTorch but execution aborts with `'PtychoPINN_Lightning' object has no attribute 'loss_name'` before Lightning completes (see `red/blocked_20251113T183500Z_loss_name.md` and `train_debug.log:30621-30623`). Need supervised loss-function fix before rerunning.
- [P] **Hub + documentation refresh:** Artifact inventory updated with the training execution-config summary and blocker reference; summaries still need to cite the AttributeError once the supervised-loss fix lands. Blocker logged under `$HUB/red/`.

<plan_update version="1.0">
  <trigger>Supervised loss mapping and regression coverage landed (commit 2f36b6c6, GREEN pytest log), but the CLI smoke now fails earlier: Lightning rejects `accumulate_grad_batches>1` when `automatic_optimization=False`, and the fly001 dataset lacks `label_amp/label_phase`, producing `KeyError: 'label_amp'` before logging.</trigger>
  <focus_id>INTEGRATE-PYTORCH-PARITY-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/workflows/pytorch.md, docs/fix_plan.md, plans/ptychodus_pytorch_integration_plan.md, plans/pytorch_integration_test_plan.md, plans/active/INTEGRATE-PYTORCH-001/summary.md, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/{analysis/artifact_inventory.txt,cli/pytorch_cli_smoke_training/train.log,cli/pytorch_cli_smoke_training/train_clean.log,red/blocked_20251113T184100Z_manual_optim_accum.md,red/blocked_20251113T184300Z_supervised_data_contract.md,summary/summary.md}, input.md, galph_memory.md, scripts/training/train.py, ptycho_torch/workflows/components.py, tests/scripts/test_training_backend_selector.py</documents_read>
  <current_plan_path>plans/ptychodus_pytorch_integration_plan.md</current_plan_path>
  <proposed_changes>Mark the supervised-loss checklist items complete, document the new blockers, and add a fresh Do Now that guards `--torch-accumulate-grad-batches`, fails fast when supervised datasets lack labels, and pivots the CLI smoke to the known-good PINN command while we source labeled data.</proposed_changes>
  <impacts>Without the guardrails, users hit opaque Lightning exceptions before our logging hooks. We need actionable errors, updated documentation, and a PINN-mode CLI rerun (plus pytest evidence) to keep the hub current while tracking the supervised-data dependency.</impacts>
  <ledger_updates>Update docs/fix_plan.md attempts/Do Now, refresh input.md and both summary files, and record the new EXEC-ACCUM-001 / DATA-SUP-001 findings in docs/findings.md.</ledger_updates>
  <status>approved</status>
</plan_update>

#### Next Do Now — Supervised loss-function mapping for PyTorch training CLI (2025-11-13)

- [x] **Map canonical supervised configs to a supported PyTorch loss:** `_train_with_lightning` now forces `loss_function='MAE'` for `model_type='supervised'`, eliminating the missing `loss_name` AttributeError (`train_debug.log:30621-30623`). Commit 2f36b6c6.
- [x] **Add regression tests:** `tests/scripts/test_training_backend_selector.py::TestTrainingCliBackendDispatch::test_supervised_mode_enforces_mae_loss` exercises the override via mocks so future refactors cannot regress the MAE mapping.
- [x] **Rerun validation:** `pytest tests/scripts/test_training_backend_selector.py::TestTrainingCliBackendDispatch::test_pytorch_execution_config_flags tests/scripts/test_inference_backend_selector.py::TestInferenceCliBackendDispatch::test_pytorch_execution_config_flags tests/scripts/test_training_backend_selector.py::TestTrainingCliBackendDispatch::test_supervised_mode_enforces_mae_loss -vv` now passes (GREEN log at `$HUB/green/pytest_backend_selector_cli.log`).
- [x] **Repeat PyTorch CLI smoke:** Training CLI (`--model_type supervised`) now progresses past Lightning instantiation but surfaces two new blockers: Lightning rejects `accumulate_grad_batches=2` for manual optimization, and the fly001 dataset lacks `label_amp/label_phase`, producing `KeyError: 'label_amp'`. Both logs captured under `cli/pytorch_cli_smoke_training/{train.log,train_clean.log}` and blockers recorded under `$HUB/red/blocked_*.md`.
- [x] **Refresh hub docs:** `analysis/artifact_inventory.txt`, `summary.md`, and `summary/summary.md` document the supervised-loss fix, GREEN pytest evidence, and the new blockers. Findings EXEC-ACCUM-001 and DATA-SUP-001 capture the durable lessons.

#### Next Do Now — Execution-config guardrails + PINN CLI rerun (2025-11-13)

- [x] **Manual-optimization guard for gradient accumulation:** `ptycho_torch/workflows/components.py::_train_with_lightning` now raises a RuntimeError when `automatic_optimization=False` and `execution_config.accum_steps>1`, and `tests/scripts/test_training_backend_selector.py::test_manual_accumulation_guard` locks the behavior. Evidence: commit `9daa00b7`, hub `green/pytest_backend_selector_cli.log`.
- [x] **Supervised data-contract detection:** `_train_with_lightning` inspects the first training batch for `label_amp` / `label_phase` and fails fast with DATA-SUP-001 guidance when labels are missing; docs/workflows/pytorch.md §12 documents the requirement. Evidence: commit `9daa00b7`, hub blockers `red/blocked_20251113T184300Z_supervised_data_contract.md`.
- [P] **CLI smoke rerun (PINN mode) + hub refresh:** PINN-mode training against `tike_outputs/fly001_final_downsampled/...` succeeded (see `cli/pytorch_cli_smoke_training/train_clean.log` + `train_outputs/wts.h5.zip`), but the PyTorch inference CLI still needs to be rerun against the new bundle so stdout/PNGs and summaries reflect the refreshed pipeline.

<plan_update version="1.0">
  <trigger>Execution-config guardrails and supervised data detection landed via commit 9daa00b7 with GREEN backend-selector tests and a clean PINN-mode training CLI log; we now need to finish the PINN smoke (inference + summaries) and move the config bridge toward spec-default parity.</trigger>
  <focus_id>INTEGRATE-PYTORCH-PARITY-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/workflows/pytorch.md, docs/specs/spec-ptycho-config-bridge.md, docs/fix_plan.md, plans/ptychodus_pytorch_integration_plan.md, plans/active/INTEGRATE-PYTORCH-001/summary.md, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/{analysis/artifact_inventory.txt,cli/pytorch_cli_smoke_training/train_clean.log,green/pytest_backend_selector_cli.log}, input.md</documents_read>
  <current_plan_path>plans/ptychodus_pytorch_integration_plan.md</current_plan_path>
  <proposed_changes>Mark the guard checklist complete with evidence, note the outstanding inference rerun, and add a new Do Now that (1) backfills spec-mandated defaults in ptycho_torch/config_params.py + config_factory/config_bridge and (2) reruns the PyTorch inference CLI plus parity tests so params.cfg no longer relies on ad-hoc overrides.</proposed_changes>
  <impacts>Tightening config defaults touches shared factories/tests and requires rerunning the parity selector; finishing the PINN smoke ensures we have end-to-end evidence before modifying config plumbing.</impacts>
  <ledger_updates>Update docs/fix_plan.md Do Now + Attempts History, rewrite input.md, prepend the hub summary with today’s Turn Summary, and log the outstanding inference rerun in artifact_inventory.txt.</ledger_updates>
  <status>approved</status>
</plan_update>

#### Next Do Now — Config default parity + inference refresh (2025-11-13)

- [x] **Backfill spec defaults in PyTorch configs:** Extended `ptycho_torch/config_params.py::{DataConfig,ModelConfig,TrainingConfig}` with the spec-critical defaults (`pad_object`, `probe_scale`, `gaussian_smoothing_sigma`, `n_groups`, `train_data_file`, `test_data_file`, `n_subsample`, `subsample_seed`, `output_dir`) so we no longer rely on ad-hoc overrides when bridging (commit dd0a5b0e).
- [x] **Update factories + bridge:** Updated `ptycho_torch/config_factory.py::{create_training_payload,create_inference_payload}` and `ptycho_torch/config_bridge.py::{to_model_config,to_training_config,to_inference_config}` to consume the richer dataclass fields and keep `populate_legacy_params()` + KEY_MAPPINGS aligned with CONFIG-001.
- [x] **Parity regression coverage:** Extended `tests/torch/test_config_bridge.py::TestConfigBridgeParity` to assert the new fields propagate into TensorFlow dataclasses/`params.cfg`, then reran the selector (47 PASSED; log at `$HUB/green/pytest_config_bridge.log`).
- [x] **Validation selector:** `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -vv | tee "$HUB"/green/pytest_config_bridge.log` captured the GREEN evidence cited above.
- [P] **PyTorch inference CLI refresh:** CPU execution (same command with `--torch-accelerator cpu`) succeeded and produced amplitude/phase PNGs, but CUDA still fails with `Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)` because bundle-loaded Lightning modules remain on CPU. Evidence in `$HUB/cli/pytorch_cli_smoke_training/{inference.log,inference_cpu.log}` plus blocker `red/blocked_2025-11-13T033117Z_device_mismatch.md`.

<plan_update version="1.0">
  <trigger>CPU inference evidence is in the hub, but CUDA runs still fail because bundle-loaded Lightning modules stay on CPU. We need to fix device placement before deepening PyTorch parity work.</trigger>
  <focus_id>INTEGRATE-PYTORCH-PARITY-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, docs/INITIATIVE_WORKFLOW_GUIDE.md, docs/DEVELOPER_GUIDE.md, docs/workflows/pytorch.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/specs/spec-ptycho-config-bridge.md, docs/fix_plan.md, plans/ptychodus_pytorch_integration_plan.md, plans/active/INTEGRATE-PYTORCH-001/summary.md, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/{summary/summary.md,analysis/artifact_inventory.txt,cli/pytorch_cli_smoke_training/{inference.log,inference_cpu.log},red/blocked_2025-11-13T033117Z_device_mismatch.md}, scripts/inference/inference.py, ptycho_torch/{inference.py,workflows/components.py,model_manager.py}, tests/scripts/test_inference_backend_selector.py, tests/torch/test_cli_inference_torch.py, input.md</documents_read>
  <current_plan_path>plans/ptychodus_pytorch_integration_plan.md</current_plan_path>
  <proposed_changes>Add DEVICE-MISMATCH-001 context, record the completed config-default work, and introduce a new Do Now that moves bundle-loaded models to the execution-config accelerator, adds regression tests, and reruns the CUDA inference CLI with refreshed hub evidence.</proposed_changes>
  <impacts>CUDA inference parity is a Phase R exit criterion; leaving this broken blocks downstream PyTorch parity tasks and the fly64 study pipelines that depend on GPU reconstructions.</impacts>
  <ledger_updates>Update docs/fix_plan.md/input.md, prepend both summaries with today’s Turn Summary, refresh the hub artifact inventory, and cite DEVICE-MISMATCH-001 in docs/findings.md.</ledger_updates>
  <status>approved</status>
</plan_update>

#### Next Do Now — Device placement + CUDA inference rerun (2025-11-13)

- [x] **Move bundle-loaded models to the requested accelerator:** After `load_inference_bundle_with_backend` returns a PyTorch module, call `model.to(device)` + `model.eval()` using the resolved execution-config accelerator in `scripts/inference/inference.py`, and plumb the same `device` into `_run_inference_and_reconstruct`. (commit 85478a67; see `cli/pytorch_cli_smoke_training/inference_cuda.log` for “PyTorch model moved to device: cuda”.)
- [x] **Guard `_run_inference_and_reconstruct` tensors:** Update `ptycho_torch/inference.py::_run_inference_and_reconstruct` so Lightning modules coming from `train_results['models']` also move to `device`, and ensure diffraction/probe/position/scale tensors are created on that device before invoking `forward_predict`. (Verified via GPU log + defense in helper.)
- [x] **Regression tests for device placement:** Add a CLI-level test (e.g., `tests/scripts/test_inference_backend_selector.py::TestInferenceCliBackendDispatch::test_pytorch_backend_moves_model_to_execution_device`) that patches `_run_inference_and_reconstruct`, injects a MagicMock model, and asserts the CLI calls `model.to('cuda')` when `--torch-accelerator cuda` is provided. Add a companion test under `tests/torch/test_cli_inference_torch.py` (or a new helper) that exercises `_run_inference_and_reconstruct` with a fake model and verifies it honors the provided device without requiring real CUDA hardware. (GREEN log `green/pytest_pytorch_inference_device.log`.)
- [x] **Targeted pytest run:** Execute the new selector(s), e.g. `pytest tests/scripts/test_inference_backend_selector.py::TestInferenceCliBackendDispatch::test_pytorch_backend_moves_model_to_execution_device tests/torch/test_cli_inference_torch.py::TestInferenceCLI::test_accelerator_flag_roundtrip -vv | tee "$HUB"/green/pytest_pytorch_inference_device.log`. (Completed 2025-11-13; log stored in hub.)
- [x] **CUDA inference CLI rerun:** Run `CUDA_VISIBLE_DEVICES="0" python scripts/inference/inference.py --model_path "$HUB"/cli/pytorch_cli_smoke_training/train_outputs/wts.h5.zip --test_data tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --backend pytorch --torch-accelerator cuda --torch-num-workers 0 --torch-inference-batch-size 2 --output_dir "$HUB"/cli/pytorch_cli_smoke_training/inference_outputs` and capture stdout to `cli/pytorch_cli_smoke_training/inference.log`, regenerate amplitude/phase PNGs, and log failures (if any) under `$HUB/red/blocked_<timestamp>.md`. (See `inference_cuda.log` + `inference_outputs_cuda/*.png`.)
- [x] **Hub + summary refresh:** Update `analysis/artifact_inventory.txt`, `summary.md`, and `summary/summary.md` with the device-placement code/tests, GREEN pytest log, CUDA inference evidence (or blocker), GPU model/driver, and references to findings POLICY-001 / CONFIG-001 / CONFIG-LOGGER-001 / EXEC-ACCUM-001 / DATA-SUP-001 / DEVICE-MISMATCH-001. (Completed via commit 17d57a87.)

<plan_update version="1.0">
  <trigger>Device-placement work is complete; the next parity gap is enforcing CUDA as the default accelerator per docs/workflows/pytorch.md §12 while still providing a safe CPU fallback.</trigger>
  <focus_id>INTEGRATE-PYTORCH-PARITY-001</focus_id>
  <documents_read>docs/index.md, docs/workflows/pytorch.md, docs/findings.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/COMMANDS_REFERENCE.md, plans/ptychodus_pytorch_integration_plan.md, plans/active/INTEGRATE-PYTORCH-001/summary.md, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/analysis/artifact_inventory.txt, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/cli/pytorch_cli_smoke_training/train_clean.log</documents_read>
  <current_plan_path>plans/ptychodus_pytorch_integration_plan.md</current_plan_path>
  <proposed_changes>- Switch the training/inference CLI defaults for `--torch-accelerator` from `auto` to `cuda` to match the GPU-first policy.
- Teach `ptycho_torch.cli.shared.resolve_accelerator()` to map `auto` to CUDA when available (or CPU with a warning) so existing scripts inherit the new baseline safely.
- Update backend-selector + CLI-shared tests and rerun the targeted pytest selector alongside new GPU smoke logs.</proposed_changes>
  <impacts>Without CUDA-by-default, policy POLICIES (docs/workflows/pytorch.md §12) are violated and CLI smoke runs silently fall back to CPU (see `train_clean.log`). Forcing CUDA requires ensuring CPU-only hosts get actionable guidance instead of cryptic Lightning errors.</impacts>
  <ledger_updates>Update docs/fix_plan.md Do Now, hub `summary.md`, and the initiative summary/input briefs so Ralph executes the CUDA-default change with new pytest + CLI evidence.</ledger_updates>
  <status>approved</status>
</plan_update>

#### Next Do Now — CUDA-by-default execution configs (2025-11-13)

- [x] **Enforce CUDA defaults in canonical CLIs:** `scripts/training/train.py` and `scripts/inference/inference.py` now default `--torch-accelerator` to `'cuda'` with explicit POLICY-001 help text (commit `420e2f14`; see `analysis/artifact_inventory.txt`).
- [x] **Upgrade accelerator resolution logic:** Confirmed `ptycho_torch/cli/shared.py::resolve_accelerator` already auto-selects CUDA and emits POLICY-001 warnings on CPU fallback; documented behavior in hub summary.
- [x] **Regression tests for defaults + auto-detection:** Backend-selector + CLI-shared suites updated to expect CUDA defaults; GREEN log captured at `reports/2025-11-13T150000Z/parity_reactivation/green/pytest_cuda_default_exec_config.log`.
- [x] **Targeted pytest run:** `pytest tests/scripts/test_training_backend_selector.py::TestTrainingCliBackendDispatch::test_pytorch_execution_config_flags tests/scripts/test_inference_backend_selector.py::TestInferenceCliBackendDispatch::test_pytorch_backend_moves_model_to_execution_device tests/torch/test_cli_shared.py::TestResolveAccelerator::test_resolve_accelerator_auto_defaults -vv | tee "$HUB"/green/pytest_cuda_default_exec_config.log` (3 PASSED in 3.65 s).
- [x] **GPU smoke rerun without explicit accelerator flags:** Replayed the PyTorch training/inference CLIs without `--torch-accelerator`; logs in `cli/pytorch_cli_smoke_training/{train_cuda_default.log,inference_cuda_default.log}` plus regenerated PNGs.
- [x] **Hub + summary refresh:** `analysis/artifact_inventory.txt`, initiative summary, and hub summaries now cite CUDA-default behavior, GPU model/driver, and resolved DEVICE-MISMATCH-001 references.

<plan_update version="1.0">
  <trigger>CUDA defaults are in place for the canonical CLIs, but callers that omit `torch_execution_config` (e.g., backend_selector from Ptychodus) still inherit the dataclass default `'cpu'`, silently violating POLICY-001.</trigger>
  <focus_id>INTEGRATE-PYTORCH-PARITY-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md (POLICY-001 / CONFIG-001 / CONFIG-LOGGER-001 / EXEC-ACCUM-001 / DATA-SUP-001), docs/workflows/pytorch.md, docs/DEVELOPER_GUIDE.md, docs/COMMANDS_REFERENCE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/fix_plan.md, plans/ptychodus_pytorch_integration_plan.md, plans/active/INTEGRATE-PYTORCH-001/summary.md, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/analysis/artifact_inventory.txt, input.md</documents_read>
  <current_plan_path>plans/ptychodus_pytorch_integration_plan.md</current_plan_path>
  <proposed_changes>Add a new immediate-focus checklist to elevate `PyTorchExecutionConfig` defaults (auto-detect CUDA, warn + fall back to CPU), ensure backend selector call sites inherit the GPU baseline even when no execution config is provided, and add regression tests for the new behavior.</proposed_changes>
  <impacts>Updating the dataclass touches `ptycho/config/config.py`, the PyTorch workflow helpers, and new pytest coverage. Without this work, library consumers continue to run PyTorch on CPU by default.</impacts>
  <ledger_updates>Revise docs/fix_plan.md, summary.md, and input.md to describe the new Do Now; capture future evidence under the existing hub.</ledger_updates>
  <status>approved</status>
</plan_update>

#### Next Do Now — PyTorchExecutionConfig GPU defaults (2025-11-13)

- [x] **Promote GPU-first defaults:** Update `ptycho/config/config.py::PyTorchExecutionConfig` so `accelerator` defaults to `'auto'` and `__post_init__` resolves `'auto'` to `'cuda'` when `torch.cuda.is_available()` (fallback to `'cpu'` with a POLICY-001 warning). Document the behavior inline. (commit `3efa2dc3` + artifact inventory)
- [x] **Honor defaults in backend selector:** Ensure `ptycho_torch/workflows/components.py` call sites that synthesize an execution config (`_run_inference_and_reconstruct`, `_train_with_lightning`, etc.) inherit the GPU-first dataclass, log the resolved accelerator, and avoid silently running on CPU when execution_config is `None`. (commit `3efa2dc3`, auto-instantiation logs now emitted in both helpers)
- [x] **Regression tests:** Add a dedicated pytest module (`tests/torch/test_execution_config_defaults.py`) that monkeypatches `torch.cuda.is_available` to assert auto→CUDA on GPU hosts, warning + auto→CPU on CPU-only hosts, and verifies `backend_selector.run_cdi_example_with_backend(..., torch_execution_config=None)` requests CUDA when available. (GREEN log `green/pytest_execution_config_defaults.log`)
- [x] **Targeted selector:** `pytest tests/torch/test_execution_config_defaults.py::TestPyTorchExecutionConfigDefaults::test_auto_prefers_cuda tests/torch/test_execution_config_defaults.py::TestPyTorchExecutionConfigDefaults::test_auto_warns_and_falls_back_to_cpu -vv | tee "$HUB"/green/pytest_execution_config_defaults.log`. (2 PASSED in 0.83s)
- [x] **Hub + summary refresh:** Captured the new pytest log + warning output, cited GPU/CPU detection results, and updated `analysis/artifact_inventory.txt`, initiative summary, and hub summaries accordingly. (reports/2025-11-13T150000Z/parity_reactivation/analysis/artifact_inventory.txt)

<plan_update version="1.0">
  <trigger>GPU-first dataclass defaults landed (commit 3efa2dc3) with targeted pytest coverage, but we still lack regression tests that prove backend_selector + CLI dispatchers inherit the GPU baseline when callers omit `torch_execution_config` and that CPU-only fallback emits the documented warning.</trigger>
  <focus_id>INTEGRATE-PYTORCH-PARITY-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md (POLICY-001 / CONFIG-001 / CONFIG-LOGGER-001 / EXEC-ACCUM-001 / DATA-SUP-001), docs/workflows/pytorch.md §§11-12, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/fix_plan.md, plans/ptychodus_pytorch_integration_plan.md, plans/pytorch_integration_test_plan.md, plans/active/INTEGRATE-PYTORCH-001/summary.md, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/{analysis/artifact_inventory.txt,summary.md}, input.md</documents_read>
  <current_plan_path>plans/ptychodus_pytorch_integration_plan.md</current_plan_path>
  <proposed_changes>Close the completed GPU-default checklist, then author a new Do Now that adds backend-selector regression tests capturing the auto-instantiated `PyTorchExecutionConfig` (GPU when CUDA available, CPU+warning otherwise), reuses the execution-config pytest selector, and refreshes hub evidence/log summaries.</proposed_changes>
  <impacts>Without dispatcher-level tests we could regress back to CPU defaults when Ptychodus omits `torch_execution_config`; adding coverage keeps POLICY-001 enforceable without relying solely on dataclass tests.</impacts>
  <ledger_updates>Update docs/fix_plan.md Do Now + status, rewrite input.md with the dispatcher-test brief, and prepend today’s Turn Summary to the initiative + hub summaries.</ledger_updates>
  <status>approved</status>
</plan_update>

#### Next Do Now — Backend selector GPU-default regression tests (2025-11-13T213500Z)

- [x] **Training dispatcher guard:** Extended `tests/torch/test_execution_config_defaults.py::test_backend_selector_inherits_gpu_first_defaults` to capture the auto-instantiated `execution_config` (mocked CUDA host) and assert it resolves to `'cuda'` (commit `83ae55af`, log: `green/pytest_execution_config_defaults.log`).
- [x] **CPU-only fallback coverage:** Added companion test forcing `torch.cuda.is_available()` → False and validating `'cpu'` fallback plus POLICY-001 warning text.
- [x] **Selector run:** Reran `pytest tests/torch/test_execution_config_defaults.py -vv | tee "$HUB"/green/pytest_execution_config_defaults.log` (8 items: 7 PASSED, 1 SKIPPED) and archived the GREEN log.
- [x] **Hub + ledger refresh:** Updated `analysis/artifact_inventory.txt`, initiative summary, and hub summaries with the dispatcher-test additions and evidence links.

<plan_update version="1.0">
  <trigger>Dispatcher-level GPU-default tests landed (commit 83ae55af + GREEN `pytest_execution_config_defaults.log`), but the canonical CLIs still lack regression coverage proving they defer to backend_selector GPU defaults when users omit `--torch-*` flags.</trigger>
  <focus_id>INTEGRATE-PYTORCH-PARITY-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md (POLICY-001 / CONFIG-001 / CONFIG-LOGGER-001), docs/workflows/pytorch.md §12, docs/fix_plan.md, plans/ptychodus_pytorch_integration_plan.md, plans/pytorch_integration_test_plan.md, plans/active/INTEGRATE-PYTORCH-001/summary.md, plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/{analysis/artifact_inventory.txt,green/pytest_execution_config_defaults.log,summary.md}, tests/torch/test_execution_config_defaults.py, tests/scripts/test_training_backend_selector.py, tests/scripts/test_inference_backend_selector.py, input.md, galph_memory.md</documents_read>
  <current_plan_path>plans/ptychodus_pytorch_integration_plan.md</current_plan_path>
  <proposed_changes>Document completion of the dispatcher tests, then add a new Do Now targeting CLI-level GPU-default enforcement: (1) log when `scripts/training/train.py` and `scripts/inference/inference.py` defer to backend_selector auto-instantiation, (2) add pytest coverage proving the CLIs pass `torch_execution_config=None` when no PyTorch flags are provided, and (3) rerun the backend-selector selectors + CLI smoke to archive the new logs under the active hub.</proposed_changes>
  <impacts>Without CLI regression tests, future refactors could silently introduce CPU-first defaults by instantiating `PyTorchExecutionConfig('cpu')` inside the scripts. Logging + tests make the GPU-baseline intent explicit to both users and CI.</impacts>
  <ledger_updates>Update docs/fix_plan.md status/Do Now, refresh initiative + hub summaries with the new evidence and next steps, and rewrite input.md so Ralph executes the CLI logging + pytest additions.</ledger_updates>
  <status>approved</status>
</plan_update>

#### Next Do Now — CLI GPU-default handoff coverage (2025-11-13T220500Z)

- [ ] **Training CLI logging + test:** Teach `scripts/training/train.py::main` to log when no `--torch-*` flags are provided and the CLI defers to backend_selector GPU defaults (reference POLICY-001). Add `TestTrainingCliBackendDispatch::test_pytorch_backend_defaults_auto_execution_config` to `tests/scripts/test_training_backend_selector.py` that patches `run_cdi_example_with_backend`, omits PyTorch flags, and asserts the CLI log + `torch_execution_config is None`.
- [ ] **Inference CLI logging + test:** Mirror the logging/test addition in `scripts/inference/inference.py`, adding `TestInferenceCliBackendDispatch::test_pytorch_backend_defaults_auto_execution_config` that verifies the CLI passes through `torch_execution_config=None` and emits the GPU-baseline log when torch flags are absent.
- [ ] **Selector run:** Execute `pytest tests/scripts/test_training_backend_selector.py::TestTrainingCliBackendDispatch::test_pytorch_backend_defaults_auto_execution_config tests/scripts/test_inference_backend_selector.py::TestInferenceCliBackendDispatch::test_pytorch_backend_defaults_auto_execution_config -vv | tee "$HUB"/green/pytest_backend_selector_cli.log` so the new coverage lands in the existing GREEN log.
- [ ] **Hub + summary refresh:** Append the new log references + CLI logging behavior to `analysis/artifact_inventory.txt`, initiative summary, and hub summaries (blockers → `$HUB/red/`). Ensure CLI docs reference the new log strings where appropriate.


### 1. Scope & Goals

- Deliver a PyTorch implementation of the PtychoPINN backend that satisfies every contract defined in `specs/ptychodus_api_spec.md`.
- Keep the existing TensorFlow path fully operational while allowing runtime backend selection from ptychodus.
- Ensure configuration, data, training, inference, and persistence semantics remain identical for both backends so that third-party tooling can operate without divergence.
- **Dual Backend Surface:** The PyTorch implementation provides both a high-level API layer (`ptycho_torch/api/base_api.py`) for orchestration and low-level module access for direct integration. The integration strategy must select between these surfaces based on maintainability and spec alignment. See <doc-ref type="arch">docs/architecture_tf.md</doc-ref> and <doc-ref type="arch">docs/architecture_torch.md</doc-ref> for backend-specific diagrams and component context.
- **Configuration Bridge as First Milestone:** Establishing dataclass-driven configuration synchronization with `ptycho.params.cfg` is the critical dependency for all downstream workflows and must be completed in Phase 1 before data pipeline or training integration.

### 2. Authoritative References

| Topic | Spec Section | Key Files |
| --- | --- | --- |
| Reconstructor lifecycle & behaviour | `specs/ptychodus_api_spec.md:127-211` | `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py`, `ptycho/workflows/components.py` |
| Configuration surface & legacy bridge | `specs/ptychodus_api_spec.md:20-125`, `213-291` | `ptycho/config/config.py`, `ptycho/params.py` |
| Data ingestion & grouping | `specs/ptychodus_api_spec.md:150-190` | `ptycho/raw_data.py`, `ptycho/loader.py`, `ptycho_torch/dset_loader_pt_mmap.py` |
| Model persistence contract | `specs/ptychodus_api_spec.md §4.6` | `ptycho/model_manager.py` |

### 3. Deliverables

1. PyTorch-backed configuration bridge that updates `ptycho.params.cfg` via the same dataclass pipeline as TensorFlow.
   - Configuration schema harmonization (resolve field name/type divergence: `grid_size` → `gridsize`, `mode` → `model_type`, add missing fields)
   - `KEY_MAPPINGS` translation layer for legacy dot-separated keys
   - Parameterized tests verifying all 75+ config fields propagate correctly
2. Data adapters producing grouping outputs and tensor layouts compatible with the reconstructor contract.
   - `RawDataTorch` shim delegating to `ptycho/raw_data.py` with memory-map bridging
   - `PtychoDataContainerTorch` matching TensorFlow tensor shapes/dtypes
   - Shared NPZ contract compliance with `datagen/` package
3. PyTorch model wrappers exposing the inference and training signatures referenced in the spec, including intensity scaling behaviour.
   - Barycentric reassembly modules with numeric parity validation vs TensorFlow
   - Multi-GPU inference path with DataParallel support
4. Workflow orchestration (training, inference, export) usable by ptychodus without changes to the UI layer.
   - Lightning + MLflow orchestration adapter or lower-level API exposure
   - Multi-stage training logic (stage_1/2/3_epochs with physics weight scheduling)
5. Save/load routines compatible with the existing archive semantics, including `params.cfg` restoration.
   - Lightning checkpoint + MLflow persistence adapter
   - Archive shim bundling `.ckpt` + `params.cfg` into `.h5.zip` format
   - Backend auto-detection for dual-format archives
6. Extended reconstructor capable of selecting the PyTorch backend, with regression tests demonstrating parity across backends.
   - Dual-backend reconstructor with runtime selection
   - Integration test suite (config bridge, data pipeline, training, persistence, Ptychodus integration)

### 4. Phased Work Breakdown

#### Phase 0 – Discovery & Design (1 sprint)

- Audit `ptycho_torch/` modules for reusable components and gaps relative to spec references.
- Confirm the capability to toggle backends from `PtychoPINNReconstructorLibrary` (`ptychodus/src/ptychodus/model/ptychopinn/core.py:22-59`).
- Produce a component parity map that lists each TensorFlow workflow dependency (`ptycho.probe`, `ptycho.train_pinn`, `ptycho.tf_helper`, `run_cdi_example`, `ModelManager`, etc.) alongside the existing or missing `ptycho_torch` counterpart. Flag gaps that need new code versus thin shims.
- Produce sequence diagrams mapping ptychodus calls to PyTorch equivalents, referencing both the reconstructor contract (`specs/ptychodus_api_spec.md §4`) and the parity map.
- **Decision Gate: High-Level API vs Low-Level Integration** — Determine whether `ptychodus` should invoke the `ptycho_torch/api/` layer (which provides `ConfigManager`, `PtychoModel`, `Trainer`, `InferenceEngine` classes with MLflow orchestration) or bypass it for direct module calls. Document ownership of MLflow persistence strategy and Lightning dependency policy. See `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/delta_log.md` Delta-2 for analysis.
- **API Package Structure Documentation** — Document the new `ptycho_torch/api/base_api.py` layer (994 lines), which abstracts Lightning orchestration and provides MLflow-centric persistence (`save_mlflow()`, `load_from_mlflow()`). Cross-reference with `specs/ptychodus_api_spec.md:129-212` to verify contract alignment.
- Acceptance: signed-off architecture note describing module boundaries, extension points, and integration surface decision (API layer or low-level modules).

**Phase 0 Artifact – TensorFlow ↔ PyTorch Parity Map**

| TensorFlow / Legacy Component | Responsibility | PyTorch Counterpart | Status / Gaps |
| --- | --- | --- | --- |
| `ptycho.probe` (probe guess setup) | Load/stash probe guesses, handle masks | `ptycho_torch/dset_loader_pt_mmap.py:get_probes`, `ptycho_torch/model.ProbeIllumination` | Core functionality exists; need shim so reconstructor can reuse without direct CLI assumptions. |
| `ptycho.raw_data.RawData.generate_grouped_data` | NPZ ingestion, neighbor grouping | `ptycho_torch/dset_loader_pt_mmap.py`, `ptycho_torch/patch_generator.py` | Grouping logic present; add `RawDataTorch` wrapper to consume ptychodus-exported NPZs and expose spec-compliant dictionary. |
| `ptycho.loader.PtychoDataContainer` | Tensor packaging for model input | _No direct equivalent_ | Must implement `PtychoDataContainerTorch` mirroring TensorFlow shapes/dtypes. |
| `ptycho.tf_helper` utilities | Patch reassembly, translation, diffraction | `ptycho_torch/helper.py` | Most helpers ported; verify APIs and add thin adapters where TensorFlow-specific signatures differ. |
| `ptycho.model` (Keras models) | Core PINN network + loss wiring | `ptycho_torch/model.py` | Architectural parity achieved; integrate with dataclass-driven configs and inference wrapper. |
| `ptycho.train_pinn`, `run_cdi_example`, `train_cdi_model`, `save_outputs` | End-to-end training/inference orchestration | _Missing_ | Need PyTorch orchestration layer adhering to reconstructor contract and existing CLI expectations. |
| `ptycho.model_manager`, `load_inference_bundle` | Model persistence & params restoration | _Missing_ | Design PyTorch archive format (or adaptor) compatible with current loader side effects. |
| TensorFlow `Model.predict` signature | Inference entry `model.predict([X * scale, offsets])` | `ptycho_torch/model.PtychoPINN`, `ptycho_torch/train.PtychoPINN.forward` | Forward path available; require wrapper that matches TensorFlow call signature and intensity scaling semantics. |
| `update_legacy_dict` usage through dataclasses | Config propagation to `params.cfg` | _Pending integration_ | Reuse existing dataclasses; replace singleton configs in `ptycho_torch/config_params.py` with dataclass-backed factories. |

#### Phase 1 – Configuration Parity (1 sprint)

- **Task 1.1**: Introduce shared dataclasses for PyTorch by importing `ModelConfig`, `TrainingConfig`, `InferenceConfig` instead of singleton dictionaries. Map existing default dictionaries in `ptycho_torch/config_params.py` into dataclass factory helpers.
  - **Critical Schema Harmonization Required:** Current `ptycho_torch/config_params.py` uses divergent schema (e.g., `grid_size: Tuple[int, int]` vs spec-mandated `gridsize: int`, `mode: 'Supervised' | 'Unsupervised'` vs `model_type: 'pinn' | 'supervised'`). See `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/delta_log.md` Delta-1 for full field mismatch inventory.
  - **Missing Spec-Mandated Fields:** Add `gaussian_smoothing_sigma`, `probe_scale`, `pad_object` to PyTorch config classes per `specs/ptychodus_api_spec.md:220-273`.
  - **Configuration Schema Mapping Table:** Create a mapping table documenting every PyTorch config field → spec-required field transformation (e.g., `grid_size[0]` → `gridsize`, `mode` → `model_type`) to guide implementation and testing.
- **Task 1.2**: Implement a PyTorch-friendly `update_legacy_dict()` invoker that calls the existing bridge (`specs/ptychodus_api_spec.md §§2-3`) immediately after dataclass instantiation.
  - **Add KEY_MAPPINGS for PyTorch:** Extend or create equivalent to `ptycho/config/config.py:KEY_MAPPINGS` to translate modern field names to legacy dot-separated keys (e.g., `object_big` → `object.big`, `probe_trainable` → `probe.trainable`).
  - **params.cfg Population Verification:** Ensure `ptycho.params.cfg` is correctly populated for all fields consumed by downstream modules (`ptycho/raw_data.py:365`, `ptycho/loader.py:178-181`, `ptycho/model.py:280`).
- **Task 1.3**: Auto-generate a parity checklist from every field in the configuration tables (`specs/ptychodus_api_spec.md §5`) and translate it into parameterized tests that set and round-trip values such as `probe_scale`, `gaussian_smoothing_sigma`, `sequential_sampling`, and other newly documented knobs.
  - **Test Coverage:** Parameterized tests must verify all 75+ fields across `ModelConfig`, `TrainingConfig`, `InferenceConfig` tables (§5.1-5.3) propagate correctly through dataclass → `update_legacy_dict` → `params.cfg` → downstream consumers.
- Acceptance: automated parity tests confirm each documented field maps identically into `ptycho.params.cfg`, and comparative snapshots show no differences between TensorFlow and PyTorch updates for a matrix of representative configurations.

#### Phase 2 – Data Ingestion & Grouping (2 sprints)

- **Task 2.1**: Implement a `RawDataTorch` shim that consumes the NPZ schema produced by `export_training_data()` (`specs/ptychodus_api_spec.md §4.5`) and exposes methods mirroring `RawData.generate_grouped_data` semantics.
  - **RawDataTorch Wrapper Scope:** Create adapter delegating to existing `ptycho/raw_data.py` for neighbor-aware grouping logic while bridging to PyTorch's memory-mapped dataset infrastructure.
  - **Memory-Map to Cache Bridging:** Reconcile PyTorch's `ptycho_torch/dset_loader_pt_mmap.py` memory-mapped tensor approach with TensorFlow's `.groups_cache.npz` caching strategy. Ensure cache reuse across backends for performance parity.
- **Task 2.2**: Map the Torch memory-mapped dataset outputs (`ptycho_torch/dset_loader_pt_mmap.py:1-260`) onto the dictionary keys enumerated in the contract (`specs/ptychodus_api_spec.md §4.3`).
  - **TensorDict Format Conversion:** PyTorch dataloader provides `TensorDict` format; must expose dictionary with keys `diffraction`, `coords_offsets`, `coords_relative`, `Y`, `nn_indices` to satisfy spec.
- **Task 2.3**: Provide a `PtychoDataContainerTorch` that matches TensorFlow tensor shapes and dtype expectations, ensuring compatibility with downstream reassembly helpers.
- **Task 2.4**: Document `ptycho_torch/datagen/` package for synthetic data parity.
  - **Synthetic Data Generation:** `ptycho_torch/datagen/datagen.py` provides `from_simulation()`, `simulate_multiple_experiments()` for dataset creation with Poisson scaling and beamstop. Verify outputs conform to `specs/data_contracts.md` NPZ schema.
  - **Shared NPZ Contract:** Ensure `datagen/` package produces NPZ files consumable by both TensorFlow and PyTorch backends without format divergence. Cross-reference with TensorFlow `ptycho.diffsim` equivalence for physics parity.
  - **Experimental Data Extraction:** Note `generate_data_from_experiment()` capability for supervised label extraction from experimental datasets, bypassing simulation workflow.
- Acceptance: integration tests load identical NPZ inputs through both backends and compare grouped dictionary keys, shapes, and coordinate values (float tolerances defined up front). Synthetic datasets from `datagen/` pass NPZ validation against `specs/data_contracts.md`.

#### Phase 3 – Inference Entry Point (1 sprint)

- **Task 3.1**: Wrap the PyTorch model so `predict([diffraction * intensity_scale, local_offsets])` is supported (per `specs/ptychodus_api_spec.md §4.4`).
- **Task 3.2**: Bridge the PyTorch intensity scaler to `params.cfg['intensity_scale']` and `params.cfg['intensity_scale.trainable']` using the logic in `ptycho_torch/model.py:480-565`.
- **Task 3.3**: Resolve probe initialisation: either reuse `ptycho.probe.set_probe_guess` via thin adapters or port equivalent functionality into `ptycho_torch` so the grouped data pipeline receives the expected probe tensor (`specs/ptychodus_api_spec.md §4.3`).
- **Task 3.4**: Document barycentric reassembly modules and establish parity with TensorFlow stitching.
  - **Alternative Reassembly Implementation:** PyTorch provides `ptycho_torch/reassembly_alpha.py`, `reassembly_beta.py`, `reassembly.py` implementing vectorized barycentric accumulator for patch stitching (alternative to `ptycho.tf_helper.reassemble_position`). See `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/delta_log.md` Delta-4.
  - **Multi-GPU Inference Path:** `reassembly_alpha.py:VectorizedBarycentricAccumulator` includes DataParallel support for multi-GPU patch stitching with performance profiling (inference time vs assembly time tracking).
  - **Parity Requirements:** Establish numeric comparators to verify PyTorch barycentric output matches TensorFlow `reassemble_position()` outputs within acceptable tolerances on synthetic test fixtures.
  - **Adaptation Strategy:** Either (A) adapt `ptycho.tf_helper.reassemble_position` usage by providing data conversion utilities, or (B) validate PyTorch reassembly parity and use native implementation. Document decision and testing approach.
- Acceptance: reconstructor `reconstruct()` executes end-to-end with the PyTorch backend on sample data, successfully initialises probes, and produces an object array of the expected shape and dtype. Numeric parity tests confirm stitching outputs match TensorFlow within defined tolerances.

#### Phase 4 – Training Workflow Parity (2 sprints)

- **Task 4.1**: Expose orchestrators analogous to `run_cdi_example`, `train_cdi_model`, and `save_outputs` that operate on PyTorch models (`specs/ptychodus_api_spec.md §4.5`).
  - **Lightning + MLflow Orchestration Divergence:** PyTorch training uses `ptycho_torch/train.py` with PyTorch Lightning `Trainer` (callbacks, DataModule, DDP strategy) and MLflow autologging, diverging from TensorFlow's `ptycho.workflows.components.run_cdi_example()` direct orchestration. See `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/delta_log.md` Delta-5.
  - **Multi-Stage Training Logic:** `train.py` embeds multi-stage training (`stage_1/2/3_epochs` with physics weight scheduling) in Lightning orchestration. Ensure spec alignment or document as PyTorch-specific enhancement.
  - **Orchestration Surface Decision:** Clarify whether `ptychodus` integration will invoke Lightning trainer directly or require lower-level model API for training. Cross-reference Phase 0 decision gate (API layer vs low-level integration).
- **Task 4.2**: Update `ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py:229-269` to dispatch to the PyTorch workflow when selected, preserving logging and output directory semantics.
- **Task 4.3**: Ensure NPZ exports from `export_training_data()` remain consumable by the PyTorch training path without schema divergence.
- **Task 4.4**: Integrate the probe strategy selected in Phase 3 into the training pipeline so probe guesses are initialised consistently before batching/grouping.
- Acceptance: training end-to-end test executes on a reduced dataset, writes artifacts to `output_dir`, restores the trained model for immediate inference, and probe initialisation runs without backend-specific fallbacks.

#### Phase 5 – Model Persistence & Archives (1 sprint)

- **Task 5.1**: Extend `ptycho.model_manager` with PyTorch-aware branches (or introduce a companion module) that produce bundle metadata alongside TensorFlow artefacts while sharing the `wts.h5.zip` packaging contract (`specs/ptychodus_api_spec.md §4.6`).
  - **Lightning Checkpoint + MLflow Workflow:** Current PyTorch implementation uses Lightning `.ckpt` format with MLflow artifact logging (`ptycho_torch/train.py:238-240`). Design adapter to wrap Lightning checkpoints in `.h5.zip`-compatible archives.
  - **Archive Shim Proposal:** Introduce persistence adapter that bundles Lightning checkpoint + `ptycho.params.cfg` snapshot + custom layer registry into unified `.h5.zip` format satisfying `MODEL_FILE_NAME = 'wts.h5.zip'` contract.
  - **MLflow Dependency Mitigation:** Define policy for optional MLflow/Lightning dependencies in CI and production environments. Consider graceful degradation or alternative persistence paths when MLflow unavailable.
- **Task 5.2**: Define the PyTorch payload layout (e.g., `diffraction_to_obj.pt`, optional optimizer state, serialized custom layers) and how it coexists with Keras assets inside the archive; document format versioning.
  - **Dual-Format Archive Structure:** Document how Lightning checkpoints and TensorFlow SavedModel assets coexist within `.h5.zip`, including versioning strategy to prevent loader collisions.
- **Task 5.3**: Update `load_inference_bundle` to inspect archive contents, dispatch to the appropriate loader (TensorFlow vs PyTorch), and restore `params.cfg` side effects for both paths.
  - **Backend Auto-Detection:** Implement archive introspection logic to detect TensorFlow vs PyTorch payloads and dispatch to appropriate loader without requiring user-specified backend flag.
- **Task 5.4**: Provide migration tooling or guidance for existing archives and adjust reconstructor file filters/tooltips to reflect dual-backend support.
- Acceptance: automated save→load tests confirm PyTorch models round-trip via `wts.h5.zip`, `load_inference_bundle` returns backend-specific objects with `params.cfg` restored, and legacy TensorFlow bundles remain unaffected.

#### Phase 6 – Reconstructor & UI Integration (1 sprint)

- **Task 6.1**: Extend `PtychoPINNReconstructorLibrary` to register PyTorch variants in addition to TensorFlow (`specs/ptychodus_api_spec.md §4.1`).
- **Task 6.2**: Provide backend selection controls (UI toggle or configuration flag) and ensure file filters remain accurate.
- **Task 6.3**: Update logging to indicate the active backend for traceability.
- Acceptance: switching between backends at runtime works without restarting ptychodus, and both paths respect the reconstructor contract.

#### Phase 7 – Validation & Regression Testing (1 sprint)

- **Task 7.1**: Develop regression suites comparing TensorFlow and PyTorch outputs on shared fixtures (object reconstructions, grouped data, loss curves) with acceptable tolerance thresholds.
- **Task 7.2**: Add integration tests covering save/load, inference, and training flows to prevent contract regressions.
- **Task 7.3**: Document manual verification steps aligned with the spec (e.g., verifying key mappings, checking NPZ contents).
- Acceptance: CI passes with new test coverage, and manual checklist is signed off.

#### Phase 8 – Spec & Ledger Synchronization

- **Task 8.1**: Update `specs/ptychodus_api_spec.md` to document PyTorch backend semantics once implementation finalizes.
  - **Configuration Schema Amendment:** If dual-schema approach is adopted (TensorFlow dataclasses vs PyTorch singletons), document both schemas and translation layer in spec §2-3 and field tables (§5.1-5.3).
  - **Persistence Contract Extension:** Document PyTorch archive format (Lightning checkpoint wrapper), backend auto-detection logic, and coexistence with TensorFlow bundles in §4.6.
  - **Workflow Divergence Notes:** Capture Lightning/MLflow orchestration differences vs TensorFlow direct orchestration in §4.5 (training workflow).
- **Task 8.2**: Update `docs/findings.md` knowledge ledger with PyTorch-specific lessons learned.
  - Document config schema harmonization patterns
  - Record reassembly parity validation approach
  - Capture persistence adapter design decisions
- **Task 8.3**: Cross-reference integration plan with downstream initiatives.
  - Ensure `plans/active/INTEGRATE-PYTORCH-001/implementation.md` consumes refreshed plan sections
  - Coordinate with `plans/pytorch_integration_test_plan.md` (TEST-PYTORCH-001) for fixture requirements
- **Task 8.4**: Update `docs/workflows/pytorch.md` with integration-specific usage patterns.
  - Document how to invoke PyTorch backend from Ptychodus
  - Provide configuration examples for dual-backend scenarios
  - Add troubleshooting guidance for common integration issues
- Acceptance: Spec updates are merged, knowledge ledger entries are validated against actual implementation outcomes, and downstream initiative plans reference refreshed canonical plan sections.

### 5. Dependencies & Risks

- **TensorFlow helper reuse**: section `4.4` of the spec assumes TensorFlow-specific helpers. Replacing them may require heavy refactoring or careful interop wrappers.
- **Archive compatibility**: deviating from `wts.h5.zip` could break existing saved models; introduce versioning and migration guidance if change is unavoidable.
- **Performance parity**: PyTorch data pipeline must be profiled against TensorFlow to ensure similar throughput, particularly for grouping operations.
- **API layer drift vs low-level integration**: The new `ptycho_torch/api/` high-level layer (ConfigManager, PtychoModel, InferenceEngine) may diverge from low-level module interfaces, creating maintenance burden if Ptychodus integration bypasses the API layer. Decision required in Phase 0 to commit to either (A) API-first integration with formal contract or (B) low-level integration with API layer as optional convenience. See `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/delta_log.md` Delta-2 for analysis.
- **Configuration schema divergence**: PyTorch config fields (`grid_size`, `mode`) differ from spec-mandated TensorFlow fields (`gridsize`, `model_type`), risking silent contract violations. Requires explicit harmonization strategy (refactor PyTorch schema vs dual-schema documentation) to prevent downstream integration failures.
- **Lightning/MLflow dependency policy**: PyTorch training relies on Lightning and MLflow; CI/production environments may lack these dependencies. Requires mitigation strategy (optional dependency with graceful degradation, or mandatory dependency with environment setup guidance).
- **Reassembly parity validation complexity**: PyTorch barycentric reassembly diverges from TensorFlow `tf_helper.reassemble_position`; numeric parity testing across different tensor layouts and interpolation strategies may reveal edge cases requiring tolerance tuning or algorithm adjustments.

### 6. Communication & Handoff

- Weekly sync to review progress against the spec sections cited above.
- Maintain a living checklist mapping completed tasks to spec requirements to guarantee no contracts are overlooked.
- Final deliverable includes updated documentation (README, user guides) explaining backend selection and any new configuration knobs.
