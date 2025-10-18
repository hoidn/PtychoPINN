# Phase B.B1 Test Design — Lightning Orchestration (2025-10-18T000606Z)

## Context
- Initiative: INTEGRATE-PYTORCH-001
- Focus: Phase D2 Completion — Task B1 (author failing Lightning regression tests)
- Source plan: `plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md` (Phase B checklist)
- Data/Spec references:
  - `specs/ptychodus_api_spec.md` §4.5 — Reconstructor lifecycle + persistence contract
  - `docs/workflows/pytorch.md` §§5–7 — Lightning trainer expectations
  - `docs/TESTING_GUIDE.md` §"Test Types" — Integration vs unit test boundaries
  - `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T233109Z/phase_d2_completion/baseline.md` §§1–2 — Current stub behaviour and failure log

## Test Objectives
1. Encode the Lightning orchestration contract before implementation, ensuring `_train_with_lightning` builds the Lightning module with the expected config objects.
2. Validate that Lightning's `Trainer.fit` is invoked with containers-backed dataloaders, establishing integration with Phase C adapters.
3. Require the training results dict to surface the trained module for downstream persistence (`save_torch_bundle`) and parity logging.

## Proposed Test Additions (all in `tests/torch/test_workflows_components.py`)

### 1. `test_train_with_lightning_instantiates_module`
- **Purpose:** Assert `_train_with_lightning` constructs `ptycho_torch.model.PtychoPINN_Lightning` using the four config objects required for checkpoint reload.
- **Setup:**
  - Create minimal `TrainingConfig` via existing fixture.
  - Build sentinel train/test containers (plain dicts acceptable for red phase).
  - Monkeypatch `ptycho_torch.model.PtychoPINN_Lightning` to record `__init__` args and return a stub module with `save_hyperparameters` no-op.
- **Assertion:** After calling `_train_with_lightning`, the spy must report exactly `(model_config, data_config, training_config, inference_config)` as positional args.
- **Expected failure today:** Stub never instantiates the Lightning module → spy not called → assertion fails.

### 2. `test_train_with_lightning_runs_trainer_fit`
- **Purpose:** Confirm `_train_with_lightning` wires Lightning's `Trainer.fit` with dataloaders derived from the provided containers.
- **Setup:**
  - Monkeypatch `lightning.pytorch.Trainer` constructor to return a stub object exposing `fit_called` flag.
  - Monkeypatch helper(s) that will produce dataloaders (e.g., `ptycho_torch.workflows.components._build_lightning_dataloaders`) once implemented; for red phase, expose a sentinel returning `(train_loader, val_loader)` so the test can validate they were passed through.
  - Invoke `_train_with_lightning`; capture arguments passed to the stub `fit` method.
- **Assertions:**
  - `Trainer.fit` invoked exactly once.
  - First dataloader arg matches the sentinel derived from `train_container`.
  - Validation loader is `None` when `test_container` is `None` (also add a parametrized case where validation data exists).
- **Expected failure today:** Stub never constructs trainer or calls `fit` → flag remains False.

### 3. `test_train_with_lightning_returns_models_dict`
- **Purpose:** Require training results to include the trained Lightning module under a deterministic key for downstream persistence tests (Phase D2.D and Phase D4).
- **Setup:** Reuse monkeypatch from Test 1 so `_train_with_lightning` returns a stub Lightning module handle.
- **Assertion:** Results dict must contain `"models"` mapping with `'lightning_module'` (or `'diffraction_to_obj'`) pointing to the returned module. This mirrors TensorFlow workflow expectations (`train_cdi_model` returning model handles for saving).
- **Expected failure today:** Stub returns only history/train/test containers → missing `models` key → assertion fails.

## Pytest Selector & Logging
- Author tests in a new class `TestTrainWithLightningRed` inside `tests/torch/test_workflows_components.py` to keep scope isolated from existing scaffold.
- Targeted red run: `pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv`
- Capture output with `tee` into `plans/active/INTEGRATE-PYTORCH-001/reports/<TS>/phase_d2_completion/pytest_train_red.log` once the tests exist.

## Implementation Notes for Future Loop
- When turning tests green, prefer factoring Lightning-specific helpers inside `ptycho_torch/workflows/components.py` (e.g., `_build_lightning_dataloaders`) to keep monkeypatching ergonomic.
- Ensure tests remain torch-optional by guarding imports (`pytest.importorskip("torch")`) only when absolutely necessary; current plan keeps everything pure-python via monkeypatching.
- Coordinate with Phase D4 persistence tests so the `'models'` key naming is consistent across workflow and inference paths.

## Next Actions
1. Update `plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md` B1 row to reference this design and new artifact path.
2. Instruct engineer (input.md) to implement these tests verbatim, run the targeted selector, and attach the red log under `2025-10-18T000606Z` report directory.
3. After tests land, advance to Phase B2 implementation with the same timestamp series for continuity.
