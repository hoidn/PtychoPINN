# Phase D2.B — PyTorch Training Path Evidence

**Date:** 2025-10-17
**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** D2.B (Implement training orchestration)
**Mode:** Parity

---

## Context & Scope
- TensorFlow baseline lives in `ptycho/workflows/components.py:573` (`train_cdi_model`) and `run_cdi_example` just below it. This flow: RawData → `create_ptycho_data_container` → probe init → `train_pinn.train_eval` → optional stitching.
- PyTorch scaffolding exists via Phase D2.A (`ptycho_torch/workflows/components.py:1-213`) but all entry points still raise `NotImplementedError`.
- Phase C adapters (`ptycho_torch/raw_data_bridge.py`, `ptycho_torch/data_container_bridge.py`, `ptycho_torch/memmap_bridge.py`) now provide torch-optional RawData/PtychoDataContainer equivalents ready for orchestration.
- We must orchestrate PyTorch Lightning training while keeping CONFIG-001 enforced and honoring spec requirements in `specs/ptychodus_api_spec.md:186-189` (config-handshake and persistence expectations).

---

## Baseline Callchain (TensorFlow)
1. **Config sync** — `run_cdi_example` calls `update_legacy_dict(params.cfg, config)` immediately (`ptycho/workflows/components.py:650`).
2. **Container construction** — `train_cdi_model` converts RawData/PtychoDataContainer via `create_ptycho_data_container` (`ptycho/workflows/components.py:535-566`).
3. **Probe init** — `probe.set_probe_guess(None, train_container.probe)` ensures global probe state (`ptycho/workflows/components.py:600`).
4. **Training** — `train_pinn.train_eval(PtychoDataset(train_container, test_container))` returns history dict with containers (`ptycho/workflows/components.py:607-614`).
5. **Results contract** — Caller expects keys `train_container`, `test_container`, and optionally `reconstructed_obj` for stitching (`ptycho/workflows/components.py:615-666`).

---

## Current PyTorch Assets
- **Lightning training entry point** — `ptycho_torch/train.py:1-170` assembles configs, builds `PtychoDataModule`, instantiates `PtychoPINN_Lightning`, and runs `trainer.fit` with MLflow autologging.
- **DataModule** — `ptycho_torch/train_utils.py:217-320` handles memmap dataset creation (`PtychoDataset`), splits, and dataloaders.
- **Model module** — `ptycho_torch/model.py` (not opened this loop per CLAUDE guidance) already mirrors TensorFlow architecture.
- **Adapters** — Phase C bridges (RawDataTorch, PtychoDataContainerTorch, MemmapDatasetBridge) expose tensor-ready structures without needing PyTorch runtime.
- **Scaffold** — `ptycho_torch/workflows/components.py:98-210` ensures CONFIG-001 compliance and torch-optional imports; ready for concrete implementation.

---

## Gap Analysis (What D2.B Must Deliver)
| Requirement | Baseline Source | PyTorch Target | Notes |
| --- | --- | --- | --- |
| Train/Test container creation | `create_ptycho_data_container` (`ptycho/workflows/components.py:535-566`) | Use Phase C adapters to normalize inputs (`RawData`, `RawDataTorch`, or grouped dict) → `PtychoDataContainerTorch` | Ensure `update_legacy_dict` already called at entry; reuse `RawDataTorch.generate_grouped_data` for RawData inputs. |
| Probe initialization | `probe.set_probe_guess` (`ptycho/workflows/components.py:600`) | Adopt analogous hook via Torch-specific probe utility (likely reuse `ptycho.probe` for now) | Validate spec compliance for `probe.trainable`; capture TODO if PyTorch lacks equivalent setter. |
| Trainer orchestration | `train_pinn.train_eval` + `PtychoDataset` | Invoke Lightning workflow (`ptycho_torch/train.py`) without CLI side effects; prefer extracting reusable helper instead of shelling out. | Consider factoring a new `ptycho_torch/training_orchestrator.py` that accepts dataclass config + containers, bypassing MLflow when config requests. |
| Results object | TF returns dict with containers/history | Mirror structure: include `train_container`, `test_container`, `history`, plus Lightning/MLflow metadata as needed. | Document differences explicitly if unavoidable; update spec if semantics change. |
| Torch-optional testability | N/A (TF only) | Keep ability to import and run tests without torch. For D2.B tests, use monkeypatch + sentinel classes when torch absent. | Tests may rely on stubs or skip if `TORCH_AVAILABLE` False per `tests/conftest.py`. |

---

## Proposed Implementation Steps
1. **Normalize Train/Test Inputs**
   - Accept `RawData`, `RawDataTorch`, or `PtychoDataContainerTorch` (matching scaffold type hints).
   - Write helper `_ensure_container(data, config)` that returns `PtychoDataContainerTorch` using:
     - RawData → `RawDataTorch(..., config)` → `generate_grouped_data` → `PtychoDataContainerTorch.from_grouped_data(...)` (constructor to add if missing) or direct constructor.
     - RawDataTorch → call `generate_grouped_data` with config fields.
     - Existing container → return as-is after asserting dtype/shape per `tests/torch/test_data_pipeline.py` contract.

2. **Bridge to Lightning**
   - Prefer extracting a reusable orchestration helper from `ptycho_torch/train.py:30-170`. Options:
     - Move config-handling + trainer setup into new function `build_trainer(config, train_container, test_container)` inside `ptycho_torch/workflows/components.py`.
     - Or create dedicated module `ptycho_torch/workflows/trainer.py` returning `(trainer, model, datamodule)`; ensures torch-optional import guard.
   - Respect spec fields: `config.nepochs`, `config.batch_size`, `config.n_groups`, `config.neighbor_count`, `config.model.probe_trainable`.
   - Provide toggle to disable MLflow when running in parity tests (pull from config override or env var; record decision in plan).

3. **Assemble Results Dict**
   - Collect Lightning history (loss curves) via `trainer.callback_metrics` or manual logging to mimic `train_pinn.train_eval` output.
   - Ensure returning `train_container` and optional `test_container` for downstream inference/stitching.
   - Capture checkpoint path to feed Phase D3 persistence work.

4. **Error Handling & Logging**
   - Mirror TensorFlow warnings for missing test data.
   - Validate required overrides (e.g., nphotons) via existing config_bridge checks before training begins.

---

## TDD & Test Strategy
- **New pytest target**: Extend `tests/torch/test_workflows_components.py` with `TestWorkflowsComponentsTraining`. Start with red-phase test that spies on `_ensure_container` + Lightning helper to confirm orchestration order without running full training.
- **Selector**: `pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_train_cdi_model_torch_invokes_lightning -vv`
- **Approach**: Use monkeypatch to stub Lightning trainer/model so tests remain torch-optional; assert: 
  1. `update_legacy_dict` already invoked (existing test covers entry point).
  2. `_ensure_container` called for train/test inputs.
  3. Lightning helper invoked with translated config + containers.
  4. Result dict includes expected keys.
- **Deferred full training**: Actual Lightning run will be validated later in integration tests (TEST-PYTORCH-001). Keep unit tests fast.

---

## Open Questions / Risks
- **Probe handling**: TensorFlow writes into global `probe` module (`ptycho/workflows/components.py:596-603`). Do we replicate via TensorFlow helper or add PyTorch-native probe management? Need design decision before unfrozen-probe features.
- **Lightning dependency injection**: decide whether to factor training loop into reusable function or instantiate inside `train_cdi_model_torch`. Reuse improves testability but increases refactor scope.
- **MLflow optionality**: Must expose config-driven disable switch to keep CI green. Evaluate reading `config.output_dir` vs. env var.
- **Device management**: Respect CPU-only runs by honoring config flags (`config.device` once ported from PyTorch singleton).

---

## Recommended Next Actions for Phase D2.B
1. Author `_ensure_container` helper + targeted pytest (red).
2. Stub Lightning orchestrator (monkeypatchable) and update test to expect call.
3. Implement training path using Phase C adapters + Lightning helper (green).
4. Capture artifact logs (pytest + summary) under `plans/active/INTEGRATE-PYTORCH-001/reports/<new-iso>/` per parity_green_plan guidance.

