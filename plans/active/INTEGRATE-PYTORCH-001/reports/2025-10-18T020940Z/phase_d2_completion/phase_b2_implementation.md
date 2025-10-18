# Phase B.B2 Lightning Orchestration Blueprint (2025-10-18T020940Z)

## Context
- Initiative: INTEGRATE-PYTORCH-001 — PyTorch backend integration
- Phase Goal: Turn `_train_with_lightning` from a stub into a full Lightning training orchestrator that satisfies the new red tests and unblocks checkpoint persistence for Phase D2.
- Dependencies:
  - `plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md` (source checklist)
  - `specs/ptychodus_api_spec.md` §4.5 (reconstructor lifecycle + persistence contract)
  - `docs/workflows/pytorch.md` §§5–7 (Lightning workflow requirements)
  - Findings: POLICY-001 (PyTorch mandatory), CONFIG-001 (params.cfg init), FORMAT-001 (data contract guardrails)
  - Baseline evidence: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T233109Z/phase_d2_completion/baseline.md`
  - Test design: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T000606Z/phase_d2_completion/phase_b_test_design.md`
- Reference Surfaces:
  - `ptycho_torch/workflows/components.py` (stub to implement)
  - `ptycho_torch/train.py` (existing Lightning orchestration for CLI parity)
  - `ptycho_torch/model.py` (`PtychoPINN_Lightning` constructor contract)
  - `ptycho_torch/dset_loader_pt_mmap.py` (TensorDict dataloaders)
  - `tests/torch/test_workflows_components.py::TestTrainWithLightningRed` (acceptance tests)

### Phase B.B2 — Lightning Orchestration Implementation
Goal: Satisfy the TestTrainWithLightningRed suite by wiring `_train_with_lightning` to instantiate the Lightning module, build dataloaders, execute training, and return a persistence-ready results dict.
Prereqs: 
- Phase B.B1 red tests committed (✅)
- Torch extras installed (`pip install -e .[torch]`)
- Canonical dataset available (Run1084... NPZ or synthetic fixture); ensure data contract compliance (FORMAT-001)
- Artifact directory prepared: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T031500Z/phase_d2_completion/`
Exit Criteria:
- `pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv` passes with logs stored at `.../pytest_train_green.log`
- Results dict from `_train_with_lightning` includes `'history'`, `'train_container'`, `'test_container'`, and `'models'` with a Lightning module handle
- Checkpoint saved with hyperparameters so `PtychoPINN_Lightning.load_from_checkpoint(...)` succeeds (validated during Phase D tasks)

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B2.1 | Derive Lightning configs | [ ] | Inside `_train_with_lightning`, unpack `config` into the four Lightning config objects. Prefer reusing helpers already used by `train_cdi_model_torch` (e.g., precomputed `config.model`, `config.training`). Capture any additional inference config needed for persistence. |
| B2.2 | Ensure torch-optional imports | [ ] | Import `PtychoPINN_Lightning` and `Trainer` inside the function to keep module torch-optional. Guard missing torch with informative `RuntimeError` honoring POLICY-001 (reuse pattern from `train.py`). |
| B2.3 | Build dataloaders | [ ] | Introduce a helper (`_build_lightning_dataloaders`) if needed to convert `PtychoDataContainerTorch` into `TensorDictDataLoader` instances. Training loader should honor `batch_size=config.batch_size or default`, `shuffle=not config.sequential_sampling`, deterministic seeds from `config.random_seed`/`config.subsample_seed`. Validation loader optional when `test_container` exists. Document helper in module docstring for future parity tests. |
| B2.4 | Instantiate Lightning module | [ ] | Call `PtychoPINN_Lightning(model_config, data_config, training_config, inference_config)`, then `model.save_hyperparameters()` so checkpoint serialization captures configs. Update `params.cfg` is already handled by caller; no duplication. |
| B2.5 | Configure Trainer | [ ] | Construct `Trainer` with `max_epochs=config.nepochs`, `accelerator='auto'`, `devices=1` unless `config.device` is provided, `log_every_n_steps=1`, `default_root_dir=config.output_dir`. Respect `config.debug` for deterministic flags (e.g., set `enable_progress_bar=config.debug`). Defer MLflow toggles to B3. |
| B2.6 | Execute fit cycle | [ ] | Run `trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)` capturing exceptions to raise actionable errors. After fit, materialize history via `trainer.callback_metrics` (`train_loss`, `val_loss`). |
| B2.7 | Build results payload | [ ] | Return dict including `history`, original containers, and `'models': {'lightning_module': model}`. Add `'trainer'` handle if downstream needs manual checkpointing. Ensure structure mirrors TensorFlow `train_cdi_model` result semantics. |
| B2.8 | Document artifacts & tests | [ ] | After implementation, run targeted selector `pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T031500Z/phase_d2_completion/pytest_train_green.log`. Update docs/fix_plan Attempts and plan checklist B2 with artifact references. |

### Validation & Evidence Capture
- **Primary selector:** `pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv`
- **Optional smoke:** `pytest tests/torch/test_integration_workflow_torch.py -k train_save -vv` (if runtime permits; log under same timestamp as `pytest_train_green.log`)
- **Artifacts:** Store logs and any debugging notes within the timestamped report directory (`summary.md`, `pytest_train_green.log`, `trainer_debug.json` if captured).

### Risks & Mitigations
- **Torch availability:** If torch import fails, attach failure log and mark attempt blocked, referencing POLICY-001.
- **Dataloader mismatch:** If `PtychoDataContainerTorch` lacks direct dataloader API, design thin adapter that mirrors existing CLI pipeline.
- **Checkpoint size:** Ensure `config.output_dir` is writable; use tmpdir in tests to avoid polluting repo.
- **Non-deterministic seeds:** Propagate `config.subsample_seed` to both dataset shuffling and Lightning's global seed (`lightning.pytorch.seed_everything`).

### Follow-on Work (Phase B.B3)
- After orchestration is green, extend plan to cover MLflow toggles and documentation updates. Maintain reference to this blueprint for context.
