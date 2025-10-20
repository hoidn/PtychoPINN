# ADR-003 C4.D3 Integration Failure Triage — Missing `wts.h5.zip`

**Date:** 2025-10-20
**Supervisor:** galph (debug loop)
**Focus:** [ADR-003-BACKEND-API] Phase C4.D3 — PyTorch integration workflow still failing after channel-sync fix

## Observed Failure
- Target selector: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv`
- Current result: `ValueError: Model archive not found .../training_outputs/wts.h5.zip`
- Evidence: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T070500Z/phase_c4_cli_integration_debug/pytest_integration.log` lines 160-227

## Hypotheses + Triage

### H1 — Training CLI still executes legacy `main()` pathway (no bundle persistence)
- **Expectation:** Phase C4 CLI refactor should delegate to workflow factories that call `run_cdi_example_torch`, which saves `wts.h5.zip` when `output_dir` is provided (spec §4.6, workflows/components.py:174-212).
- **Observation:** `ptycho_torch/train.py:489-620` shows the new CLI branch still invokes legacy `main()` instead of `run_cdi_example_torch`. The legacy `main()` path never calls `save_torch_bundle` (confirmed via `rg 'save_torch_bundle' ptycho_torch/train.py` → no matches).
- **Triage Outcome:** ✅ Likely root cause. Legacy flow only writes Lightning checkpoints under `checkpoints/`; inference CLI now requires spec-compliant bundle, so `wts.h5.zip` is never generated.

### H2 — Workflow persistence expects dual-model handles not returned by `_train_with_lightning`
- **Expectation:** `save_torch_bundle()` (`ptycho_torch/model_manager.py:40-148`) requires `models_dict` to contain `'autoencoder'` and `'diffraction_to_obj'` (nn.Module instances) to build the archive.
- **Observation:** `_train_with_lightning()` (`ptycho_torch/workflows/components.py:470-566`) currently returns `{'lightning_module': model, 'trainer': trainer}`. Even after we delegate to workflows, persistence will raise `RuntimeError` if `'trainer'` (not an nn.Module) is passed through.
- **Triage Outcome:** ⚠️ Secondary issue. Needs adjustment before enabling bundle persistence; otherwise integration will fail with `RuntimeError` even once H1 is addressed.

### H3 — Integration test helper still assumes checkpoint-only contract
- **Observation:** `_run_pytorch_workflow()` (`tests/torch/test_integration_workflow_torch.py:49-128`) captures only `checkpoints/last.ckpt`. It never asserts on `wts.h5.zip`, so Phase C4 updates changed inference expectation without updating helper.
- **Implication:** Test currently fails during inference before it can assert outputs. Once bundle persistence is restored, helper should capture both checkpoint and bundle paths to keep parity with TensorFlow baseline.

## Next Confirming Step
1. Update Phase C4 plan row C4.D3 to track the new persistence gap (legacy CLI vs workflow bundling).
2. Direct Ralph to prototype workflow-based training invocation (or inject `save_torch_bundle` into legacy `main()`) and adjust `_train_with_lightning` to emit dual-model dict compatible with `save_torch_bundle`.
3. Extend integration helper to validate presence of `wts.h5.zip` once persistence works.

## References
- specs/ptychodus_api_spec.md:149-205 — persistence contract (`wts.h5.zip` requirement)
- docs/workflows/pytorch.md §6 — training workflow must save `{output_dir}/wts.h5.zip`
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md (C4.D3 checklist)
