# Bug Report: Torch `object_big` Config Handling Is Ad Hoc

## Summary
The PyTorch path currently handles `object_big` in an ad hoc way (runner-level overrides and env toggles), rather than through consistent configuration dataclasses and bridge logic. This creates workflow-specific regressions (notably grid-lines Hybrid ResNet) and makes behavior depend on execution entrypoints instead of explicit config.

## Impact
- **Regression risk:** When `object_big=True` is propagated from TF `ModelConfig` defaults into Torch workflows, gridsize=1 pipelines can diverge from TF behavior and regress metrics.
- **Config drift:** Torch behavior depends on wrapper defaults and env overrides instead of the canonical `TrainingConfig/ModelConfig` dataclasses.
- **Hidden coupling:** `object_big` behavior becomes implicit in orchestration code rather than explicitly encoded in configs or dataset metadata.

## Evidence (Session Findings)
- Grid-lines TF workflow sets `ModelConfig(..., object_big=False)` during legacy config (`ptycho/workflows/grid_lines_workflow.py:184-205`).
- Torch grid-lines runner originally used TF `ModelConfig` defaults (`object_big=True`), which triggered object-big reassembly and regressed the hybrid_resnet integration test. Setting `object_big=False` in `scripts/studies/grid_lines_torch_runner.py` restored passing metrics.
- Object-big behavior is enforced via orchestration overrides and environment toggles rather than a stable config contract.

## Outstanding Issues
1. **Config ownership:** `object_big` should be defined and carried through config dataclasses (and/or dataset metadata), not patched in individual runners. This should be consistent between TF and Torch paths.
2. **Default semantics:** The correct default for `object_big` in Torch should be explicitly tied to workflow intent (e.g., grid-lines vs general ptychography), not inherited implicitly from TF defaults.
3. **Parity guarantees:** With `object_big=True` and gridsize=1, Torch reassembly applies centermask normalization. If this path is legitimate, it must be codified in configs and tested; otherwise it should be disabled or guarded at configuration time.

## Suggested Direction
- Define a single source of truth for `object_big` in configuration dataclasses (or derive it from dataset metadata / gridsize with a documented rule).
- Require workflow entrypoints to set the value explicitly in `TrainingConfig.model` instead of ad hoc overrides.
- Add a small config-bridge validation that warns when Torch + workflow defaults disagree with TF workflow defaults for the same dataset type.

## Related Artifacts
- Integration log (post-override): `.artifacts/object_big_relative_offsets/pytest_grid_lines_hybrid_resnet_object_big_false_default.log`
- Plan notes: `docs/plans/2026-02-05-bisect-recent-hybrid-resnet.md`

