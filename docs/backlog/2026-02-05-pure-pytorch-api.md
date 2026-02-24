# Backlog: Implement Pure PyTorch API (PtychoModel)

## Summary
Add a pure PyTorch API (`PtychoModel`) that loads raw NPZ files, performs grouping in NumPy/SciPy, and runs training/inference without touching `ptycho.params.cfg` or TensorFlow bridge code.

## Impact
- **Clean API:** Enables a stateless, explicit data flow without CONFIG-001 global mutation.
- **Legacy isolation:** Keeps the legacy bridge path intact while providing a pure alternative.
- **Generator coverage:** Supports CNN/FNO/Hybrid/Hybrid-ResNet via the existing generator registry.

## Scope (Day One)
- Raw NPZ ingestion with `diffraction`/`diff3d` handling and NHW normalization.
- Random sampling only (`neighbor_count`, `n_groups`, `gridsize`), no sequential or oversampling.
- Unsupervised training path; supervised support deferred.
- Stitching via `ptycho_torch.reassembly` (no TF helpers).

## Out of Scope (Day One)
- Sequential sampling and oversampling controls.
- Pure extraction of `Y` patches from `objectGuess`.
- Legacy config bridge interop or `params.cfg` synchronization.

## Plan Reference
- Implementation plan: `docs/plans/2026-02-05-pure-pytorch-api.md`

## Risks
- Grouping parity may differ from TF `RawData.generate_grouped_data`.
- Reassembly parity depends on Torch helper settings and normalization behavior.

## Suggested Next Steps
- Implement the plan tasks with TDD and targeted pytest selectors.
- Add optional parity tests once the pure path is stable.

## 2026-02-24 Follow-up: Control-Plane Debt Alignment

This backlog item remains the long-term path for reducing dependency on TensorFlow dataclasses and `params.cfg` in Torch workflows.

### Additional Acceptance Criteria
1. Pure Torch API can express Torch-only architecture knobs without requiring shared dataclass additions.
2. Pure Torch path does not depend on `update_legacy_dict` for training/inference execution.
3. Parity-sensitive wrappers/runbooks can choose pure-Torch execution mode where cross-backend bridge churn is undesirable.

### Cross-Reference
- Ownership decision backlog: `docs/backlog/2026-02-24-torch-only-knob-ownership.md`
