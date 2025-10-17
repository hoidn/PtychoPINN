# Phase C.C3 Evidence — Memory-Mapped Dataset Bridge

**Initiative:** INTEGRATE-PYTORCH-001  
**Focus:** Phase C.C3 — Bridge memory-mapped dataset usage to RawDataTorch/PtychoDataContainerTorch  
**Date:** 2025-10-17T082035Z  
**Author:** galph supervisor (evidence loop)

---

## 1. Documents & Sources Reviewed
- `specs/data_contracts.md` — canonical grouped-data schema (keys, shapes, dtypes) referenced for parity.
- `specs/ptychodus_api_spec.md:164-215` — reconstructor data ingestion contract + grouping requirements.
- `docs/architecture.md:91-173` — TensorFlow data pipeline overview (RawData → loader).
- `plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md` — Phase C checklist + C.C3 scope guidance.
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/data_contract.md` — authoritative ROI + tensor expectations.
- `ptycho_torch/raw_data_bridge.py` — current RawDataTorch adapter delegating to TensorFlow RawData.
- `ptycho_torch/data_container_bridge.py` — new torch-optional container (Phase C.C2 artifact).
- `ptycho_torch/dset_loader_pt_mmap.py` & `ptycho_torch/dataloader.py` — existing PyTorch memory-mapped datasets.
- `tests/torch/test_data_pipeline.py` — TDD guardrails for RawDataTorch + container parity.

## 2. Current PyTorch Dataset Behaviour
- Both `ptycho_torch/dset_loader_pt_mmap.py` and `ptycho_torch/dataloader.py` build a `TensorDict`-based memory map backed by `tensordict.MemoryMappedTensor`.
- They rely on singleton configs (`ModelConfig`, `DataConfig`, `TrainingConfig`) and global `params` access (e.g., `ModelConfig().get('object.big')`).
- Grouping logic re-implements nearest-neighbour selection (via `group_coords`, `get_relative_coords`) instead of delegating to TensorFlow `RawData.generate_grouped_data()`.
- Output payload per `__getitem__` is `(TensorDict, probe_tensor, scaling_constant)` — not the grouped-data dict required by `RawDataTorch`/`PtychoDataContainerTorch`.
- Cache directory (`data/memmap/`) stores diffraction stacks + coordinate tensors but is unaware of new dataclass-driven config bridge.

## 3. Gap Analysis vs Phase C Objectives
1. **Duplicate grouping implementation** — TensorDict datasets compute their own neighbour indices and offsets. Risk: divergence from TensorFlow grouping contract already encoded in `RawDataTorch` tests.
2. **Config bridge bypass** — Singleton configs mean `update_legacy_dict` never runs, violating CONFIG-001 when adapters are used downstream.
3. **Output format mismatch** — Downstream PyTorch workflow expects `TensorDict` objects while Phase C requires grouped-data dict compatible with `PtychoDataContainerTorch`.
4. **Cache lifecycle** — `.groups_cache.npz` produced by TensorFlow RawData is not reused; current PyTorch cache lives under `data/memmap/` with different semantics.
5. **Torch hard dependency** — Dataset modules import torch unconditionally, conflicting with torch-optional parity expectations enforced in tests.
6. **Probe scaling + metadata** — Present dataset injects probe scaling + normalization factors that have no analogue in TensorFlow wrapper; integration plan must clarify retention strategy.

## 4. Recommended Bridge Strategy (Phased)
### C.C3.A — Delegate Grouping to RawDataTorch
- Instantiate `RawDataTorch` inside a new adapter (e.g., `ptycho_torch/memmap_bridge.py`) using dataclass configs created via `config_bridge`.
- Use RawDataTorch.generate_grouped_data(...) to obtain canonical grouped data. Persist results to memmap tensors to preserve streaming benefits.
- Preserve `.groups_cache.npz` by delegating to TensorFlow RawData; capture evidence showing cache reuse when running twice with same dataset.

### C.C3.B — Harmonise Dataset Output
- Refactor existing `PtychoDataset` to emit grouped-data dicts (or lightweight view objects) consumable by `PtychoDataContainerTorch`.
- Provide thin compatibility wrapper so Lightning training paths can request `TensorDict` batches by wrapping grouped data after PyTorch container construction.
- Ensure adapters remain torch-optional: import torch lazily and fall back to NumPy arrays when unavailable.

### C.C3.C — Config & Metadata Alignment
- Replace singleton accesses with dataclass configs produced via `ptycho_torch.config_bridge.to_training_config()` (for training) or `.to_inference_config()` (for inference).
- Call `update_legacy_dict` exactly once per dataset initialisation to satisfy CONFIG-001; document the call in code comments referencing finding ID.
- Decide how to surface `scaling_constant`/`probe_scaling`: either enrich grouped-data dict with optional keys (documented in plan) or handle in higher-level workflow (Phase D).

## 5. Test & Artifact Implications
- Extend `tests/torch/test_data_pipeline.py` with new parametrized cases covering:
  1. Memory-mapped dataset path returning grouped data identical to RawDataTorch baseline (`np.allclose` on coords/nn_indices, dtype checks).
  2. Cache reuse: assert second instantiation reuses `.groups_cache.npz` and `data/memmap` without recomputation (capture timestamps / log markers).
- Targeted selectors to capture once implementation lands:
  - `pytest tests/torch/test_data_pipeline.py -k "memmap" -vv`
  - `pytest tests/torch/test_data_pipeline.py -k "cache_reuse" -vv`
- Artifact directory for implementation loop should include cache state snapshots plus pytest logs (per Phase C.D2 guidance).

## 6. Open Questions for Implementation Loop
1. **TensorDict retention** — Do we maintain TensorDict outputs for Lightning compatibility, or can orchestrators consume `PtychoDataContainerTorch` directly? Proposal: keep TensorDict convenience function that wraps canonical grouped data, but treat grouped dict as source of truth.
2. **Cache location standardisation** — Should PyTorch reuse TensorFlow `.groups_cache.npz` path or adopt shared location under `tmp/`? Need decision before finalising tests (tie into Phase C.D2).
3. **Performance parity** — If delegation to RawDataTorch becomes a bottleneck, consider caching grouped data once and memory-mapping the result. Document profiling hooks for follow-up.

---

**Outcome:** Evidence captured for Phase C.C3. Ready to brief engineer on delegation strategy, required tests, and open decisions. Next supervisor step: update `docs/fix_plan.md` Attempts History and refresh Phase C checklist states before issuing new input.md directives.
