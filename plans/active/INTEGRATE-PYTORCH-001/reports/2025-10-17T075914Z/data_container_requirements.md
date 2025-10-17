# Phase C.C2 Evidence — PtychoDataContainerTorch Requirements

**Initiative:** INTEGRATE-PYTORCH-001  
**Focus:** Phase C.C2 — Implement torch-optional `PtychoDataContainerTorch`  
**Date:** 2025-10-17  
**Author:** galph supervisor (evidence loop)

---

## 1. Documents Consulted
- `specs/ptychodus_api_spec.md:164-188` — Defines reconstructor data ingestion contract and enumerates `PtychoDataContainer` tensor expectations.
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/data_contract.md` — Phase C.A1 canonical table for grouped data + container attributes (shapes, dtypes, ROI).
- `ptycho/loader.py:93-210` — Source of TensorFlow `PtychoDataContainer` constructor and `load()` helper that transforms grouped data dicts into tensors.
- `tests/torch/test_data_pipeline.py:1-260` — TDD red-phase tests documenting the expected PyTorch container API, dtype rules, and DATA-001 enforcement.
- `ptycho_torch/raw_data_bridge.py` — Confirms upstream adapter returns the exact grouped-data dict consumed by TensorFlow loader.

## 2. TensorFlow Baseline Inspection
A synthetic dataset matching Phase C ROI (N=64, gridsize=2, nsamples=10) was fed through `loader.load()` to capture authoritative shapes/dtypes.

| Attribute | Shape | dtype | Notes |
|-----------|-------|-------|-------|
| `X` | `(10, 64, 64, 4)` | `float32` | Diffraction tensor (normalized amplitude) |
| `Y` | `(10, 64, 64, 4)` | `complex64` | Combined ground-truth patches (critical DATA-001 guardrail) |
| `Y_I` | `(10, 64, 64, 4)` | `float32` | Amplitude component from `tf.math.abs(Y)` |
| `Y_phi` | `(10, 64, 64, 4)` | `float32` | Phase component from `tf.math.angle(Y)` |
| `coords_nominal` | `(10, 1, 2, 4)` | `float32` | Nominal scan coordinates; shaped per gridsize^2 |
| `coords_true` | `(10, 1, 2, 4)` | `float32` | True scan coordinates (Phase C tests compare to nominal) |
| `nn_indices` | `(10, 4)` | `int32` | Nearest neighbor indices per grouped sample |
| `global_offsets` | `(10, 1, 2, 1)` | `float64` | Translation offsets; note double precision in TF baseline |
| `local_offsets` | `(10, 1, 2, 4)` | `float64` | Per-channel offsets |
| `probe` | `(64, 64)` | `complex64` | Probe guess forwarded from grouped data |

Supporting log snippet (captured via standalone script, see `trace_baseline.txt` in this directory):
```
<PtychoDataContainer X=(10, 64, 64, 4) Y_I=(10, 64, 64, 4) Y_phi=(10, 64, 64, 4) ...>
X: shape=(10, 64, 64, 4), dtype=float32
Y: shape=(10, 64, 64, 4), dtype=complex64
...
```

## 3. Required Torch-Optional Behaviour
`tests/torch/test_data_pipeline.py` encodes the parity contract:
- Module must import without PyTorch; when torch unavailable, adapters should emit NumPy arrays. If torch is available, emit torch tensors with identical shapes/dtypes (`float32` ↔ `torch.float32`, `complex64` ↔ `torch.complex64`).
- Accessor surface must match TensorFlow container (`X`, `Y`, `Y_I`, `Y_phi`, `coords_nominal`, `coords_true`, `nn_indices`, `global_offsets`, `local_offsets`, `probe`).
- Construction path should reuse grouped-data dict produced by `RawDataTorch.generate_grouped_data()` to avoid divergence (delegation pattern mirrors Phase C.C1). Recompute Y amplitude/phase using NumPy/Torch helpers that respect complex dtypes.
- DATA-001 guardrail demands explicit dtype assertions and informative error messages if conversions drift to float64.

## 4. Implementation Guidance for Ralph
1. **Module placement:** recommended `ptycho_torch/data_container_bridge.py` (exported from `ptycho_torch/__init__.py` alongside `RawDataTorch`). Keep file torch-optional via guarded imports.  
2. **Factory API:** add `PtychoDataContainerTorch.from_grouped_data(grouped: dict, probe, *, use_torch=None)` and convenience constructor `from_raw_data(raw: RawDataTorch, ...)` mirroring TensorFlow `load()`. Accept optional flag to force NumPy fallback for deterministic tests.  
3. **Conversion helpers:**  
   - When torch available: wrap arrays with `torch.from_numpy(...)` and ensure `.to(torch.float32)` / `.to(torch.complex64)` conversions after copy.  
   - When torch missing: return NumPy arrays; use `np.asarray(..., dtype=...)` to lock types.  
   - For amplitude/phase: if torch tensors, use `torch.abs` and `torch.angle`; otherwise `np.abs`, `np.angle`.  
4. **Metadata preservation:** forward `nn_indices`, `global_offsets`, `local_offsets`, and optional `objectGuess` exactly as TensorFlow loader. Tests expect equality with TF baseline (for now they assert shape/dtype; follow-up phases may compare values).  
5. **Params bridge:** rely on upstream RawDataTorch for `params.cfg` initialization; container should not mutate legacy state to avoid double-initialization. Document this assumption in module docstring.  
6. **Torch detection:** replicate `TORCH_AVAILABLE` pattern from `raw_data_bridge.py` to keep import side effects predictable.  
7. **Debug string:** implement `__repr__` summarizing shapes/dtypes (mirrors TensorFlow container) to aid parity diffing.

## 5. Next Steps & Test Hooks
- Targeted selectors to unblock once implementation lands:  
  - `pytest tests/torch/test_data_pipeline.py -k "data_container" -vv`  
  - `pytest tests/torch/test_data_pipeline.py -k "Y dtype" -vv`
- Artifacts from this evidence loop stored alongside this file; stash green logs beside implementation when available (`pytest_data_container_green.log`, etc.).

## 6. Open Questions
- **global/local offsets dtype:** TensorFlow baseline currently emits float64. Decide whether to preserve float64 (simpler parity) or downcast to float32 when torch is available. Recommend following TensorFlow exactly (float64) to avoid silent precision change; document if casting occurs.  
- **Probe handling:** Tests currently require attribute presence only; if `torch` present, consider returning torch tensor created via `torch.from_numpy(probe)`. Validate against downstream consumer expectations in Phase D.  
- **Split handling:** TensorFlow loader accepts optional train/test splits (via `create_split=True`). Determine whether PyTorch adapter needs equivalent entry point or whether Phase C scope can focus on unsplit datasets (current tests use `create_split=False`).

---

**Outcome:** Evidence captured for Phase C.C2. Ready to brief engineer with implementation checklist in `input.md` and update `docs/fix_plan.md` Attempts History accordingly.
