# PtychoModel Facade — Design

**Goal:** Provide a minimal, policy‑safe OO façade (`PtychoModel`) that exposes a clean `train`/`infer` flow while preserving existing PyTorch workflows and legacy guardrails.

**Constraints (must honor):**
- `update_legacy_dict(params.cfg, config)` is still mandatory before touching legacy modules (CONFIG‑001).
- Core physics modules remain unchanged.
- Probe is data; optional override via `probe_path`.
- Dataset‑level physics scale (`intensity_scale`) stays distinct from model‑internal scaling (`IntensityScalerModule`).
- Memmap data flow must remain valid and explicit.

## Architecture Overview

**Primary idea:** Add a thin façade on top of `ptycho_torch/workflows/components.py` without changing its public entry points or config bridge. The façade owns config construction, data/probe resolution, and legacy‑bridge invocation, then delegates to existing helpers.

**Modules (new):**
- `ptycho_torch/api/model.py` — `PtychoModel` façade.
- `ptycho_torch/api/data_resolver.py` — helper to normalize data inputs (paths vs objects).
- `ptycho_torch/legacy_bridge.py` — single place to call `update_legacy_dict`.

**Key dependencies (existing):**
- `ptycho_torch/workflows/components.py` (`train_cdi_model_torch`, `_ensure_container`).
- `ptycho_torch/dataloader.py` (`PtychoDataset`, `TensorDictDataLoader`).
- `ptycho_torch/model_manager.py` (`save_torch_bundle`, `load_torch_bundle`).
- `ptycho_torch/inference.py` (`_run_inference_and_reconstruct`).

## API Surface

```python
model = PtychoModel(
    arch="hybrid_resnet",
    model_params={...},
    training_params={...},
    execution_params={...},
    probe_path=None,
    use_memmap=False,
)

results = model.train(train_data)  # train_data can be path or object
amp, phase, meta = model.infer(test_data)  # test_data can be path or object
model.save(output_dir)
model = PtychoModel.load(output_dir)
```

**Accepted train/infer inputs:**
- `PtychoDataset` (memmap), `PtychoDataContainerTorch`, `RawData`, `RawDataTorch`, or `Path/str` to NPZ.

**Probe resolution:**
- If `probe` argument provided → use it.
- Else if `probe_path` set → load from path.
- Else → use dataset probe (default).
- Resolved probe source is recorded in model metadata (string path or "dataset").

**Scale resolution:**
- If container path: attach physics scale via `_attach_physics_scale` (uses metadata `nphotons` when present).
- If memmap path: use `physics_scaling_constant` from the memmap dataset.
- Model‑internal `IntensityScalerModule` remains a separate concern.

## Data Flow (Memmap & Container)

**Path → Memmap:**
1. Path provided and `use_memmap=True`.
2. `PtychoDataset` builds/loads `TensorDict` memory map with diffraction, coords, scaling constants, and probe.
3. `TensorDictDataLoader` yields `(tensor_dict, probe, scaling)` batches.
4. Training uses existing `train_cdi_model_torch` flow (memmap dataloaders are already supported).

**Path → Container:**
1. Path provided and `use_memmap=False`.
2. `RawData.from_file` loads NPZ.
3. `_ensure_container` returns `PtychoDataContainerTorch`.
4. `_attach_physics_scale` populates dataset‑level physics scale.

## Error Handling

- Unsupported input type → `TypeError`.
- `probe_path` missing/unreadable → `FileNotFoundError` with clear message.
- NPZ missing required keys → bubble up as `ValueError` with data contract pointer.
- Memmap creation errors → raise with dataset path and memmap target.

## Testing Strategy

- Resolution tests for each accepted input type (path vs object).
- Memmap path uses `PtychoDataset` and yields TensorDict batch contract.
- Probe override precedence: explicit `probe` > `probe_path` > dataset probe.
- Physics scale handling: container path uses `_attach_physics_scale`; memmap path uses `physics_scaling_constant`.
- Legacy bridge called before data loading for path‑based inputs.

## Non‑Goals

- Removing `config_bridge.py` or `config_factory.py`.
- Introducing a new `ptycho_torch.api.*` orchestration layer beyond the façade.
- Changing legacy module contracts or skipping `update_legacy_dict`.

