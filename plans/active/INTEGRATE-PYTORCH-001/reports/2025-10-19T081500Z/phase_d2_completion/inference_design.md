# Phase C1 — PyTorch Inference & Stitching Design (2025-10-19T081500Z)

## Context
- **Initiative:** INTEGRATE-PYTORCH-001 — PyTorch backend integration
- **Phase:** D2.C / Checklist row C1 in `phase_d2_completion.md`
- **Objective:** Provide a concrete design for implementing `_reassemble_cdi_image_torch` so PyTorch workflows can execute `run_cdi_example_torch(..., do_stitching=True)` without raising `NotImplementedError`.
- **Dependencies reviewed:**
  - `specs/ptychodus_api_spec.md` §4.5–4.6 — specifies reconstructor lifecycle expectations (model load → inference → stitching) that PyTorch must honor.
  - `docs/workflows/pytorch.md` §§5–7 — documents training/inference knobs and highlights current stitching gap.
  - `ptycho/workflows/components.py:582-666` — TensorFlow reference implementation (`reassemble_cdi_image`).
  - `ptycho_torch/workflows/components.py:195-610` — PyTorch orchestration helpers (`_ensure_container`, `_build_lightning_dataloaders`, `_reassemble_cdi_image_torch` stub).
  - `ptycho_torch/reassembly.py` + `ptycho_torch/reassembly_beta.py` — existing Torch reassembly utilities (Translation, patch aggregation) that we should reuse instead of re-implementing stitching math.
  - `plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md` — authoritative checklist for Phase D2 tasks.

## Current State Snapshot
- `_reassemble_cdi_image_torch` currently raises `NotImplementedError` (lines 532–580). No inference pipeline runs when `do_stitching=True`.
- `_ensure_container` already normalizes RawData/RawDataTorch/PtychoDataContainerTorch inputs (lines 195–243), providing grouped tensors and probe reference needed for inference.
- `_train_with_lightning` now returns a results payload containing the trained `PtychoPINN_Lightning` module (Attempt #21), which we can reuse for prediction.
- Persistence layer `load_inference_bundle_torch` has parity tests but still defers Lightning reassembly (Phase D4 tasks cover archive loading; Phase D2.C must deliver the live stitching path).
- TensorFlow path (`reassemble_cdi_image`) performs:
  1. Container normalization
  2. Model inference via `nbutils.reconstruct_image`
  3. Optional flips/transpose on offsets
  4. Patch reassembly with `tf_helper.reassemble_position`
  5. Amplitude/phase extraction + results dict
- Tests covering PyTorch stitching do not exist yet; Phase C2 will author them using the design below.

## Target Flow (PyTorch)
```
RawData/Container
   │
   ├─> _ensure_container(...) → PtychoDataContainerTorch
   │        ├─ tensors: tensor_dict['X'], tensor_dict['coords_nominal'], probe
   │        └─ metadata: config snapshot (params.cfg already updated upstream)
   │
   ├─> Build prediction DataLoader (TensorDictDataLoader, batch_size=1 by default)
   │
   ├─> Lightning module.eval(); torch.no_grad()
   │        └─ forward_predict(...) to obtain complex patches
   │
   ├─> Collect patch tensor + coordinate tensors
   │
   ├─> Apply transpose / flip adjustments to offsets if requested
   │
   └─> Reassemble with `ptycho_torch.reassembly.reassemble_multi_channel` (or
        single-channel helper) → complex image canvas
            ├─> amplitude = torch.abs(canvas).cpu().numpy()
            └─> phase = torch.angle(canvas).cpu().numpy()
```

## Detailed Design Decisions
1. **Container Normalization:** Reuse `_ensure_container` to guarantee we operate on `PtychoDataContainerTorch`. For inference we will call `container = _ensure_container(test_data, config)`; this preserves deterministic sampling and honors `config.n_groups`.
2. **DataLoader Construction:** Introduce a lightweight `_build_inference_dataloader(container, batch_size=None)` helper (Phase C2 implementation) that wraps `TensorDictDataLoader` with `shuffle=False` and `batch_size=config.batch_size or 1`. Avoid reusing training dataloader builder to keep inference deterministic.
3. **Lightning Prediction Path:** Invoke `lightning_module = models_dict["lightning_module"] if provided else train_results["models"]["lightning_module"]`. Ensure `lightning_module.eval()` and guard with `torch.no_grad()` to prevent gradient tracking. Use existing `PtychoPINN_Lightning.forward_predict` to handle probe + normalization:
   - Input tensors required: `container.X`, `container.coords_nominal`, `container.probe`.
   - Derive `input_scale_factor` from container metadata (Phase D3 persisted this via params snapshot).
4. **Batch Loop:** Iterate over the inference dataloader; accumulate `batch_output` predictions (complex tensor) and associated coordinate tensors (`coords_nominal`, `coords_relative`, `coord_centers`) into lists. Concatenate at the end to produce `obj_tensor_full` identical to TensorFlow’s shape `(B, C, H, W)`.
5. **Coordinate Adjustments:** Convert offsets to CPU NumPy arrays. Apply:
   - `if transpose: obj_tensor_full = obj_tensor_full.permute(0, 2, 1, 3)` (match TF behavior).
   - `if flip_x/flip_y: coords[:, :, 0/1, :] *= -1`. Include optional `coord_scale` parameter parity once spec clarifies requirement (currently hard-coded 1.0 in TF; keep same default).
6. **Reassembly Helper:** Reuse `ptycho_torch.reassembly.reassemble_multi_channel` if `container.C > 1`, otherwise `reassemble_single_channel`. Inputs needed:
   - `obj_tensor_full` (torch tensor on CPU)
   - `global_offsets` (torch tensor of shape `[B,1,2]` or `[B,C,1,2]`)
   - `data_config` / `model_config` → available via container metadata (Phase D3 ensures config snapshot; for live runs we can construct from `config` dataclass).
   - `center_of_mass` (COM) and `max_offset` can be derived via helpers in `ptycho_torch.helper`. Phase C2 tests will validate numeric parity.
7. **Amplitude/Phase Output:** Convert final stitched tensor to NumPy using `.detach().cpu().numpy()`. Amplitude `np.abs`, phase `np.angle`. Package results dict containing:
   - `"obj_tensor_full"` (pre-stitched torch tensor or np array)
   - `"global_offsets"` (NumPy array)
   - `"recon_amp"`, `"recon_phase"`
   - `"coord_metadata"` (optional) for downstream evaluation parity.
8. **Error Handling:** If inference bundle lacks trained module, raise `RuntimeError` with actionable message referencing Phase D3 exit criteria. Maintain compatibility with `load_inference_bundle_torch` returning `(models_dict, params_dict)`.

## Test Harness Strategy (for Phase C2)
- **Unit-level stitching test:** Add `TestRunCdiExampleTorch.test_stitching_path` that monkeypatches Lightning module’s `predict_step` to return deterministic complex patches. Assertions:
  - `_reassemble_cdi_image_torch` returns amplitude/phase with expected shapes.
  - Global offsets honor flips/transpose.
  - Results dict exposes `"recon_amp"`/`"recon_phase"` keys.
- **Contract test for flips/transpose:** Parametrize over combinations of `flip_x`, `flip_y`, `transpose` verifying coordinate transforms.
- **Integration smoke test:** Extend existing PyTorch integration selector with `do_stitching=True` once implementation lands; use canonical dataset (Run1084...) and verify no `NotImplementedError`.
- **Artifact capture:** Store red log at `plans/active/INTEGRATE-PYTORCH-001/reports/<TS>/phase_d2_completion/pytest_stitch_red.log`; green log once implementation succeeds (`pytest_stitch_green.log`).

## Open Questions / Risks
1. **Coordinate scale (`coord_scale`) parity:** TensorFlow path multiplies offsets by `coord_scale` but defaults to 1.0. Need to confirm whether PyTorch data loaders already emit scaled coordinates; design assumes parity with TF default. Action: Validate during C2 tests.
2. **Complex tensor handling:** Lightning `forward_predict` likely returns `torch.complex64`. Ensure reassembly helpers accept complex dtype; may require splitting real/imag channels. If helpers expect stacked real/imag, implement conversion helper.
3. **Device management:** Training may run on GPU. `_reassemble_cdi_image_torch` must move tensors to CPU before invoking NumPy conversions to avoid device mismatch.
4. **Batch memory footprint:** Stitching requires holding entire dataset in memory. For large datasets we might need to stream; out-of-scope for Phase D2 but note for Phase E parity tests.

## Phase C1 Checklist
| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1.A | Document end-to-end inference flow and component responsibilities | [x] | Sections “Target Flow” and “Detailed Design Decisions” capture the step-by-step plan for `_reassemble_cdi_image_torch`. |
| C1.B | Define test harness expectations for Phase C2 | [x] | See “Test Harness Strategy”; includes selectors, fixtures, and artifact paths. |
| C1.C | Record outstanding questions/risks and decision owners | [x] | Listed under “Open Questions / Risks”; escalate unresolved items in docs/fix_plan attempts if they block implementation. |

## Next Actions
1. **Phase C2** — Author failing pytest coverage per “Test Harness Strategy” (owner: Ralph). Artifact hub: `reports/2025-10-19T081500Z/.../pytest_stitch_red.log`.
2. **Phase C3** — Implement `_reassemble_cdi_image_torch` following this design; ensure complex tensor pipeline matches helper expectations.
3. **Phase C4** — Capture green logs and promote artifacts to parity summary.
