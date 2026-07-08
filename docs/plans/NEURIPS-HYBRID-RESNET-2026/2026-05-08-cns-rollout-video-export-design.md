# CNS Rollout Video Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a model-agnostic PDEBench CNS rollout video exporter that can render side-by-side GIF/MP4 videos for FNO, SRU-Net, FFNO, U-Net, U-NO, and other CNS rows through one shared predictor interface.

**Architecture:** Split the feature into four reusable units: trajectory/data loading, model/checkpoint loading, autoregressive rollout, and rendering. The rollout engine must not branch on model type; model-specific construction is isolated in a loader that returns a callable predictor with the same input/output tensor contract for every row. Future CNS training runs must persist enough model-state metadata for video export; existing runs without checkpoints can only be visualized if a compatible checkpoint is supplied or the row is rerun.

**Tech Stack:** Python, PyTorch, h5py, NumPy, Matplotlib Agg, imageio or Pillow, existing `scripts/studies/pdebench_image128` data/model/normalization/visualization helpers, pytest.

---

## Design Summary

The video exporter should answer this workflow:

```bash
python -m scripts.studies.pdebench_image128.render_cns_rollout_video \
  --run-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history5-comparator-gap-fill/history5-gap-fill-40ep-20260504T214614Z \
  --row-id fno_base \
  --sample-id 0 \
  --start-time 5 \
  --steps 20 \
  --field density \
  --output-root tmp/cns_rollout_videos
```

Expected outputs:

- `tmp/cns_rollout_videos/fno_base_sample000_density_rollout.gif`
- `tmp/cns_rollout_videos/fno_base_sample000_density_rollout.mp4` when MP4 support is available or explicitly requested
- `tmp/cns_rollout_videos/fno_base_sample000_rollout_manifest.json`

Each animation frame should show:

```text
Initial state at t0 | True state at t0+k | Predicted state at t0+k
```

Optional layout mode:

```text
Initial state at t0 | True state at t0+k | Predicted state at t0+k | Absolute error
```

The initial panel is static across frames. The true and predicted panels evolve over the rollout. Color limits are shared across initial, true, and predicted panels for one field and one rollout. Error uses a separate nonnegative scale. Reuse `cfd_cns_field_visual_spec()` and `cfd_cns_shared_scale_bundle()` from `scripts/studies/pdebench_image128/visualization.py`.

## Required Contract

The exporter consumes a CNS row root with:

- `invocation.json` or equivalent run metadata.
- `model_profile_<row_id>.json`.
- `normalization_stats_state.json`.
- `split_manifest.json`.
- `dataset_manifest.json` or `hdf5_metadata.json` containing the CNS HDF5 path and field/axis contract.
- A model state file, preferably `model_state_<row_id>.pt`.

Current CNS run roots often contain one-step `comparison_<row_id>_sample0.npz` and PNG files but do not appear to persist model weights. Those artifacts are insufficient for time-evolved videos because they contain one predicted target frame, not an autoregressive model. The implementation should support old roots only if `--checkpoint-path` is supplied explicitly; otherwise it should fail with a clear message explaining that no model state exists.

## File Structure

- Create `scripts/studies/pdebench_image128/cns_rollout_data.py`
  - Owns HDF5 trajectory reads, split/sample resolution, start-time validation, normalized history construction, and true future extraction.

- Create `scripts/studies/pdebench_image128/cns_rollout_models.py`
  - Owns row/profile metadata loading, model construction through `build_model_from_profile()`, checkpoint loading, and predictor wrapper creation.
  - This file is the only layer that knows about `fno_base`, `spectral_resnet_bottleneck_base`, `author_ffno_cns_base`, `unet_strong`, `neuralop_uno_cns_base`, or future row ids.

- Create `scripts/studies/pdebench_image128/cns_rollout.py`
  - Owns pure autoregressive rollout.
  - Takes normalized initial history and a predictor callable.
  - Returns normalized and denormalized prediction/target arrays plus per-frame metadata.

- Create `scripts/studies/pdebench_image128/cns_rollout_render.py`
  - Owns GIF/MP4 frame rendering.
  - No model or HDF5 code.
  - Reuses existing CNS visual-scale helpers.

- Create `scripts/studies/pdebench_image128/render_cns_rollout_video.py`
  - CLI wrapper that wires data, model, rollout, and rendering.

- Modify `scripts/studies/pdebench_image128/cfd_cns.py`
  - Persist `model_state_<profile_id>.pt` and a minimal `model_state_<profile_id>.json` manifest for future CNS rows.
  - Do not change metrics or training behavior.

- Add `tests/studies/test_pdebench_image128_rollout_video.py`
  - Focused unit tests for data shape, rollout recurrence, model-loader errors, rendering outputs, and CLI dry-run behavior.

- Modify `tests/studies/test_pdebench_image128_runner.py`
  - Add a narrow assertion that future CNS runs write model-state files when a trainable row completes.

## Interfaces

### Data Interface

`cns_rollout_data.py` should expose:

```python
@dataclass(frozen=True)
class CnsTrajectoryWindow:
    trajectory_id: int
    sample_id: int
    start_time: int
    history_len: int
    field_order: tuple[str, ...]
    initial_history_norm: torch.Tensor  # (history_len, C, H, W)
    initial_history_phys: torch.Tensor  # (history_len, C, H, W)
    true_future_norm: torch.Tensor      # (steps, C, H, W)
    true_future_phys: torch.Tensor      # (steps, C, H, W)
    dt: float | None
```

Resolution rules:

- `--sample-id` indexes into the selected split, default `test`.
- The resolved trajectory id must be recorded in the output manifest.
- `--start-time` is the first target time after the history window.
- Validity condition: `start_time >= history_len` and `start_time + steps <= time_steps`.

### Model Interface

`cns_rollout_models.py` should expose:

```python
@dataclass(frozen=True)
class CnsPredictor:
    row_id: str
    history_len: int
    field_order: tuple[str, ...]
    device: torch.device
    model: torch.nn.Module

    def __call__(self, history_norm: torch.Tensor) -> torch.Tensor:
        ...
```

Input contract:

- `history_norm` shape: `(history_len, C, H, W)` or `(1, history_len * C, H, W)` after internal flattening.
- Output shape: `(C, H, W)`.
- The wrapper handles flattening from temporal history to the existing model input channel layout.

Loader signature:

```python
def load_cns_predictor(
    *,
    run_root: Path,
    row_id: str,
    checkpoint_path: Path | None = None,
    device: str = "cpu",
) -> CnsPredictor:
    ...
```

Checkpoint resolution:

1. Use `checkpoint_path` if supplied.
2. Else look for `run_root / f"model_state_{row_id}.pt"`.
3. Else fail with:
   `MissingCnsCheckpointError: CNS rollout video export requires model_state_<row_id>.pt or --checkpoint-path; one-step comparison NPZ files cannot produce autoregressive videos.`

### Rollout Interface

`cns_rollout.py` should expose:

```python
@dataclass(frozen=True)
class CnsRolloutResult:
    initial_state_phys: np.ndarray  # (C, H, W), last observed history state
    true_phys: np.ndarray           # (steps, C, H, W)
    pred_phys: np.ndarray           # (steps, C, H, W)
    abs_error_phys: np.ndarray      # (steps, C, H, W)
    field_order: tuple[str, ...]
    frame_time_indices: tuple[int, ...]

def autoregressive_rollout(
    *,
    window: CnsTrajectoryWindow,
    predictor: CnsPredictor,
    state_stats: dict[str, Any],
) -> CnsRolloutResult:
    ...
```

Implementation detail:

- Rollout uses normalized tensors for model input/output.
- After each predicted next state, append the normalized prediction to the history and drop the oldest frame.
- Denormalize once for returned render arrays.

### Render Interface

`cns_rollout_render.py` should expose:

```python
def render_field_rollout_gif(
    *,
    result: CnsRolloutResult,
    field: str,
    output_path: Path,
    fps: float = 4.0,
    include_error: bool = False,
) -> Path:
    ...
```

Also support:

```python
def render_all_field_rollouts(...)
```

Rendering rules:

- Use one shared value scale for initial/true/pred for the selected field across all rollout frames.
- Use separate error scale for absolute error.
- Use field-specific colormaps from `cfd_cns_field_visual_spec`.
- Default GIF dimensions should be readable in a paper supplement or slide deck: 3 panels by one field, not all four fields tiled unless explicitly requested.

## Implementation Tasks

### Task 1: Add Data Loading Unit

**Files:**
- Create: `scripts/studies/pdebench_image128/cns_rollout_data.py`
- Test: `tests/studies/test_pdebench_image128_rollout_video.py`

- [ ] **Step 1: Write tests for trajectory window extraction**

Create tests with a small temporary HDF5 containing `density`, `Vx`, `Vy`, `pressure` datasets shaped `(N,T,H,W)` and coordinate datasets. Assert:

```python
window.initial_history_phys.shape == (history_len, 4, H, W)
window.true_future_phys.shape == (steps, 4, H, W)
window.initial_history_norm.shape == (history_len, 4, H, W)
window.trajectory_id == expected_split_trajectory_id
window.start_time == start_time
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_rollout_video.py -k trajectory_window
```

Expected: fails because `cns_rollout_data.py` does not exist.

- [ ] **Step 3: Implement data extraction**

Use existing helpers:

- `CFD_CNS_FIELD_ORDER`
- `read_dynamic_state_channel_first`
- `inspect_cfd_cns_hdf5`
- `normalize_batch`

Read physical frames, stack to `(T,C,H,W)`, normalize framewise, and slice the requested window.

- [ ] **Step 4: Verify**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_rollout_video.py -k trajectory_window
```

Expected: passes.

### Task 2: Persist CNS Model State For Future Runs

**Files:**
- Modify: `scripts/studies/pdebench_image128/cfd_cns.py`
- Modify: `tests/studies/test_pdebench_image128_runner.py`

- [ ] **Step 1: Write runner test**

Add or extend a tiny CNS runner test that completes one trainable row and asserts:

```python
assert (run_root / "model_state_unet_tiny_smoke.pt").exists()
manifest = json.loads((run_root / "model_state_unet_tiny_smoke.json").read_text())
assert manifest["profile_id"] == "unet_tiny_smoke"
assert manifest["state_dict_format"] == "torch_state_dict"
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_runner.py -k "model_state and cfd"
```

Expected: fails because CNS rows do not persist model state.

- [ ] **Step 3: Save state dict after training**

In `_run_profile()`, after evaluation and before writing final metrics on rank zero:

```python
state_path = output_root / f"model_state_{profile_id}.pt"
torch.save(raw_model.state_dict(), state_path)
_write_json(
    output_root / f"model_state_{profile_id}.json",
    {
        "schema_version": "pdebench_cns_model_state_manifest_v1",
        "profile_id": profile_id,
        "state_dict_format": "torch_state_dict",
        "state_path": str(state_path),
        "model_profile_path": str(output_root / f"model_profile_{profile_id}.json"),
        "normalization_stats_path": str(output_root / "normalization_stats_state.json"),
    },
)
```

Do not save DDP wrapper state. Save `raw_model.state_dict()`.

- [ ] **Step 4: Verify**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_runner.py -k "model_state or unet_tiny_smoke"
```

Expected: passes.

### Task 3: Add Model-Agnostic Predictor Loader

**Files:**
- Create: `scripts/studies/pdebench_image128/cns_rollout_models.py`
- Test: `tests/studies/test_pdebench_image128_rollout_video.py`

- [ ] **Step 1: Write loader tests**

Use a tiny model profile/run-root fixture. Assert:

- Missing checkpoint raises `MissingCnsCheckpointError` with a message saying one-step comparison NPZ is insufficient.
- A saved state dict loads and returns a `CnsPredictor`.
- `predictor(history)` accepts `(history_len,C,H,W)` and returns `(C,H,W)`.

- [ ] **Step 2: Run failing tests**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_rollout_video.py -k "predictor or checkpoint"
```

Expected: fails because loader does not exist.

- [ ] **Step 3: Implement loader**

Implementation requirements:

- Read `model_profile_<row_id>.json`.
- Reconstruct a `ModelProfile` or call `get_model_profile(row_id)` when the row id is registered.
- Infer `history_len`, `field_order`, `input_channels`, `target_channels`, and `spatial_shape` from `hdf5_metadata.json`, `dataset_manifest.json`, or `invocation.json`.
- Call `build_model_from_profile(...)`.
- Load state dict with `map_location=device`.
- Set `model.eval()`.
- Return `CnsPredictor`.

- [ ] **Step 4: Verify**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_rollout_video.py -k "predictor or checkpoint"
```

Expected: passes.

### Task 4: Add Pure Autoregressive Rollout Unit

**Files:**
- Create: `scripts/studies/pdebench_image128/cns_rollout.py`
- Test: `tests/studies/test_pdebench_image128_rollout_video.py`

- [ ] **Step 1: Write recurrence tests**

Use a fake predictor that returns `last_history_frame + 1` in normalized space. Assert:

```python
pred[0] == initial_last + 1
pred[1] == initial_last + 2
pred[2] == initial_last + 3
```

Also assert the true frames are not fed into later prediction steps.

- [ ] **Step 2: Run failing tests**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_rollout_video.py -k rollout
```

Expected: fails because rollout unit does not exist.

- [ ] **Step 3: Implement rollout**

Use normalized tensor recurrence:

```python
history = window.initial_history_norm.clone()
predictions = []
for _ in range(steps):
    pred_next = predictor(history)
    predictions.append(pred_next.detach().cpu())
    history = torch.cat([history[1:], pred_next.detach().cpu().unsqueeze(0)], dim=0)
```

Then denormalize predictions using `denormalize_batch()`.

- [ ] **Step 4: Verify**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_rollout_video.py -k rollout
```

Expected: passes.

### Task 5: Add GIF/MP4 Renderer

**Files:**
- Create: `scripts/studies/pdebench_image128/cns_rollout_render.py`
- Test: `tests/studies/test_pdebench_image128_rollout_video.py`

- [ ] **Step 1: Write renderer tests**

Construct a small `CnsRolloutResult` with two frames. Assert:

- GIF path exists after rendering.
- Manifest or returned metadata records field, frame count, panel layout, value scale, and error scale.
- Rendering uses shared value scale across initial/true/pred arrays.

- [ ] **Step 2: Run failing tests**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_rollout_video.py -k render
```

Expected: fails because renderer does not exist.

- [ ] **Step 3: Implement renderer**

Use Matplotlib Agg to draw each frame into an RGB array. Use `imageio.v2.mimsave()` if available; fallback to Pillow if needed. Keep MP4 optional:

- GIF is required.
- MP4 is emitted only when `imageio-ffmpeg` or system ffmpeg is available, or when `--format mp4` is requested and the dependency check passes.

- [ ] **Step 4: Verify**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_rollout_video.py -k render
```

Expected: passes.

### Task 6: Add CLI Wrapper

**Files:**
- Create: `scripts/studies/pdebench_image128/render_cns_rollout_video.py`
- Test: `tests/studies/test_pdebench_image128_rollout_video.py`

- [ ] **Step 1: Write CLI tests**

Test `--dry-run` against a tiny fixture root. Assert it writes or prints a manifest with:

```json
{
  "row_id": "fno_base",
  "sample_id": 0,
  "trajectory_id": 123,
  "start_time": 5,
  "steps": 20,
  "field": "density",
  "requires_checkpoint": true
}
```

Test missing checkpoint exits nonzero with the controlled error.

- [ ] **Step 2: Run failing tests**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_rollout_video.py -k cli
```

Expected: fails because CLI does not exist.

- [ ] **Step 3: Implement CLI**

Arguments:

```text
--run-root PATH
--row-id ROW
--checkpoint-path PATH optional
--data-file PATH optional override
--split test|val|train default test
--sample-id INT default 0
--start-time INT default history_len
--steps INT default 20
--field density|Vx|Vy|pressure|all default density
--include-error
--fps FLOAT default 4
--format gif|mp4|both default gif
--device cpu|cuda default cpu
--output-root PATH required
--dry-run
```

The CLI should write `rollout_manifest.json` with all resolved paths and contracts.

- [ ] **Step 4: Verify**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_rollout_video.py -k cli
python -m scripts.studies.pdebench_image128.render_cns_rollout_video --help
```

Expected: tests pass and help prints.

### Task 7: Add One End-To-End Smoke

**Files:**
- Test: `tests/studies/test_pdebench_image128_rollout_video.py`

- [ ] **Step 1: Write small end-to-end test**

Use a tiny synthetic HDF5 and a tiny row model that saves a state dict. Run CLI with `--steps 3 --field density --format gif`. Assert:

- GIF exists.
- Manifest exists.
- Manifest records the model row and trajectory id.
- GIF frame count equals `3`.

- [ ] **Step 2: Run smoke**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_rollout_video.py
```

Expected: all rollout-video tests pass.

### Task 8: Optional Real-Artifact Smoke

**Files:**
- No code changes unless a real checkpoint row is available.

- [ ] **Step 1: Locate a row root with checkpoint**

Run:

```bash
find .artifacts/NEURIPS-HYBRID-RESNET-2026 .artifacts/work/NEURIPS-HYBRID-RESNET-2026 \
  -name 'model_state_fno_base.pt' -o -name 'model_state_spectral_resnet_bottleneck_base.pt' -o -name 'model_state_author_ffno_cns_base.pt'
```

Expected: if no result, skip real-artifact smoke and record that older CNS roots need rerun or explicit checkpoint supply.

- [ ] **Step 2: Run real video export if checkpoint exists**

Example:

```bash
python -m scripts.studies.pdebench_image128.render_cns_rollout_video \
  --run-root <run-root> \
  --row-id fno_base \
  --sample-id 0 \
  --start-time 5 \
  --steps 20 \
  --field density \
  --output-root tmp/cns_rollout_videos \
  --format gif
```

Expected: a readable GIF and manifest under `tmp/cns_rollout_videos`.

## Verification Commands

Run these before declaring implementation complete:

```bash
python -m compileall -q scripts/studies/pdebench_image128
pytest --collect-only -q tests/studies/test_pdebench_image128_rollout_video.py
pytest -q tests/studies/test_pdebench_image128_rollout_video.py
pytest -q tests/studies/test_pdebench_image128_runner.py -k "model_state or unet_tiny_smoke"
python -m scripts.studies.pdebench_image128.render_cns_rollout_video --help
```

If a real checkpoint exists, also run one real export and inspect:

```bash
python -m scripts.studies.pdebench_image128.render_cns_rollout_video \
  --run-root <run-root> \
  --row-id <row-id> \
  --sample-id 0 \
  --steps 8 \
  --field density \
  --output-root tmp/cns_rollout_videos \
  --format gif
```

## Non-Goals

- Do not retrain CNS rows inside the video exporter.
- Do not make videos from one-step `comparison_*.npz` artifacts and present them as rollouts.
- Do not add manuscript figures in this implementation. This is an artifact-generation capability first.
- Do not change CNS metrics, row ranking, training objective, or table-generation behavior.
- Do not make model-specific rollout scripts for each architecture.

## Known Follow-Up

After this lands, any CNS row intended for video export should be rerun or recovered with checkpoint persistence. For existing paper rows where no checkpoint was saved, a narrow checkpoint-producing rerun is required before FNO/SRU-Net/FFNO rollout videos can be generated from that row.
