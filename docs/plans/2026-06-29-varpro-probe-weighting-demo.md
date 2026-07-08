# VarPro Probe Weighting Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic numerical and visual demonstration comparing reconstruction quality with and without VarPro intensity scaling and probe-weighted patch stitching.

**Architecture:** Add an explicit inference-time `varpro_scaling` switch beside the existing `patch_weighting` switch, then factor the final VarPro canvas scaling into a testable helper. Use the existing `VectorizedWeightedAccumulator` and `VarProScaler` primitives for a synthetic demo that emits four comparable reconstructions: uniform/no-VarPro, uniform/VarPro, probe/no-VarPro, and probe/VarPro.

**Tech Stack:** Python, PyTorch, NumPy, Matplotlib, pytest, `ptycho_torch.config_params`, `ptycho_torch.config_factory`, `ptycho_torch.reassembly`.

---

## Context

The current code already supports probe-weighted stitching through `InferenceConfig.patch_weighting`:

- `ptycho_torch/config_params.py` defines `InferenceConfig.patch_weighting`.
- `ptycho_torch/config_factory.py` forwards `patch_weighting` into `PTInferenceConfig`.
- `ptycho_torch/reassembly.py::reconstruct_image_barycentric()` reads `patch_weighting` and passes `uniform_weighting` into `VectorizedWeightedAccumulator.accumulate_batch()`.

The current VarPro path is always enabled inside `reconstruct_image_barycentric()`:

- `VarProScaler` is instantiated unconditionally.
- `scaler.accumulate_batch_from_basis(...)` is called during every inference batch.
- `scaler.solve_lbfgs()` is called after stitching.
- `scaled_canvas = (s1 * texture_canvas.real) + 1j * (s2 * texture_canvas.imag)` is always returned.

That means the repo can compare `patch_weighting="uniform"` vs `"probe"` today, but it cannot produce a clean "without VarPro" control through `InferenceConfig`.

## Documents Read

- `docs/index.md`
- `docs/findings.md`
- `docs/workflows/pytorch.md`
- `docs/TESTING_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `ptycho_torch/config_params.py`
- `ptycho_torch/config_factory.py`
- `ptycho_torch/reassembly.py`
- Existing `tests/torch` config and reassembly selectors discovered with `rg`

## File Structure

- Modify `ptycho_torch/config_params.py`
  - Add `InferenceConfig.varpro_scaling: bool = True`.
  - Keep the default behavior unchanged.

- Modify `ptycho_torch/config_factory.py`
  - Forward `varpro_scaling` from overrides into both training and inference `PTInferenceConfig` construction sites.

- Modify `ptycho_torch/reassembly.py`
  - Add a small helper that applies or bypasses VarPro scaling to a stitched complex canvas.
  - Validate `patch_weighting` values before running accumulation.
  - Read `inference_config.varpro_scaling`, defaulting to `True` for backward compatibility.
  - Preserve diagnostics shape and existing return contract.

- Create `tests/torch/test_varpro_probe_weighted_reassembly.py`
  - Cover VarPro scale recovery on a synthetic basis.
  - Cover probe weighting improving an overlap seam when a patch edge is intentionally corrupted.
  - Cover the new no-VarPro scaling helper behavior.

- Modify `tests/torch/test_config_factory.py`
  - Cover training and inference payload propagation of `varpro_scaling`.

- Create `scripts/studies/demo_varpro_probe_weighted_reassembly.py`
  - Generate deterministic synthetic complex object patches.
  - Add controlled real/imag scale distortion and edge corruption.
  - Reassemble four variants.
  - Save PNG panels plus metrics JSON under `.artifacts/varpro_probe_weighting_demo/`.

## Demonstration Contract

The demo must produce these files:

- `.artifacts/varpro_probe_weighting_demo/metrics.json`
- `.artifacts/varpro_probe_weighting_demo/reconstruction_grid.png`
- `.artifacts/varpro_probe_weighting_demo/error_grid.png`
- `.artifacts/varpro_probe_weighting_demo/probe_weight_map.png`

The metrics JSON must contain one record for each variant:

```json
{
  "uniform_no_varpro": {"complex_mae": 0.0, "amp_mae": 0.0, "phase_mae": 0.0, "seam_mae": 0.0},
  "uniform_varpro": {"complex_mae": 0.0, "amp_mae": 0.0, "phase_mae": 0.0, "seam_mae": 0.0},
  "probe_no_varpro": {"complex_mae": 0.0, "amp_mae": 0.0, "phase_mae": 0.0, "seam_mae": 0.0},
  "probe_varpro": {"complex_mae": 0.0, "amp_mae": 0.0, "phase_mae": 0.0, "seam_mae": 0.0}
}
```

Acceptance expectation:

- `probe_no_varpro.seam_mae < uniform_no_varpro.seam_mae`
- `uniform_varpro.complex_mae < uniform_no_varpro.complex_mae`
- `probe_varpro.complex_mae < probe_no_varpro.complex_mae`
- `probe_varpro.complex_mae` is the best or tied-best of the four variants within a small tolerance.

## Tasks

### Task 1: Add the VarPro Inference Config Surface

**Files:**
- Modify: `ptycho_torch/config_params.py`
- Modify: `ptycho_torch/config_factory.py`
- Modify: `tests/torch/test_config_factory.py`

- [ ] **Step 1: Write failing factory tests**

Add tests near the existing `PTInferenceConfig` assertions:

```python
def test_create_training_payload_propagates_varpro_scaling(tmp_path):
    train_file = tmp_path / "train.npz"
    train_file.write_bytes(b"placeholder")

    payload = create_training_payload(
        train_data_file=train_file,
        output_dir=tmp_path / "out",
        overrides={"n_groups": 8, "varpro_scaling": False},
    )

    assert payload.pt_inference_config.varpro_scaling is False


def test_create_inference_payload_propagates_varpro_scaling(tmp_path):
    test_file = tmp_path / "test.npz"
    model_file = tmp_path / "model.ckpt"
    test_file.write_bytes(b"placeholder")
    model_file.write_bytes(b"placeholder")

    payload = create_inference_payload(
        model_path=model_file,
        test_data_file=test_file,
        output_dir=tmp_path / "out",
        overrides={"varpro_scaling": False},
    )

    assert payload.pt_inference_config.varpro_scaling is False
```

- [ ] **Step 2: Run the focused failing tests**

Run:

```bash
python -m pytest -q tests/torch/test_config_factory.py -k varpro_scaling
```

Expected: fail because `InferenceConfig` does not yet expose `varpro_scaling`.

- [ ] **Step 3: Implement the config field**

In `ptycho_torch/config_params.py`, add:

```python
varpro_scaling: bool = True
```

Place it immediately after `patch_weighting` so stitching controls stay together.

- [ ] **Step 4: Forward factory overrides**

In both `PTInferenceConfig(...)` construction sites in `ptycho_torch/config_factory.py`, add:

```python
varpro_scaling=overrides.get('varpro_scaling', True),
```

- [ ] **Step 5: Verify config tests pass**

Run:

```bash
python -m pytest -q tests/torch/test_config_factory.py -k "varpro_scaling or patch_weighting"
```

Expected: pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add ptycho_torch/config_params.py ptycho_torch/config_factory.py tests/torch/test_config_factory.py
git commit -m "feat: expose varpro inference scaling switch"
```

### Task 2: Gate VarPro Scaling in Reassembly

**Files:**
- Modify: `ptycho_torch/reassembly.py`
- Create: `tests/torch/test_varpro_probe_weighted_reassembly.py`

- [ ] **Step 1: Write helper tests**

Create `tests/torch/test_varpro_probe_weighted_reassembly.py` with:

```python
import torch

from ptycho_torch.reassembly import apply_varpro_canvas_scaling


class DummyScaler:
    def __init__(self):
        self.calls = 0

    def solve_lbfgs(self, *args, **kwargs):
        self.calls += 1
        return torch.tensor(2.0), torch.tensor(0.5)


def test_apply_varpro_canvas_scaling_bypasses_solver_when_disabled():
    canvas = torch.tensor([[1 + 4j, 2 + 6j]], dtype=torch.complex64)
    scaler = DummyScaler()

    scaled, s1, s2 = apply_varpro_canvas_scaling(canvas, scaler, enabled=False, verbose=False)

    assert scaler.calls == 0
    torch.testing.assert_close(scaled, canvas)
    torch.testing.assert_close(s1, torch.tensor(1.0))
    torch.testing.assert_close(s2, torch.tensor(1.0))


def test_apply_varpro_canvas_scaling_uses_solver_when_enabled():
    canvas = torch.tensor([[1 + 4j, 2 + 6j]], dtype=torch.complex64)
    scaler = DummyScaler()

    scaled, s1, s2 = apply_varpro_canvas_scaling(canvas, scaler, enabled=True, verbose=False)

    assert scaler.calls == 1
    torch.testing.assert_close(scaled, torch.tensor([[2 + 2j, 4 + 3j]], dtype=torch.complex64))
    torch.testing.assert_close(s1, torch.tensor(2.0))
    torch.testing.assert_close(s2, torch.tensor(0.5))
```

- [ ] **Step 2: Run the focused failing tests**

Run:

```bash
python -m pytest -q tests/torch/test_varpro_probe_weighted_reassembly.py -k apply_varpro
```

Expected: fail because `apply_varpro_canvas_scaling` does not exist.

- [ ] **Step 3: Implement the helper**

In `ptycho_torch/reassembly.py`, add near `VarProScaler` or immediately before `reconstruct_image_barycentric()`:

```python
def apply_varpro_canvas_scaling(
    texture_canvas: torch.Tensor,
    scaler: VarProScaler,
    *,
    enabled: bool = True,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not enabled:
        one = torch.tensor(1.0, device=texture_canvas.device, dtype=torch.float32)
        return texture_canvas, one, one

    s1, s2 = scaler.solve_lbfgs(verbose=verbose)
    scaled_canvas = torch.complex(s1 * texture_canvas.real, s2 * texture_canvas.imag)
    return scaled_canvas, s1, s2
```

If Python version compatibility rejects `tuple[...]`, use `Tuple[...]` from `typing`, matching the file style.

- [ ] **Step 4: Wire the helper into reconstruction**

In `reconstruct_image_barycentric()`:

```python
patch_weighting = getattr(inference_config, 'patch_weighting', 'probe')
if patch_weighting not in {'uniform', 'probe'}:
    raise ValueError("patch_weighting must be 'uniform' or 'probe'")
uniform_weighting = (patch_weighting == 'uniform')
varpro_scaling = getattr(inference_config, 'varpro_scaling', True)
```

Replace the unconditional solve/apply block with:

```python
scaler_solve_time_start = time.time()
scaled_canvas, s1, s2 = apply_varpro_canvas_scaling(
    texture_canvas,
    scaler,
    enabled=varpro_scaling,
    verbose=verbose,
)
scaler_solve_time_end = time.time() - scaler_solve_time_start
```

Only print solved scalars when `verbose` is true or keep existing print behavior if downstream logs depend on it. Prefer quiet behavior for tests and demo:

```python
if verbose:
    print(f"Scalars solved: S1 = {s1}, S2 = {s2}")
```

- [ ] **Step 5: Verify helper tests pass**

Run:

```bash
python -m pytest -q tests/torch/test_varpro_probe_weighted_reassembly.py -k apply_varpro
```

Expected: pass.

- [ ] **Step 6: Run existing reassembly regression tests**

Run:

```bash
python -m pytest -q \
  tests/torch/test_inference_reassembly_parity.py \
  tests/torch/test_inference_reassembly_aggregation.py \
  tests/torch/test_reassembly_multi_patch_parity.py \
  tests/torch/test_reassembly_sign_parity.py
```

Expected: pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add ptycho_torch/reassembly.py tests/torch/test_varpro_probe_weighted_reassembly.py
git commit -m "feat: gate varpro scaling during reassembly"
```

### Task 3: Add Numerical Tests for VarPro and Probe Weighting

**Files:**
- Modify: `tests/torch/test_varpro_probe_weighted_reassembly.py`

- [ ] **Step 1: Add a VarPro scale recovery test**

Add a deterministic test that directly exercises `VarProScaler.accumulate_batch_from_basis(...)`:

```python
from ptycho_torch.reassembly import VarProScaler


def test_varpro_scaler_recovers_known_channel_scales():
    device = torch.device("cpu")
    scaler = VarProScaler(device)

    y = torch.linspace(-1.0, 1.0, 16, device=device)
    x = torch.linspace(-1.0, 1.0, 16, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    X1 = (1.0 + xx.square()).unsqueeze(0)
    X2 = (0.8 + yy.square()).unsqueeze(0)
    X3 = torch.zeros_like(X1)
    expected_s1 = torch.tensor(1.7)
    expected_s2 = torch.tensor(0.6)
    I_raw = expected_s1.square() * X1 + expected_s2.square() * X2

    scaler.accumulate_batch_from_basis(I_raw, X1, X2, X3)
    s1, s2 = scaler.solve_lbfgs(verbose=False)

    torch.testing.assert_close(s1, expected_s1, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(s2, expected_s2, rtol=2e-2, atol=2e-2)
```

- [ ] **Step 2: Add a probe weighting seam test**

Add a direct accumulator test:

```python
from ptycho_torch.reassembly import VectorizedWeightedAccumulator


def _assemble_two_patch_overlap(*, uniform_weighting: bool):
    device = torch.device("cpu")
    patch_size = 8
    canvas_shape = (12, 16)
    canvas = torch.zeros(canvas_shape, dtype=torch.complex64, device=device)
    weights = torch.zeros(canvas_shape, dtype=torch.float32, device=device)
    accumulator = VectorizedWeightedAccumulator(canvas_shape, device)

    truth_left = torch.ones((patch_size, patch_size), dtype=torch.complex64, device=device)
    truth_right = torch.ones((patch_size, patch_size), dtype=torch.complex64, device=device)
    truth_left[:, -2:] = 5 + 0j

    patches = torch.stack([truth_left, truth_right])
    positions = torch.tensor([[6.0, 6.0], [10.0, 6.0]], dtype=torch.float32, device=device)

    edge_downweight = torch.ones((patch_size, patch_size), dtype=torch.float32, device=device)
    edge_downweight[:, -2:] = 0.05

    accumulator.accumulate_batch(
        canvas,
        weights,
        patches,
        positions,
        edge_downweight,
        patch_size=patch_size,
        uniform_weighting=uniform_weighting,
    )
    return canvas / (weights + 1e-12)


def test_probe_weighting_reduces_corrupted_overlap_seam_error():
    uniform = _assemble_two_patch_overlap(uniform_weighting=True)
    probe = _assemble_two_patch_overlap(uniform_weighting=False)

    seam = (slice(2, 10), slice(6, 8))
    expected = torch.ones((8, 2), dtype=torch.complex64)

    uniform_mae = torch.mean(torch.abs(uniform[seam] - expected))
    probe_mae = torch.mean(torch.abs(probe[seam] - expected))

    assert probe_mae < uniform_mae
```

If this exact seam window is too sensitive to the accumulator bounds, adjust the fixture positions but keep the assertion: probe weighting must reduce the corrupted overlap error.

- [ ] **Step 3: Run the new numerical test module**

Run:

```bash
python -m pytest -q tests/torch/test_varpro_probe_weighted_reassembly.py
```

Expected: pass.

- [ ] **Step 4: Commit**

Run:

```bash
git add tests/torch/test_varpro_probe_weighted_reassembly.py
git commit -m "test: prove varpro and probe weighted reassembly primitives"
```

### Task 4: Build the Visual Demo Script

**Files:**
- Create: `scripts/studies/demo_varpro_probe_weighted_reassembly.py`

- [ ] **Step 1: Write the script skeleton**

Create a CLI with:

```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=".artifacts/varpro_probe_weighting_demo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()
```

The script must create `output_root` and write `invocation.json` with the command, seed, and git commit when available.

- [ ] **Step 2: Generate deterministic synthetic object and probe**

Use a small synthetic complex object:

```python
size = 64
y = np.linspace(-1.0, 1.0, size)
x = np.linspace(-1.0, 1.0, size)
yy, xx = np.meshgrid(y, x, indexing="ij")
amplitude = 1.0 + 0.25 * np.exp(-((xx + 0.25) ** 2 + (yy - 0.1) ** 2) / 0.08)
phase = 0.6 * np.sin(2 * np.pi * xx) + 0.35 * np.cos(2 * np.pi * yy)
truth = amplitude * np.exp(1j * phase)
```

Use a nonuniform probe weight map:

```python
patch_size = 24
grid = np.linspace(-1.0, 1.0, patch_size)
py, px = np.meshgrid(grid, grid, indexing="ij")
probe_weight = np.exp(-3.5 * (px**2 + py**2)).astype(np.float32)
probe_weight /= probe_weight.max()
```

- [ ] **Step 3: Extract overlapping patches and inject controlled defects**

Use overlapping top-left positions such as `(0, 0), (12, 0), (24, 0), ...` until the `64x64` object is covered.

For every patch:

```python
patch = truth[y0:y0 + patch_size, x0:x0 + patch_size]
patch_pred = (patch.real / 1.6) + 1j * (patch.imag / 0.65)
patch_pred[:, -4:] += 0.30 + 0.20j
patch_pred[-4:, :] += 0.15 - 0.10j
```

This creates two independent problems:

- VarPro should recover the real/imag scale distortion.
- Probe weighting should reduce edge corruption in overlaps.

- [ ] **Step 4: Reassemble the four variants with production primitives**

Use `VectorizedWeightedAccumulator` for both uniform and probe modes:

```python
accumulator.accumulate_batch(
    canvas,
    canvas_weights,
    patches_tensor,
    positions_tensor,
    probe_weight_tensor,
    patch_size=patch_size,
    uniform_weighting=(patch_weighting == "uniform"),
)
texture_canvas = canvas / (canvas_weights + 1e-12)
```

For VarPro-enabled variants, use `apply_varpro_canvas_scaling(...)` with a `VarProScaler` whose basis is accumulated from the known synthetic distortion. Keep this direct and deterministic; the purpose is to demonstrate the inference-time scale correction, not to train a model.

- [ ] **Step 5: Compute metrics**

For each variant, compute:

```python
complex_mae = np.mean(np.abs(recon - truth))
amp_mae = np.mean(np.abs(np.abs(recon) - np.abs(truth)))
phase_mae = np.mean(np.abs(np.angle(recon * np.conj(truth))))
seam_mae = np.mean(np.abs(recon[seam_mask] - truth[seam_mask]))
```

Save the metrics as `metrics.json` with sorted keys and indentation.

- [ ] **Step 6: Save visual artifacts**

Use Matplotlib to write:

- `reconstruction_grid.png`: amplitude and phase for all four variants plus truth.
- `error_grid.png`: absolute complex error maps for all four variants.
- `probe_weight_map.png`: the probe weighting map used for stitching.

Use fixed `vmin`/`vmax` for comparable panels.

- [ ] **Step 7: Run the demo**

Run:

```bash
python scripts/studies/demo_varpro_probe_weighted_reassembly.py \
  --output-root .artifacts/varpro_probe_weighting_demo \
  --seed 0
```

Expected:

- The four expected artifact files exist.
- `metrics.json` satisfies the demonstration contract.

- [ ] **Step 8: Commit**

Run:

```bash
git add scripts/studies/demo_varpro_probe_weighted_reassembly.py
git commit -m "feat: add varpro probe weighting visual demo"
```

### Task 5: Add Demo Regression Coverage

**Files:**
- Create or modify: `tests/torch/test_varpro_probe_weighted_reassembly.py`

- [ ] **Step 1: Add a lightweight CLI smoke test**

Add a test that runs the demo into `tmp_path`:

```python
import json
import subprocess


def test_varpro_probe_weighting_demo_outputs_expected_artifacts(tmp_path):
    output_root = tmp_path / "demo"
    subprocess.run(
        [
            "python",
            "scripts/studies/demo_varpro_probe_weighted_reassembly.py",
            "--output-root",
            str(output_root),
            "--seed",
            "0",
        ],
        check=True,
    )

    metrics = json.loads((output_root / "metrics.json").read_text())

    assert (output_root / "reconstruction_grid.png").exists()
    assert (output_root / "error_grid.png").exists()
    assert (output_root / "probe_weight_map.png").exists()
    assert metrics["probe_no_varpro"]["seam_mae"] < metrics["uniform_no_varpro"]["seam_mae"]
    assert metrics["uniform_varpro"]["complex_mae"] < metrics["uniform_no_varpro"]["complex_mae"]
    assert metrics["probe_varpro"]["complex_mae"] < metrics["probe_no_varpro"]["complex_mae"]
```

- [ ] **Step 2: Run the smoke test**

Run:

```bash
python -m pytest -q tests/torch/test_varpro_probe_weighted_reassembly.py -k demo_outputs
```

Expected: pass in less than 10 seconds on CPU.

- [ ] **Step 3: Run the full focused module**

Run:

```bash
python -m pytest -q tests/torch/test_varpro_probe_weighted_reassembly.py
```

Expected: pass.

- [ ] **Step 4: Commit**

Run:

```bash
git add tests/torch/test_varpro_probe_weighted_reassembly.py
git commit -m "test: cover varpro probe weighting demo artifacts"
```

### Task 6: Final Verification and Evidence

**Files:**
- No expected code edits unless verification reveals a defect.

- [ ] **Step 1: Run compile check**

Run:

```bash
python -m compileall -q ptycho_torch scripts/studies
```

Expected: exit code 0.

- [ ] **Step 2: Run focused tests**

Run:

```bash
python -m pytest -q \
  tests/torch/test_varpro_probe_weighted_reassembly.py \
  tests/torch/test_config_factory.py -k "varpro_scaling or patch_weighting"
```

Expected: pass.

- [ ] **Step 3: Run reassembly regressions**

Run:

```bash
python -m pytest -q \
  tests/torch/test_inference_reassembly_parity.py \
  tests/torch/test_inference_reassembly_aggregation.py \
  tests/torch/test_reassembly_multi_patch_parity.py \
  tests/torch/test_reassembly_sign_parity.py
```

Expected: pass.

- [ ] **Step 4: Generate final demo artifacts**

Run:

```bash
python scripts/studies/demo_varpro_probe_weighted_reassembly.py \
  --output-root .artifacts/varpro_probe_weighting_demo \
  --seed 0
```

Expected:

- `.artifacts/varpro_probe_weighting_demo/metrics.json` exists.
- `.artifacts/varpro_probe_weighting_demo/reconstruction_grid.png` exists.
- `.artifacts/varpro_probe_weighting_demo/error_grid.png` exists.
- `.artifacts/varpro_probe_weighting_demo/probe_weight_map.png` exists.
- The combined `probe_varpro` variant is visibly cleaner in the overlap/error panels and numerically best or tied-best by `complex_mae`.

- [ ] **Step 5: Record evidence in this plan**

Append a short `## Verification Log` section with:

- command
- exit code
- short pass/fail summary
- metrics file path
- artifact image paths

- [ ] **Step 6: Commit the verification-log update**

Run:

```bash
git add docs/plans/2026-06-29-varpro-probe-weighting-demo.md
git commit -m "docs: record varpro probe weighting demo evidence"
```

## Risks and Guardrails

- Keep `varpro_scaling=True` as the default so existing inference behavior does not change.
- Do not route `varpro_scaling` through TensorFlow canonical configs unless a spec requires it; this is a PyTorch reassembly switch.
- Do not train a model for the demo. A trained checkpoint would make the proof slower and less deterministic.
- Keep generated PNG/JSON artifacts under `.artifacts/`, not committed.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Use PATH `python` in commands per `PYTHON-ENV-001`.

## Review Note

The writing-plans skill normally asks for a plan-document-reviewer subagent after drafting. This session's available subagent tool explicitly forbids spawning agents unless the user asks for subagents or delegation, so no subagent review was run for this plan.

## Verification Log

Completed on 2026-06-29 from branch `fno-stable`.

- `python -m compileall -q ptycho_torch scripts/studies`
  - Exit code: 0
  - Result: pass

- `python -m pytest -q tests/torch/test_varpro_probe_weighted_reassembly.py`
  - Exit code: 0
  - Result: `5 passed in 10.52s`

- `python -m pytest -q tests/torch/test_config_factory.py -k "varpro_scaling or patch_weighting"`
  - Exit code: 0
  - Result: `2 passed, 37 deselected, 3 warnings in 4.53s`

- `python -m pytest -q tests/torch/test_inference_reassembly_parity.py tests/torch/test_inference_reassembly_aggregation.py tests/torch/test_reassembly_multi_patch_parity.py tests/torch/test_reassembly_sign_parity.py`
  - Exit code: 0
  - Result: `5 passed in 7.39s`

- `python scripts/studies/demo_varpro_probe_weighted_reassembly.py --output-root .artifacts/varpro_probe_weighting_demo --seed 0`
  - Exit code: 0
  - Result: generated final visual demo artifacts and metrics

Final demo artifacts:

- `.artifacts/varpro_probe_weighting_demo/metrics.json`
- `.artifacts/varpro_probe_weighting_demo/reconstruction_grid.png`
- `.artifacts/varpro_probe_weighting_demo/error_grid.png`
- `.artifacts/varpro_probe_weighting_demo/probe_weight_map.png`

Final `complex_mae`:

- `uniform_no_varpro`: `0.5055029914195823`
- `uniform_varpro`: `0.14471995283852168`
- `probe_no_varpro`: `0.4745280794537518`
- `probe_varpro`: `0.08985779499433101`

Final `seam_mae`:

- `uniform_no_varpro`: `0.48687298831820164`
- `uniform_varpro`: `0.11348367594233229`
- `probe_no_varpro`: `0.4517085649599564`
- `probe_varpro`: `0.05120113707450472`
