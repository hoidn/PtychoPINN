# FNO/Hybrid Output Modes (Design)

## Goal
Add an **optional** output-mode toggle for Torch FNO/Hybrid models so we can choose between the current `real_imag` behavior and two amp/phase alternatives without changing defaults.

## Current Behavior
- FNO/Hybrid generators output a 2‑channel tensor interpreted as **real/imag**.
- `ptycho_torch/model.py::_predict_complex` converts real/imag → complex, then derives amp/phase via `abs`/`angle`.
- CNN PINN outputs amp/phase directly via explicit activation heads (sigmoid for amp, `π*tanh` for phase).

## Proposed Output Modes
Add a config field (name TBD, e.g. `generator_output_mode` or `fno_output_mode`) with values:
- **`real_imag`** (default, current behavior)
- **`amp_phase_logits`** (lightweight post‑head)
- **`amp_phase`** (dual‑head, more invasive)

### Option A: `amp_phase_logits` (recommended)
- Keep the generator output unchanged (2 channels).
- Reinterpret the 2 channels as **amp/phase logits** in `_predict_complex`:
  - `amp = sigmoid(x0)` (or `softplus` if preferred)
  - `phase = π * tanh(x1)`
  - `complex = amp * exp(i * phase)`
- No generator refactor required.

### Option B: `amp_phase`
- Modify FNO/Hybrid generators to emit **two explicit heads**:
  - `amp_head` → 1 channel + sigmoid/softplus
  - `phase_head` → 1 channel + `π*tanh`
- Generator returns `(amp, phase)`, and `_predict_complex` uses `CombineComplex`.

## Architecture & Data Flow
- Config wiring adds `generator_output_mode` to:
  - `ptycho/config/config.py::ModelConfig`
  - `ptycho_torch/config_params.py::ModelConfig`
  - `ptycho_torch/config_factory.py` (propagate into `pt_model_config`)
  - CLI flags in `grid_lines_torch_runner.py` and `grid_lines_compare_wrapper.py`
- `ptycho_torch/model.py::_predict_complex` branches on the mode:
  - `real_imag`: unchanged
  - `amp_phase_logits`: apply activations to logits, then combine
  - `amp_phase`: combine provided amp/phase directly

## Error Handling
- Validate `generator_output_mode` against allowed values and raise `ValueError` for invalid inputs.
- For `amp_phase_logits`, assert generator output has 2 channels; raise a descriptive error otherwise.

## Testing
- Unit tests for `_predict_complex`:
  - `real_imag` path unchanged
  - `amp_phase_logits`: amp in [0,1], phase in [-π, π]
  - `amp_phase`: accepts tuple outputs and combines correctly
- Config wiring test: CLI → runner config → factory override → model config
- Smoke test: 1‑epoch Torch run with `--torch-output-mode amp_phase_logits`

## Risks & Notes
- Switching semantics likely requires retraining to achieve meaningful results.
- `amp_phase_logits` is minimally invasive and best for quick A/B evaluation.
- `amp_phase` gives CNN‑like semantics but requires generator refactor.
