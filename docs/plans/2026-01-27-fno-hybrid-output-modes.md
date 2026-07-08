# FNO/Hybrid Output Modes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an optional output-mode toggle for Torch FNO/Hybrid models to support `real_imag` (default), `amp_phase_logits` (post‑head), and `amp_phase` (dual‑head) without changing defaults.

**Architecture:** Introduce a config field (`generator_output_mode`) that flows from CLI → runner config → config factory → PyTorch model config. In `ptycho_torch/model.py::_predict_complex`, branch by mode and interpret generator outputs accordingly. For `amp_phase`, update FNO/Hybrid generators to emit explicit amp/phase heads. Defaults remain `real_imag`.

**Tech Stack:** Python, PyTorch, Lightning, existing Torch runner scripts, pytest.

---

## Task 1: Add config field + wiring (no behavior change)

**Files:**
- Modify: `ptycho/config/config.py` (add `generator_output_mode` to `ModelConfig`)
- Modify: `ptycho_torch/config_params.py` (add field to PT `ModelConfig`)
- Modify: `ptycho_torch/config_factory.py` (propagate override into PT model config)
- Test: `tests/torch/test_config_factory.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_config_factory.py

def test_generator_output_mode_override_propagates():
    payload = create_training_payload(
        train_data_file=Path("train.npz"),
        output_dir=Path("outputs"),
        overrides={
            "n_groups": 1,
            "gridsize": 1,
            "architecture": "fno",
            "generator_output_mode": "amp_phase_logits",
        },
    )
    assert payload.pt_model_config.generator_output_mode == "amp_phase_logits"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_config_factory.py::test_generator_output_mode_override_propagates -v`

Expected: FAIL (field missing / not propagated).

**Step 3: Write minimal implementation**

- Add `generator_output_mode: Literal["real_imag", "amp_phase_logits", "amp_phase"] = "real_imag"` to:
  - `ptycho/config/config.py::ModelConfig`
  - `ptycho_torch/config_params.py::ModelConfig`
- In `ptycho_torch/config_factory.py`, if `overrides` includes `generator_output_mode`, pass it into PT model config (via `update_existing_config`).

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_config_factory.py::test_generator_output_mode_override_propagates -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho/config/config.py ptycho_torch/config_params.py ptycho_torch/config_factory.py tests/torch/test_config_factory.py
git commit -m "feat(torch): add generator_output_mode config"
```

---

## Task 2: Add CLI + runner wiring

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_grid_lines_torch_runner.py

def test_runner_accepts_output_mode_flag():
    cfg = TorchRunnerConfig(
        train_npz=Path("train.npz"),
        test_npz=Path("test.npz"),
        output_dir=Path("out"),
        architecture="fno",
        generator_output_mode="amp_phase_logits",
    )
    assert cfg.generator_output_mode == "amp_phase_logits"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_runner_accepts_output_mode_flag -v`

Expected: FAIL (field missing).

**Step 3: Write minimal implementation**

- Add `generator_output_mode: str = "real_imag"` to `TorchRunnerConfig`.
- Add CLI flag `--torch-output-mode` (wrapper) and `--output-mode` (runner) with choices.
- Pass `generator_output_mode` into `TorchRunnerConfig` in both CLI entry points.

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_runner_accepts_output_mode_flag -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py scripts/studies/grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "feat(studies): add torch output mode flag"
```

---

## Task 3: Implement `amp_phase_logits` in `_predict_complex`

**Files:**
- Modify: `ptycho_torch/model.py`
- Test: `tests/torch/test_model_output_modes.py` (new)

**Step 1: Write the failing test**

```python
# tests/torch/test_model_output_modes.py

def test_amp_phase_logits_bounds():
    model = PtychoPINN_Lightning(
        model_config=PTModelConfig(generator_output_mode="amp_phase_logits", ...),
        data_config=PTDataConfig(...),
        training_config=PTTrainingConfig(...),
        inference_config=PTInferenceConfig(...),
        generator_module=DummyTwoChannelGenerator(),
        generator_output="amp_phase_logits",
    )
    x = torch.randn(2, 1, 64, 64)
    x_complex, amp, phase = model._predict_complex(x)
    assert amp.min() >= 0 and amp.max() <= 1
    assert phase.min() >= -math.pi and phase.max() <= math.pi
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_model_output_modes.py::test_amp_phase_logits_bounds -v`

Expected: FAIL (mode unsupported / bounds not enforced).

**Step 3: Write minimal implementation**

- Add `generator_output_mode` handling in `PtychoPINN_Lightning.__init__` and `_predict_complex`.
- For `amp_phase_logits`, interpret two channels as logits:
  - `amp = torch.sigmoid(x0)` (or `softplus` if desired)
  - `phase = math.pi * torch.tanh(x1)`
  - `x_complex = amp * torch.exp(1j * phase)`
- Add a shape check: last dim must be 2.

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_model_output_modes.py::test_amp_phase_logits_bounds -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho_torch/model.py tests/torch/test_model_output_modes.py
git commit -m "feat(torch): add amp_phase_logits output mode"
```

---

## Task 4: Add dual‑head `amp_phase` generators (more invasive)

**Files:**
- Modify: `ptycho_torch/generators/fno.py`
- Modify: `ptycho_torch/model.py`
- Test: `tests/torch/test_model_output_modes.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_model_output_modes.py

def test_amp_phase_mode_accepts_tuple():
    model = PtychoPINN_Lightning(
        model_config=PTModelConfig(generator_output_mode="amp_phase", ...),
        ...,
        generator_module=DummyAmpPhaseGenerator(),
        generator_output="amp_phase",
    )
    x = torch.randn(1, 1, 64, 64)
    x_complex, amp, phase = model._predict_complex(x)
    assert amp.shape == phase.shape
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_model_output_modes.py::test_amp_phase_mode_accepts_tuple -v`

Expected: FAIL (mode unsupported).

**Step 3: Write minimal implementation**

- In `ptycho_torch/generators/fno.py`, add dual heads that return `(amp, phase)`.
- Set `generator_output` to `"amp_phase"` when this mode is chosen.
- Update `_predict_complex` to accept `(amp, phase)` tuple and call `CombineComplex`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_model_output_modes.py::test_amp_phase_mode_accepts_tuple -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho_torch/generators/fno.py ptycho_torch/model.py tests/torch/test_model_output_modes.py
git commit -m "feat(torch): add amp_phase generator output mode"
```

---

## Task 5: Documentation & smoke test

**Files:**
- Modify: `scripts/studies/README.md`
- Optional: `docs/COMMANDS_REFERENCE.md`

**Step 1: Add CLI usage**

Document the new flags and modes (real_imag / amp_phase_logits / amp_phase) with a small example.

**Step 2: Run a quick smoke test**

Run:
```
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 --gridsize 1 --output-dir outputs/grid_lines_gs1_n64_outmode \
  --architectures fno --torch-epochs 1 --nimgs-train 1 --nimgs-test 1 \
  --torch-output-mode amp_phase_logits
```
Expected: finishes and writes metrics for `pinn_fno`.

**Step 3: Commit**

```bash
git add scripts/studies/README.md docs/COMMANDS_REFERENCE.md
git commit -m "docs: document torch output mode flag"
```

---

Plan complete and saved to `docs/plans/2026-01-27-fno-hybrid-output-modes.md`. Two execution options:

1. Subagent-Driven (this session) – I dispatch a fresh subagent per task, review between tasks.
2. Parallel Session – Open a new session in a worktree and execute with superpowers:executing-plans.

Which approach?
