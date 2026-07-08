# FNO Hyperparameter Study Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reproducible FNO/Hybrid hyperparameter sweep that reports phase‑quality metrics vs. parameter count/inference time and produces CSV + Pareto plot outputs.

**Architecture:** Enhance the Torch grid‑lines runner to report parameter count and inference time, then add a dedicated sweep orchestrator that reuses cached grid‑lines NPZs, iterates a fixed config grid, aggregates metrics, and plots results.

**Tech Stack:** Python, PyTorch, NumPy, Matplotlib, pandas (if already available; otherwise use CSV + Matplotlib only).

**Required skills:** @superpowers:test-driven-development, @superpowers:executing-plans

**Constraints:** Training configs only (input_transform, modes, width, architecture). Do not modify core physics/model code.

---

### Task 1: Torch runner reports parameter count + inference time

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_grid_lines_torch_runner.py
class TestRunGridLinesTorchScaffold:
    def test_runner_reports_model_params_and_inference_time(self, synthetic_npz, tmp_path):
        train_path, test_path = synthetic_npz
        output_dir = tmp_path / "output"

        cfg = TorchRunnerConfig(
            train_npz=train_path,
            test_npz=test_path,
            output_dir=output_dir,
            architecture="fno",
            epochs=1,
        )

        with patch('scripts.studies.grid_lines_torch_runner.run_torch_training') as mock_train:
            mock_train.return_value = {
                'model': None,
                'history': {},
                'generator': 'fno',
                'scaffold': True,
            }
            with patch('scripts.studies.grid_lines_torch_runner.run_torch_inference') as mock_infer:
                mock_infer.return_value = np.random.rand(64, 64).astype(np.complex64)
                with patch('scripts.studies.grid_lines_torch_runner.compute_metrics') as mock_metrics:
                    mock_metrics.return_value = {'mse': 0.1}
                    result = run_grid_lines_torch(cfg)

        assert 'model_params' in result
        assert isinstance(result['model_params'], int)
        assert 'inference_time_s' in result
        assert isinstance(result['inference_time_s'], float)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::TestRunGridLinesTorchScaffold::test_runner_reports_model_params_and_inference_time -vv`
Expected: FAIL (missing keys).

**Step 3: Implement minimal code**

In `run_grid_lines_torch`:
- After obtaining the trained model, compute:
```python
model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```
- Wrap inference with timing:
```python
import time
if torch.cuda.is_available():
    torch.cuda.synchronize()
start = time.perf_counter()
# run inference
if torch.cuda.is_available():
    torch.cuda.synchronize()
end = time.perf_counter()
```
- Add `model_params` and `inference_time_s` to `result_dict`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::TestRunGridLinesTorchScaffold::test_runner_reports_model_params_and_inference_time -vv`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "feat(torch-runner): report params + inference time"
```

---

### Task 2: Sweep orchestrator script

**Files:**
- Create: `scripts/studies/fno_hyperparam_study.py`
- Test: `tests/test_fno_hyperparam_study.py`

**Step 1: Write the failing test**

```python
# tests/test_fno_hyperparam_study.py
import csv
from pathlib import Path

def test_sweep_writes_csv(monkeypatch, tmp_path):
    from scripts.studies.fno_hyperparam_study import run_sweep

    def fake_run_torch(cfg):
        return {
            'metrics': {
                'ssim': [0.5, 0.9],
                'psnr': [10.0, 30.0],
                'mae': [0.2, 0.05],
            },
            'model_params': 1234,
            'inference_time_s': 0.12,
        }

    monkeypatch.setattr('scripts.studies.fno_hyperparam_study.run_grid_lines_torch', fake_run_torch)

    out_dir = tmp_path / 'study'
    out_dir.mkdir()
    csv_path = run_sweep(output_dir=out_dir, epochs=1, light=True)
    assert csv_path.exists()

    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert 'ssim_phase' in rows[0]
    assert 'model_params' in rows[0]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_fno_hyperparam_study.py::test_sweep_writes_csv -vv`
Expected: FAIL (script missing).

**Step 3: Implement minimal orchestrator**

Create `scripts/studies/fno_hyperparam_study.py` with:
- `run_sweep(output_dir: Path, epochs: int, light: bool)` that:
  - Ensures NPZs exist (call `grid_lines_workflow` once if missing).
  - Defines the grid (architecture, input_transform, modes, width).
  - Calls `run_grid_lines_torch` per config.
  - Writes a CSV with `ssim_phase`, `psnr_phase`, `mae_phase`, `model_params`, `inference_time_s`, and config columns.
  - Returns CSV path.
- `main()` with CLI args (output_dir, epochs, light).

**Step 4: Run test to verify pass**

Run: `pytest tests/test_fno_hyperparam_study.py::test_sweep_writes_csv -vv`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/fno_hyperparam_study.py tests/test_fno_hyperparam_study.py
git commit -m "feat(studies): add FNO hyperparam sweep orchestrator"
```

---

### Task 3: Plotting + docs

**Files:**
- Modify: `scripts/studies/fno_hyperparam_study.py`
- Modify: `scripts/studies/README.md`

**Step 1: Add plotting output**

Extend the orchestrator to write `pareto_plot.png` (param count vs phase SSIM or PSNR; log‑scale x). Use Matplotlib only.

**Step 2: Update docs**

Add a short section in `scripts/studies/README.md` with usage example and output locations.

**Step 3: Commit**

```bash
git add scripts/studies/fno_hyperparam_study.py scripts/studies/README.md
git commit -m "docs(studies): document FNO hyperparam sweep"
```

---

### Final Verification

```bash
pytest tests/torch/test_grid_lines_torch_runner.py::TestRunGridLinesTorchScaffold::test_runner_reports_model_params_and_inference_time -vv
pytest tests/test_fno_hyperparam_study.py::test_sweep_writes_csv -vv
```

**Plan complete.**
