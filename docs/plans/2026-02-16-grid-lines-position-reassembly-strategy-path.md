# Grid-Lines Position Reassembly Strategy Path Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor Torch grid-lines external position reassembly into an explicit strategy path (`auto`, `shift_sum`, `batched`) so large N=128 external studies avoid TensorFlow GPU OOM while preserving small-job parity behavior.

**Architecture:** Keep `reassembly_mode` semantics unchanged (`grid_lines` vs `position`) and add a strategy selector only inside the `position` path in `scripts/studies/grid_lines_torch_runner.py`. Implement one normalization/shape adapter for position inputs, then route through two execution backends: existing `tf_helper.reassemble_position` (shift-sum streaming) and batched `tf_helper.reassemble_whole_object(..., batch_size=...)`. Add an auto policy based on patch count/N and wire new config knobs through wrapper CLI so studies can force backend when needed.

**Tech Stack:** Python 3.11, NumPy, TensorFlow helper API (`ptycho/tf_helper.py`), argparse, pytest

---

### Task 1: Add Runner Strategy Config Contract (RED)

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write the failing test**

Add tests that validate new config defaults and allowed values:

```python
def test_position_strategy_defaults_are_stable(tmp_path):
    cfg = TorchRunnerConfig(
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_dir=tmp_path,
    )
    assert cfg.position_reassembly_backend == "auto"
    assert cfg.position_reassembly_batch_size == 64
```

```python
@pytest.mark.parametrize("backend", ["auto", "shift_sum", "batched"])
def test_position_strategy_accepts_supported_backends(tmp_path, backend):
    cfg = TorchRunnerConfig(
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_dir=tmp_path,
        position_reassembly_backend=backend,
    )
    assert cfg.position_reassembly_backend == backend
```

```python
def test_position_strategy_rejects_unknown_backend(tmp_path):
    cfg = TorchRunnerConfig(
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_dir=tmp_path,
        position_reassembly_backend="invalid",
    )
    with pytest.raises(ValueError, match="position_reassembly_backend"):
        run_grid_lines_torch(cfg)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py -k "position_strategy" -v`
Expected: FAIL (fields/validation missing)

**Step 3: Write minimal implementation**

In `TorchRunnerConfig` add:

```python
position_reassembly_backend: str = "auto"  # auto | shift_sum | batched
position_reassembly_batch_size: int = 64
```

Add validation helper called near the top of `run_grid_lines_torch`:

```python
def _validate_position_reassembly_config(cfg: TorchRunnerConfig) -> None:
    allowed = {"auto", "shift_sum", "batched"}
    if cfg.position_reassembly_backend not in allowed:
        raise ValueError(...)
    if cfg.position_reassembly_batch_size <= 0:
        raise ValueError(...)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py -k "position_strategy" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "feat(studies): add position reassembly strategy config for torch runner"
```

---

### Task 2: Split Position Reassembly Into Strategy Functions (RED)

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write the failing test**

Add backend dispatch tests with monkeypatch:

```python
def test_position_backend_shift_sum_calls_reassemble_position(monkeypatch):
    called = {"shift": False}
    monkeypatch.setattr("ptycho.tf_helper.reassemble_position", lambda *a, **k: _mark(called, "shift"))
    _reassemble_with_coords_offsets(..., backend="shift_sum", batch_size=64)
    assert called["shift"]
```

```python
def test_position_backend_batched_calls_reassemble_whole_object(monkeypatch):
    captured = {}
    def fake_whole_object(patches, offsets, size, batch_size, norm=False):
        captured["size"] = size
        captured["batch_size"] = batch_size
        return np.ones((1, size, size, 1), dtype=np.complex64)
    monkeypatch.setattr("ptycho.tf_helper.reassemble_whole_object", fake_whole_object)
    _reassemble_with_coords_offsets(..., backend="batched", batch_size=32)
    assert captured == {"size": 64, "batch_size": 32}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py -k "backend_shift_sum or backend_batched" -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Refactor helper signatures:

```python
def _normalize_position_inputs(pred_complex, test_data):
    # returns patches_bhwc1, offsets_b12c1, offsets_b112


def _reassemble_position_shift_sum(patches, offsets_b112, M):
    return hh.reassemble_position(patches, offsets_b112, M=M)


def _reassemble_position_batched(patches, offsets_b12c1, M, batch_size):
    return hh.reassemble_whole_object(
        patches=patches,
        offsets=offsets_b12c1,
        size=M,
        batch_size=batch_size,
        norm=False,
    )
```

Update `_reassemble_with_coords_offsets(...)` to dispatch by backend.

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py -k "backend_shift_sum or backend_batched" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "refactor(studies): split position reassembly into shift-sum and batched backends"
```

---

### Task 3: Implement Auto Policy and OOM Fallback Behavior (RED)

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write the failing test**

```python
def test_auto_backend_prefers_batched_for_large_position_jobs(monkeypatch):
    pred = np.ones((4096, 128, 128, 1), dtype=np.complex64)
    test_data = {"coords_offsets": np.zeros((4096, 1, 2, 1), dtype=np.float32)}
    backend = _choose_position_backend(pred, test_data, configured="auto")
    assert backend == "batched"
```

```python
def test_auto_backend_prefers_shift_sum_for_small_jobs(monkeypatch):
    pred = np.ones((64, 64, 64, 1), dtype=np.complex64)
    test_data = {"coords_offsets": np.zeros((64, 1, 2, 1), dtype=np.float32)}
    backend = _choose_position_backend(pred, test_data, configured="auto")
    assert backend == "shift_sum"
```

```python
def test_shift_sum_oom_falls_back_to_batched(monkeypatch):
    monkeypatch.setattr("ptycho.tf_helper.reassemble_position", _raise_resource_exhausted)
    out = _reassemble_with_coords_offsets(..., backend="auto", batch_size=32)
    assert out.shape == (64, 64)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py -k "choose_position_backend or falls_back_to_batched" -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add policy helper:

```python
def _choose_position_backend(pred_complex, test_data, configured: str) -> str:
    if configured != "auto":
        return configured
    b = int(np.asarray(pred_complex).shape[0])
    n = int(np.asarray(pred_complex).shape[-2])
    if b >= 1024 or n >= 128:
        return "batched"
    return "shift_sum"
```

Add guarded dispatch:

```python
try:
    return _reassemble_position_shift_sum(...)
except tf.errors.ResourceExhaustedError:
    logger.warning("Shift-sum OOM; retrying with batched position reassembly")
    return _reassemble_position_batched(...)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py -k "choose_position_backend or falls_back_to_batched" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "feat(studies): add auto backend policy and OOM fallback for position reassembly"
```

---

### Task 4: Wire Strategy Knobs Through Wrapper CLI (RED)

**Files:**
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`
- Test: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Write the failing test**

```python
def test_parse_args_accepts_position_reassembly_strategy_flags(tmp_path):
    args = parse_args([
        "--N", "128",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--dataset-source", "external_raw_npz",
        "--train-data", "train.npz",
        "--test-data", "test.npz",
        "--torch-position-reassembly-backend", "batched",
        "--torch-position-reassembly-batch-size", "32",
    ])
    assert args.torch_position_reassembly_backend == "batched"
    assert args.torch_position_reassembly_batch_size == 32
```

```python
def test_external_mode_passes_position_strategy_to_torch_runner(monkeypatch, tmp_path):
    # intercept TorchRunnerConfig construction and assert backend/batch_size propagated
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_grid_lines_compare_wrapper.py -k "position_reassembly_strategy" -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In wrapper parser add:

```python
parser.add_argument(
    "--torch-position-reassembly-backend",
    choices=("auto", "shift_sum", "batched"),
    default="auto",
)
parser.add_argument(
    "--torch-position-reassembly-batch-size",
    type=int,
    default=64,
)
```

In Torch runner config creation pass through both values.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_grid_lines_compare_wrapper.py -k "position_reassembly_strategy" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_compare_wrapper.py tests/test_grid_lines_compare_wrapper.py
git commit -m "feat(studies): expose torch position reassembly strategy controls in wrapper CLI"
```

---

### Task 5: Add Regression Coverage for External N=128 OOM Scenario (RED)

**Files:**
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Create: `tests/torch/test_grid_lines_position_reassembly_strategy.py`
- Test: `tests/torch/test_grid_lines_position_reassembly_strategy.py`

**Step 1: Write the failing test**

```python
def test_large_external_n128_uses_batched_backend_without_tf_shift_sum(monkeypatch):
    # Build synthetic pred/test_data with B=4096, N=128
    # Monkeypatch shift_sum backend to raise if called
    # Monkeypatch batched backend to return deterministic recon
    # Assert auto mode succeeds and never touches shift_sum backend
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_position_reassembly_strategy.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement any missing import/export helpers in `grid_lines_torch_runner.py` so strategy helpers are testable (keep functions internal but deterministic).

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_position_reassembly_strategy.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/torch/test_grid_lines_position_reassembly_strategy.py tests/torch/test_grid_lines_torch_runner.py scripts/studies/grid_lines_torch_runner.py
git commit -m "test(studies): cover external N128 position reassembly strategy routing"
```

---

### Task 6: Documentation and Study Repro Guidance (GREEN)

**Files:**
- Modify: `docs/workflows/pytorch.md`
- Modify: `docs/studies/index.md`
- Modify: `docs/COMMANDS_REFERENCE.md`
- Test: `tests/studies/test_studies_index_entries.py`

**Step 1: Update docs with new strategy contract**

Document:
- new backend values and defaults
- recommended settings for dense external N=128 studies
- failure mode guidance for TensorFlow reassembly OOM

Example snippet to add:

```bash
--torch-position-reassembly-backend batched \\
--torch-position-reassembly-batch-size 32
```

**Step 2: Run docs/index tests**

Run: `pytest tests/studies/test_studies_index_entries.py -q`
Expected: PASS

**Step 3: Commit**

```bash
git add docs/workflows/pytorch.md docs/studies/index.md docs/COMMANDS_REFERENCE.md tests/studies/test_studies_index_entries.py
git commit -m "docs(studies): document position reassembly strategy path for external torch datasets"
```

---

### Final Verification Gate

Run full focused suite:

```bash
pytest tests/torch/test_grid_lines_torch_runner.py -q
pytest tests/test_grid_lines_compare_wrapper.py -q
pytest tests/torch/test_grid_lines_position_reassembly_strategy.py -q
pytest tests/test_tf_helper.py -q
```

Expected: PASS

