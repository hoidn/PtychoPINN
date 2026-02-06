# Global State & Config Flow Refactor — Synthesis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce config “pinball” and global‑state coupling while preserving mandatory `update_legacy_dict` behavior and CLI compatibility.

**Architecture:** Keep the legacy bridge intact but **centralize** it (single helper), make the wrapper/runner boundary explicit (wrapper becomes thin), and introduce a single config builder used by both wrapper and runner. Changes are incremental and gated by tests + docs updates; no changes to core physics modules without an explicit exception plan.

**Tech Stack:** Python, pytest.

---

### Task 1: Capture the expected config flow (RED)

**Files:**
- Modify: `tests/torch/test_grid_lines_compare_wrapper.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write failing tests that enforce a single config builder**

Add a test that asserts the compare wrapper delegates to a shared builder instead of reconstructing `TorchRunnerConfig` field‑by‑field.

```python
# tests/torch/test_grid_lines_compare_wrapper.py
from unittest import mock

def test_compare_wrapper_uses_runner_config_builder(tmp_path, monkeypatch):
    from scripts.studies import grid_lines_compare_wrapper as wrapper

    spy = mock.Mock()
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.build_torch_runner_config", spy)

    wrapper.main([
        "--data_dir", str(tmp_path),
        "--architectures", "fno",
        "--output_dir", str(tmp_path / "out"),
    ])

    assert spy.called, "wrapper should call runner's config builder"
```

**Step 2: Run the test to confirm it fails**

Run: `pytest tests/torch/test_grid_lines_compare_wrapper.py::test_compare_wrapper_uses_runner_config_builder -v`

Expected: FAIL (no shared builder exists yet).

**Step 3: Commit the failing tests**

```bash
git add tests/torch/test_grid_lines_compare_wrapper.py
git commit -m "test: require shared config builder for torch runner"
```

---

### Task 2: Introduce a shared config builder (GREEN)

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Add a builder function**

Add a function in the runner (or a small helper module) that owns the full defaulting/override logic:

```python
# scripts/studies/grid_lines_torch_runner.py

def build_torch_runner_config(args) -> TorchRunnerConfig:
    """Return TorchRunnerConfig with all defaults + overrides applied in one place."""
    ...
```

**Step 2: Update the compare wrapper to call the builder**

Replace manual field‑by‑field construction with:

```python
cfg = grid_lines_torch_runner.build_torch_runner_config(args)
```

**Step 3: Add a runner test that asserts builder parity**

```python
# tests/torch/test_grid_lines_torch_runner.py

def test_build_torch_runner_config_defaults_match_cli_parser():
    from scripts.studies import grid_lines_torch_runner as runner
    args = runner.parse_args([])
    cfg = runner.build_torch_runner_config(args)
    assert cfg is not None
```

**Step 4: Run tests**

Run:
- `pytest tests/torch/test_grid_lines_compare_wrapper.py::test_compare_wrapper_uses_runner_config_builder -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py::test_build_torch_runner_config_defaults_match_cli_parser -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py scripts/studies/grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "refactor: centralize torch runner config construction"
```

---

### Task 3: Centralize legacy `params.cfg` mutation (RED)

**Files:**
- Create: `ptycho_torch/legacy_bridge.py`
- Modify: `tests/torch/test_workflows_components.py`

**Step 1: Add a failing test that asserts a single bridge call**

```python
# tests/torch/test_workflows_components.py
from unittest import mock

def test_legacy_bridge_called_once_for_training_payload(monkeypatch, tmp_path):
    from ptycho_torch import workflows

    spy = mock.Mock()
    monkeypatch.setattr("ptycho_torch.legacy_bridge.populate_legacy_params", spy)

    workflows.components.run_cdi_example_torch(
        train_data=tmp_path / "train.npz",
        test_data=None,
        config=None,
    )

    assert spy.called
```

**Step 2: Run the test**

Run: `pytest tests/torch/test_workflows_components.py::test_legacy_bridge_called_once_for_training_payload -v`

Expected: FAIL (bridge module does not exist).

**Step 3: Commit the failing test**

```bash
git add tests/torch/test_workflows_components.py
git commit -m "test: require centralized legacy bridge call"
```

---

### Task 4: Implement the centralized legacy bridge (GREEN)

**Files:**
- Create: `ptycho_torch/legacy_bridge.py`
- Modify: `ptycho_torch/workflows/components.py`
- Modify: `ptycho_torch/data_container_bridge.py`

**Step 1: Add the bridge helper**

```python
# ptycho_torch/legacy_bridge.py
from ptycho.params import cfg
from ptycho.config.config import TrainingConfig, InferenceConfig
from ptycho.config.config import update_legacy_dict

def populate_legacy_params(config):
    """Centralized mutation of params.cfg required for legacy modules."""
    update_legacy_dict(cfg, config)
```

**Step 2: Replace scattered calls**

Update call sites to use `populate_legacy_params` (still respecting mandatory `update_legacy_dict` semantics).

**Step 3: Run tests**

Run: `pytest tests/torch/test_workflows_components.py::test_legacy_bridge_called_once_for_training_payload -v`

Expected: PASS.

**Step 4: Commit**

```bash
git add ptycho_torch/legacy_bridge.py ptycho_torch/workflows/components.py ptycho_torch/data_container_bridge.py
git commit -m "refactor: centralize legacy params bridge"
```

---

### Task 5: Document the refined flow + compatibility constraints

**Files:**
- Modify: `docs/architecture_torch.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `docs/CONFIGURATION.md`

**Step 1: Add a new “Config Flow” section**

Document the single builder and the centralized legacy bridge; explicitly call out that `update_legacy_dict` remains mandatory unless future exceptions are approved.

**Step 2: Update workflow docs**

Add a short note that `grid_lines_compare_wrapper.py` delegates to the runner builder and that wrapper/runner defaults are now unified.

**Step 3: Commit**

```bash
git add docs/architecture_torch.md docs/workflows/pytorch.md docs/CONFIGURATION.md
git commit -m "docs: document unified torch config flow and legacy bridge"
```

---

### Task 6: Targeted verification

**Step 1: Run focused tests**

Run:
- `pytest tests/torch/test_grid_lines_compare_wrapper.py::test_compare_wrapper_uses_runner_config_builder -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py::test_build_torch_runner_config_defaults_match_cli_parser -v`
- `pytest tests/torch/test_workflows_components.py::test_legacy_bridge_called_once_for_training_payload -v`

Expected: PASS.

---

## Notes / Constraints
- Do **not** modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` unless a plan section explicitly authorizes it.
- `update_legacy_dict(params.cfg, config)` remains mandatory for legacy modules until a governed exception exists.
- Any future “explicit‑args API” proposal must be spec‑backed and parity‑tested.

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-02-06-global-state-refactor-synthesis.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
