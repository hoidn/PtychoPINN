# Grid Lines Visuals Extension Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend grid-lines outputs to include per-model recon artifacts and visuals, and render a composite PNG that includes only the models actually run (GT, PINN, Baseline, FNO, Hybrid).

**Architecture:** Persist stitched complex reconstructions for each model in `output_dir/recons/<label>/recon.npz`, then render visuals from those artifacts. The TF workflow saves recon artifacts for GT/PINN/Baseline; the Torch runner saves recon artifacts for FNO/Hybrid. A dynamic visualizer builds `visuals/compare_amp_phase.png` and per-model `visuals/amp_phase_<label>.png` by scanning available recon artifacts.

**Tech Stack:** Python (NumPy, Matplotlib), PyTorch Lightning runner, existing grid-lines workflows.

**Required skills:** @superpowers:test-driven-development, @superpowers:executing-plans

**Constraints:** Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`. Use `python` for commands (PYTHON-ENV-001).

---

### Task 1: Add recon artifact + visualizer helpers

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Modify: `tests/test_grid_lines_workflow.py`

**Step 1: Write the failing tests**

Add tests for recon artifact saving and dynamic visuals:

```python
# tests/test_grid_lines_workflow.py
class TestReconArtifacts:
    def test_save_recon_artifact_writes_npz(self, tmp_path):
        from ptycho.workflows.grid_lines_workflow import save_recon_artifact
        recon = (np.ones((4, 4)) + 1j * np.ones((4, 4))).astype(np.complex64)
        path = save_recon_artifact(tmp_path, "pinn", recon)
        assert path.exists()
        data = np.load(path)
        assert "YY_pred" in data and "amp" in data and "phase" in data
        assert data["YY_pred"].shape == (4, 4)

    def test_save_comparison_png_dynamic(self, tmp_path):
        from ptycho.workflows.grid_lines_workflow import save_comparison_png_dynamic
        gt_amp = np.ones((4, 4))
        gt_phase = np.zeros((4, 4))
        recons = {
            "pinn": {"amp": np.zeros((4, 4)), "phase": np.zeros((4, 4))},
        }
        out = save_comparison_png_dynamic(tmp_path, gt_amp, gt_phase, recons, order=("pinn",))
        assert out.exists()
```

**Step 2: Run tests to verify failure**

Run: `pytest tests/test_grid_lines_workflow.py::TestReconArtifacts -vv`
Expected: FAIL with ImportError/AttributeError (helpers missing).

**Step 3: Implement minimal helpers**

Add helpers to `ptycho/workflows/grid_lines_workflow.py`:

```python
def save_recon_artifact(output_dir: Path, label: str, recon_complex: np.ndarray) -> Path:
    recon_dir = output_dir / "recons" / label
    recon_dir.mkdir(parents=True, exist_ok=True)
    recon = np.squeeze(recon_complex)
    if recon.ndim > 2:
        recon = recon[0]
    recon = recon.astype(np.complex64)
    amp = np.abs(recon)
    phase = np.angle(recon)
    path = recon_dir / "recon.npz"
    np.savez(path, YY_pred=recon, amp=amp, phase=phase)
    return path

_LABEL_TITLES = {
    "pinn": "PINN",
    "baseline": "Baseline",
    "pinn_fno": "FNO",
    "pinn_hybrid": "Hybrid",
    "gt": "GT",
}

def save_comparison_png_dynamic(
    output_dir: Path,
    gt_amp: np.ndarray,
    gt_phase: np.ndarray,
    recons: Dict[str, Dict[str, np.ndarray]],
    order: Tuple[str, ...],
) -> Path:
    import matplotlib.pyplot as plt

    labels = [label for label in order if label in recons]
    ncols = 1 + len(labels)
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8))

    axes[0, 0].imshow(gt_amp, cmap="viridis")
    axes[0, 0].set_title("GT Amplitude")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(gt_phase, cmap="twilight")
    axes[1, 0].set_title("GT Phase")
    axes[1, 0].axis("off")

    for idx, label in enumerate(labels, start=1):
        amp = recons[label]["amp"]
        phase = recons[label]["phase"]
        title = _LABEL_TITLES.get(label, label)
        axes[0, idx].imshow(amp, cmap="viridis")
        axes[0, idx].set_title(f"{title} Amplitude")
        axes[0, idx].axis("off")
        axes[1, idx].imshow(phase, cmap="twilight")
        axes[1, idx].set_title(f"{title} Phase")
        axes[1, idx].axis("off")

    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    out_path = visuals_dir / "compare_amp_phase.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_grid_lines_workflow.py::TestReconArtifacts -vv`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_grid_lines_workflow.py ptycho/workflows/grid_lines_workflow.py
git commit -m "test: add recon artifact helper coverage"
```

---

### Task 2: Wire TF workflow to save recon artifacts + visuals

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Modify: `tests/test_grid_lines_workflow.py`

**Step 1: Write the failing test**

Add a unit test for dynamic rendering with missing models (skip logic):

```python
class TestReconArtifacts:
    def test_save_comparison_png_skips_missing(self, tmp_path):
        from ptycho.workflows.grid_lines_workflow import save_comparison_png_dynamic
        gt_amp = np.ones((4, 4))
        gt_phase = np.zeros((4, 4))
        recons = {"baseline": {"amp": np.zeros((4, 4)), "phase": np.zeros((4, 4))}}
        out = save_comparison_png_dynamic(tmp_path, gt_amp, gt_phase, recons, order=("pinn", "baseline"))
        assert out.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_grid_lines_workflow.py::TestReconArtifacts::test_save_comparison_png_skips_missing -vv`
Expected: FAIL if helper doesn’t skip missing labels cleanly.

**Step 3: Implement minimal TF wiring**

In `run_grid_lines_workflow`, after stitching:
- Save recon artifacts for GT, baseline, and PINN (only if inference succeeded).
- Render visuals using the helper.

Example patch:

```python
# after stitched arrays are computed
recon_paths = {}
recon_paths["gt"] = save_recon_artifact(cfg.output_dir, "gt", gt_squeezed)
recon_paths["baseline"] = save_recon_artifact(cfg.output_dir, "baseline", base_stitched)
if pinn_pred is not None:
    recon_paths["pinn"] = save_recon_artifact(cfg.output_dir, "pinn", pinn_stitched)

recons = {}
for label, path in recon_paths.items():
    data = np.load(path)
    recons[label] = {"amp": data["amp"], "phase": data["phase"]}

compare_png = save_comparison_png_dynamic(
    cfg.output_dir,
    gt_amp_2d,
    gt_phase_2d,
    recons,
    order=("pinn", "baseline"),
)
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_grid_lines_workflow.py::TestReconArtifacts -vv`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_grid_lines_workflow.py ptycho/workflows/grid_lines_workflow.py
git commit -m "feat: save grid-lines recon artifacts and visuals"
```

---

### Task 3: Torch runner saves recon artifacts

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write the failing test**

```python
# tests/torch/test_grid_lines_torch_runner.py
class TestRunGridLinesTorchScaffold:
    def test_runner_writes_recon_artifact(self, synthetic_npz, tmp_path):
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
                mock_infer.return_value = np.random.rand(1, 64, 64).astype(np.complex64)
                with patch('scripts.studies.grid_lines_torch_runner.compute_metrics') as mock_metrics:
                    mock_metrics.return_value = {'mse': 0.1}
                    result = run_grid_lines_torch(cfg)
        recon_path = Path(output_dir) / "recons" / "pinn_fno" / "recon.npz"
        assert recon_path.exists()
        assert "recon_path" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::TestRunGridLinesTorchScaffold::test_runner_writes_recon_artifact -vv`
Expected: FAIL (recon not written).

**Step 3: Implement minimal Torch artifact save**

In `run_grid_lines_torch`, after `pred_for_metrics` is computed:

```python
from ptycho.workflows.grid_lines_workflow import save_recon_artifact

recon_target = pred_for_metrics
if not np.iscomplexobj(recon_target):
    recon_target = recon_target.astype(np.complex64)
recon_path = save_recon_artifact(cfg.output_dir, f"pinn_{cfg.architecture}", recon_target)
```

Include `recon_path` in `result_dict`.

**Step 4: Run test to verify pass**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::TestRunGridLinesTorchScaffold::test_runner_writes_recon_artifact -vv`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/torch/test_grid_lines_torch_runner.py scripts/studies/grid_lines_torch_runner.py
git commit -m "feat: persist torch grid-lines recon artifacts"
```

---

### Task 4: Compare wrapper re-renders composite + per-model PNGs

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Write the failing test**

```python
# tests/test_grid_lines_compare_wrapper.py
def test_wrapper_renders_visuals(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    def fake_tf_run(cfg):
        datasets_dir = cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "train.npz").write_bytes(b"stub")
        (datasets_dir / "test.npz").write_bytes(b"stub")
        (cfg.output_dir / "metrics.json").write_text(json.dumps({"pinn": {}, "baseline": {}}))
        return {"train_npz": str(datasets_dir / "train.npz"), "test_npz": str(datasets_dir / "test.npz")}

    def fake_torch_run(cfg):
        recon_dir = cfg.output_dir / "recons" / f"pinn_{cfg.architecture}"
        recon_dir.mkdir(parents=True, exist_ok=True)
        (recon_dir / "recon.npz").write_bytes(b"stub")
        return {"metrics": {"mse": 0.3}}

    called = {}
    def fake_render(output_dir, order):
        called["order"] = order
        visuals = output_dir / "visuals"
        visuals.mkdir(parents=True, exist_ok=True)
        out = visuals / "compare_amp_phase.png"
        out.write_bytes(b"stub")
        return {"compare": str(out)}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.run_grid_lines_workflow", fake_tf_run)
    monkeypatch.setattr("scripts.studies.grid_lines_torch_runner.run_grid_lines_torch", fake_torch_run)
    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.render_grid_lines_visuals", fake_render)

    run_grid_lines_compare(
        N=64,
        gridsize=1,
        output_dir=tmp_path,
        architectures=("cnn", "baseline", "fno"),
        probe_npz=Path("dummy_probe.npz"),
    )
    assert called["order"] == ("gt", "pinn", "baseline", "pinn_fno")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_renders_visuals -vv`
Expected: FAIL (renderer not called).

**Step 3: Implement wrapper wiring + renderer**

Add a renderer in `ptycho/workflows/grid_lines_workflow.py` that:
- Scans `output_dir/recons/*/recon.npz` for available labels
- Loads `amp/phase` from each recon
- Calls `save_comparison_png_dynamic`
- Writes per-model PNGs (`amp_phase_<label>.png`)

Example skeleton:

```python
def render_grid_lines_visuals(output_dir: Path, order: Tuple[str, ...]) -> Dict[str, str]:
    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    recons = {}
    for label in order:
        path = output_dir / "recons" / label / "recon.npz"
        if not path.exists():
            continue
        data = np.load(path)
        recons[label] = {"amp": data["amp"], "phase": data["phase"]}
        # per-model PNG
        save_amp_phase_png(visuals_dir, label, data["amp"], data["phase"])

    gt = recons.pop("gt", None)
    if gt is None:
        return {}
    compare = save_comparison_png_dynamic(output_dir, gt["amp"], gt["phase"], recons, order=tuple(l for l in order if l != "gt"))
    return {"compare": str(compare)}
```

In `grid_lines_compare_wrapper.py`, after Torch runs:

```python
from ptycho.workflows.grid_lines_workflow import render_grid_lines_visuals
order = ("gt", "pinn", "baseline", "pinn_fno", "pinn_hybrid")
render_grid_lines_visuals(output_dir, order=order)
```

**Step 4: Run test to verify pass**

Run: `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_renders_visuals -vv`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_grid_lines_compare_wrapper.py scripts/studies/grid_lines_compare_wrapper.py ptycho/workflows/grid_lines_workflow.py
git commit -m "feat: render grid-lines visuals from available models"
```

---

### Task 5: Update study documentation for new outputs

**Files:**
- Modify: `scripts/studies/README.md`

**Step 1: Write the failing doc expectation (optional)**

Skip if no doc test exists.

**Step 2: Update documentation**

Update the output structure to mention `recons/` and per-model PNGs:

```markdown
output_dir/
├── datasets/N{N}/gs{gridsize}/
├── models/
├── recons/
│   ├── gt/recon.npz
│   ├── pinn/recon.npz
│   ├── baseline/recon.npz
│   ├── pinn_fno/recon.npz
│   └── pinn_hybrid/recon.npz
├── visuals/compare_amp_phase.png  # dynamic grid (GT + available models)
├── visuals/amp_phase_<label>.png  # per-model amp/phase
└── metrics.json
```

**Step 3: Commit**

```bash
git add scripts/studies/README.md
git commit -m "docs: document grid-lines recon artifacts and visuals"
```

---

### Final Verification

**Run targeted tests** (adjust if any new tests added):

```bash
pytest tests/test_grid_lines_workflow.py::TestReconArtifacts -vv
pytest tests/torch/test_grid_lines_torch_runner.py::TestRunGridLinesTorchScaffold::test_runner_writes_recon_artifact -vv
pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_renders_visuals -vv
```

**Integration requirement** (production workflow touched):

```bash
pytest -m integration -v
```

---

**Plan complete.**
