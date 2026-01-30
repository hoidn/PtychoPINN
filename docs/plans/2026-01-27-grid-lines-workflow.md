# Grid Lines Workflow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a modular, self-contained grid-based workflow that reproduces the deprecated `ptycho_lines.ipynb` pipeline using the integration-test probe (64x64 and upscaled 128x128), trains PtychoPINN + Baseline, runs inference + stitching, and saves SSIM metrics + comparison artifacts.

**Architecture:** Add a workflow module under `ptycho/workflows/` that orchestrates probe prep → grid simulation → dataset persistence → training (PINN + Baseline) → inference → stitching → evaluation. Add a thin CLI wrapper under `scripts/` that runs a single (N, gridsize) combination per invocation. Use legacy grid simulation (`diffsim.mk_simdata` via `data_preprocessing.generate_data`) with explicit `update_legacy_dict(params.cfg, config)` and a local stitching helper to bypass the gridsize=1 guard.

**Tech Stack:** TensorFlow/Keras, NumPy, SciPy (`ndimage`), Matplotlib, JSON, `ptycho.*` modules (params, diffsim, data_preprocessing, train_pinn, baselines, evaluation, metadata).

---

## Scope & Constraints

- **Workflow location:** `ptycho/workflows/grid_lines_workflow.py` (module) + `scripts/studies/grid_lines_workflow.py` (thin CLI wrapper).
- **Alignment:** This workflow is the canonical harness for modular generator comparisons (see `docs/plans/2026-01-27-modular-generator-implementation.md`).
- **Probe source:** `datasets/Run1084_recon3_postPC_shrunk_3.npz` (`probeGuess` key). Produce 64x64 and upscaled 128x128 probes.
- **Probe scaling:** Default to pad+extrapolate (edge-pad amplitude + quadratic phase fit/extrapolation, then optional smoothing). Interpolation remains available via `--probe-scale-mode interpolate` (cubic spline on real/imag with `smooth_complex_array`).
- **Simulation:** Legacy grid pipeline via `data_preprocessing.generate_data()` (which calls `diffsim.mk_simdata`). Must set legacy params via `update_legacy_dict(params.cfg, config)` before invoking.
- **Config:** Match notebook defaults: `data_source='lines'`, `size=392`, `offset=4`, `outer_offset_train=8`, `outer_offset_test=20`, `nimgs_train=2`, `nimgs_test=2`, `nphotons=1e9`, `sim_jitter_scale=0.0`.
- **Runs:** Separate invocations per `N` (64 or 128) and per `gridsize` (1 or 2). Allow multiple generator architectures per call via `--architectures cnn,fno,hybrid`.
- **Mixed backends:** TF workflow runs `cnn` (PINN) + `baseline`; `fno`/`hybrid` are delegated to a Torch runner that consumes the cached NPZs and writes TF-compatible artifacts.
- **Dataset reuse:** Cache datasets under `output_dir/datasets/N{N}/gs{gridsize}/` with a manifest containing seed + sim params. If manifest matches, reuse without regeneration.
- **Baseline input:** For `gridsize > 1`, use **channel 0 only** (no flattening).
- **Metrics:** Use `ptycho.evaluation.eval_reconstruction` for SSIM/MS-SSIM; save per-run JSON + a combined JSON comparison report.
- **Artifacts:** Persist simulated NPZs under `output_dir/datasets/N{N}/gs{gridsize}/{train,test}.npz`. Each run writes NPZ predictions + stitched amplitude/phase PNGs + per-run metrics JSON under `output_dir/runs/<model_id>/`.
- **Model IDs:** Generator runs use `pinn_<arch>`; supervised baseline uses `baseline` (do not overload `cnn`).
- **Stable modules:** Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- **Stitching:** Avoid `data_preprocessing.stitch_data()` due to STITCH-GRIDSIZE-001; embed a local `stitch_predictions()` helper (based on `scripts/studies/grid_resolution_study.py`).
- **PyTorch policy:** Torch is mandatory and used via the Torch runner; do not introduce torch-optional paths.
- **Torch runner contract:** Torch uses `ptycho_torch` (torchapi-devel version) for physics/consistency; runner must emit the same artifact layout as TF under `output_dir/runs/<model_id>/` plus per-run metrics JSON for merge.

## Torch Runner Integration (Mixed Backend)
- Add a Torch runner CLI (e.g., `scripts/studies/grid_lines_torch_runner.py`) that takes cached train/test NPZs + output_dir.
- Torch runner trains/infers the requested architecture(s) (`fno`, `hybrid`) and writes artifacts/metrics matching the TF layout.
- Use `scripts/studies/grid_lines_compare_wrapper.py` as the canonical entry point to run `cnn` + `baseline` (TF) and `fno`/`hybrid` (Torch) and merge metrics into `output_dir/metrics.json`. For cross‑model comparisons, default Torch runs to MAE (`--torch-loss-mode mae`) rather than Poisson/NLL.

---

## Task 0: Test Strategy (Required Before Tests)

**Files:**
- Create: `plans/active/GRID-LINES-WORKFLOW-001/test_strategy.md`
- Reference: `plans/templates/test_strategy_template.md`

**Step 1: Copy the test strategy template**

```bash
cp plans/templates/test_strategy_template.md plans/active/GRID-LINES-WORKFLOW-001/test_strategy.md
```

**Step 2: Fill the template with this initiative’s details**

- Framework: pytest
- Unit tests only (no integration tests; avoid heavy simulation in CI)
- Mock strategy: none required; use tiny synthetic arrays and temp dirs
- **Artifact storage:** store pytest logs under `.artifacts/` and link from `plans/active/GRID-LINES-WORKFLOW-001/summary.md` (per artifact hygiene; do not create timestamped report dirs).

**Step 3: Record the test strategy link in `docs/fix_plan.md`**

Add to GRID-LINES-WORKFLOW-001 entry: `Test Strategy: plans/active/GRID-LINES-WORKFLOW-001/test_strategy.md`

---

## Task 1: Create Workflow + CLI Skeleton

**Files:**
- Create: `ptycho/workflows/grid_lines_workflow.py`
- Create: `scripts/studies/grid_lines_workflow.py`

**Step 1: Add workflow module skeleton**

```python
# ptycho/workflows/grid_lines_workflow.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import numpy as np

from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
from ptycho import params as p

@dataclass
class GridLinesConfig:
    N: int
    gridsize: int
    output_dir: Path
    probe_npz: Path
    architectures: Tuple[str, ...] = ("cnn",)
    seed: int = 0
    reuse_dataset: bool = True
    size: int = 392
    offset: int = 4
    outer_offset_train: int = 8
    outer_offset_test: int = 20
    nimgs_train: int = 2
    nimgs_test: int = 2
    nphotons: float = 1e9
    nepochs: int = 60
    batch_size: int = 16
    nll_weight: float = 0.0
    mae_weight: float = 1.0
    realspace_weight: float = 0.0
    probe_smoothing_sigma: float = 0.5


def run_grid_lines_workflow(cfg: GridLinesConfig) -> Dict[str, Any]:
    """Orchestrate probe prep → sim → train → infer → stitch → metrics."""
    raise NotImplementedError
```

**Step 2: Add thin CLI wrapper**

```python
# scripts/studies/grid_lines_workflow.py
import argparse
from pathlib import Path
from ptycho.workflows.grid_lines_workflow import GridLinesConfig, run_grid_lines_workflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True, choices=[64, 128])
    parser.add_argument("--gridsize", type=int, required=True, choices=[1, 2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-npz", type=Path, default=Path("datasets/Run1084_recon3_postPC_shrunk_3.npz"))
    parser.add_argument("--architectures", type=str, default="cnn",
                        help="Comma-separated generator list (e.g., cnn,fno,hybrid)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reuse-dataset", action="store_true", default=True)
    parser.add_argument("--nimgs-train", type=int, default=2)
    parser.add_argument("--nimgs-test", type=int, default=2)
    parser.add_argument("--nphotons", type=float, default=1e9)
    parser.add_argument("--nepochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nll-weight", type=float, default=0.0)
    parser.add_argument("--mae-weight", type=float, default=1.0)
    parser.add_argument("--realspace-weight", type=float, default=0.0)
    parser.add_argument("--probe-smoothing-sigma", type=float, default=0.5)
    args = parser.parse_args()

    cfg = GridLinesConfig(
        N=args.N,
        gridsize=args.gridsize,
        output_dir=args.output_dir,
        probe_npz=args.probe_npz,
        architectures=tuple(a.strip() for a in args.architectures.split(",") if a.strip()),
        seed=args.seed,
        reuse_dataset=args.reuse_dataset,
        nimgs_train=args.nimgs_train,
        nimgs_test=args.nimgs_test,
        nphotons=args.nphotons,
        nepochs=args.nepochs,
        batch_size=args.batch_size,
        nll_weight=args.nll_weight,
        mae_weight=args.mae_weight,
        realspace_weight=args.realspace_weight,
        probe_smoothing_sigma=args.probe_smoothing_sigma,
    )
    run_grid_lines_workflow(cfg)


if __name__ == "__main__":
    main()
```

**Step 3: Sanity check CLI**

Run: `python scripts/studies/grid_lines_workflow.py --help`
Expected: help text lists N/gridsize and loss weights.

**Step 4: Commit**

```bash
git add ptycho/workflows/grid_lines_workflow.py scripts/studies/grid_lines_workflow.py
git commit -m "feat(workflow): add grid lines workflow skeleton"
```

---

## Task 2: Probe Extraction + Upscaling Helpers

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Test: `tests/test_grid_lines_workflow.py`

**Step 1: Add probe helpers (reuse prepare_data_tool components)**

```python
from scripts.tools.prepare_data_tool import interpolate_array, smooth_complex_array


def load_probe_guess(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    if "probeGuess" not in data:
        raise KeyError("probeGuess missing from probe npz")
    return data["probeGuess"]


def scale_probe(probe: np.ndarray, target_N: int, smoothing_sigma: float) -> np.ndarray:
    if probe.shape[0] != probe.shape[1]:
        raise ValueError("probe must be square")
    if probe.shape[0] != target_N:
        zoom_factor = target_N / probe.shape[0]
        probe = interpolate_array(probe, zoom_factor)
    if smoothing_sigma and smoothing_sigma > 0:
        probe = smooth_complex_array(probe, smoothing_sigma)
    return probe.astype(np.complex64)
```

**Step 2: Add unit test for scaling**

```python
# tests/test_grid_lines_workflow.py
import numpy as np
from ptycho.workflows.grid_lines_workflow import scale_probe


def test_scale_probe_resizes_and_smooths():
    probe = (np.ones((4, 4)) + 1j * np.ones((4, 4))).astype(np.complex64)
    scaled = scale_probe(probe, target_N=8, smoothing_sigma=0.5)
    assert scaled.shape == (8, 8)
    assert scaled.dtype == np.complex64
```

**Step 3: Run test**

Run: `pytest tests/test_grid_lines_workflow.py::test_scale_probe_resizes_and_smooths -v`
Expected: PASS

**Step 4: Commit**

```bash
git add ptycho/workflows/grid_lines_workflow.py tests/test_grid_lines_workflow.py
git commit -m "feat(workflow): add probe loading and scaling helpers"
```

---

## Task 3: Simulation + Dataset Persistence

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Test: `tests/test_grid_lines_workflow.py`

**Step 1: Add config → params bridge**

```python
from ptycho import probe as probe_mod


def configure_legacy_params(cfg: GridLinesConfig, probe_np: np.ndarray, architecture: str) -> TrainingConfig:
    config = TrainingConfig(
        model=ModelConfig(N=cfg.N, gridsize=cfg.gridsize, architecture=architecture),
        nphotons=cfg.nphotons,
        nepochs=cfg.nepochs,
        batch_size=cfg.batch_size,
        nll_weight=cfg.nll_weight,
        mae_weight=cfg.mae_weight,
        realspace_weight=cfg.realspace_weight,
    )
    update_legacy_dict(p.cfg, config)
    p.set("data_source", "lines")
    p.set("size", cfg.size)
    p.set("offset", cfg.offset)
    p.set("outer_offset_train", cfg.outer_offset_train)
    p.set("outer_offset_test", cfg.outer_offset_test)
    p.set("nimgs_train", cfg.nimgs_train)
    p.set("nimgs_test", cfg.nimgs_test)
    p.set("nphotons", cfg.nphotons)
    p.set("sim_jitter_scale", 0.0)
    probe_mod.set_probe_guess(probe_guess=probe_np)
    return config
```

**Step 2: Simulate via data_preprocessing.generate_data**

```python
from ptycho import data_preprocessing


def simulate_grid_data(cfg: GridLinesConfig, probe_np: np.ndarray, architecture: str) -> Dict[str, Any]:
    configure_legacy_params(cfg, probe_np, architecture)
    X_tr, YI_tr, Yphi_tr, X_te, YI_te, Yphi_te, YY_gt, dataset, YY_full, norm_Y_I = \
        data_preprocessing.generate_data()
    return {
        "train": {
            "X": X_tr,
            "Y_I": YI_tr,
            "Y_phi": Yphi_tr,
            "coords_nominal": dataset.train_data.coords_nominal,
            "coords_true": dataset.train_data.coords_true,
            "YY_full": dataset.train_data.YY_full,
            "container": dataset.train_data,
        },
        "test": {
            "X": X_te,
            "Y_I": YI_te,
            "Y_phi": Yphi_te,
            "coords_nominal": dataset.test_data.coords_nominal,
            "coords_true": dataset.test_data.coords_true,
            "YY_full": dataset.test_data.YY_full,
            "YY_ground_truth": YY_gt,
            "norm_Y_I": norm_Y_I,
            "container": dataset.test_data,
        },
        "intensity_scale": p.get("intensity_scale"),
    }
```

**Step 3: Save NPZ with metadata**

```python
from ptycho.metadata import MetadataManager


def dataset_out_dir(cfg: GridLinesConfig) -> Path:
    return cfg.output_dir / "datasets" / f"N{cfg.N}" / f"gs{cfg.gridsize}"


def dataset_manifest_path(cfg: GridLinesConfig) -> Path:
    return dataset_out_dir(cfg) / "manifest.json"


def dataset_manifest(cfg: GridLinesConfig) -> Dict[str, Any]:
    return {
        "N": cfg.N,
        "gridsize": cfg.gridsize,
        "seed": cfg.seed,
        "nphotons": cfg.nphotons,
        "nimgs_train": cfg.nimgs_train,
        "nimgs_test": cfg.nimgs_test,
        "size": cfg.size,
        "offset": cfg.offset,
        "outer_offset_train": cfg.outer_offset_train,
        "outer_offset_test": cfg.outer_offset_test,
    }


def load_cached_dataset(cfg: GridLinesConfig) -> Optional[Dict[str, Path]]:
    if not cfg.reuse_dataset:
        return None
    out_dir = dataset_out_dir(cfg)
    manifest_path = dataset_manifest_path(cfg)
    train_path = out_dir / "train.npz"
    test_path = out_dir / "test.npz"
    if not (manifest_path.exists() and train_path.exists() and test_path.exists()):
        return None
    existing = json.loads(manifest_path.read_text())
    if existing != dataset_manifest(cfg):
        return None
    return {"train": train_path, "test": test_path}


def save_split_npz(cfg: GridLinesConfig, split: str, data: Dict[str, Any], config: TrainingConfig) -> Path:
    out_dir = dataset_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}.npz"
    payload = {
        "diffraction": data["X"],
        "Y_I": data["Y_I"],
        "Y_phi": data["Y_phi"],
        "coords_nominal": data["coords_nominal"],
        "coords_true": data["coords_true"],
        "YY_full": data["YY_full"],
        "probeGuess": data.get("probeGuess"),
    }
    if split == "test":
        payload["YY_ground_truth"] = data.get("YY_ground_truth")
        payload["norm_Y_I"] = data.get("norm_Y_I")
    metadata = MetadataManager.create_metadata(
        config, script_name="grid_lines_workflow",
        size=cfg.size, offset=cfg.offset,
        outer_offset_train=cfg.outer_offset_train,
        outer_offset_test=cfg.outer_offset_test,
        nimgs_train=cfg.nimgs_train,
        nimgs_test=cfg.nimgs_test,
    )
    MetadataManager.save_with_metadata(str(path), payload, metadata)
    return path


def write_dataset_manifest(cfg: GridLinesConfig) -> None:
    manifest_path = dataset_manifest_path(cfg)
    manifest_path.write_text(json.dumps(dataset_manifest(cfg), indent=2))
```

**Step 4: Unit test for path builder**

```python
from pathlib import Path
from ptycho.workflows.grid_lines_workflow import GridLinesConfig, dataset_out_dir


def test_dataset_out_dir_layout(tmp_path: Path):
    cfg = GridLinesConfig(N=64, gridsize=2, output_dir=tmp_path, probe_npz=Path("probe.npz"))
    assert dataset_out_dir(cfg) == tmp_path / "datasets" / "N64" / "gs2"


def test_dataset_manifest_roundtrip(tmp_path: Path):
    cfg = GridLinesConfig(N=64, gridsize=1, output_dir=tmp_path, probe_npz=Path("probe.npz"))
    write_dataset_manifest(cfg)
    cached = load_cached_dataset(cfg)
    # train/test don't exist yet, so cached should be None
    assert cached is None
```

**Step 5: Run tests**

Run: `pytest tests/test_grid_lines_workflow.py::test_dataset_out_dir_layout -v`
Expected: PASS

**Step 6: Commit**

```bash
git add ptycho/workflows/grid_lines_workflow.py tests/test_grid_lines_workflow.py
git commit -m "feat(workflow): add grid simulation + dataset persistence"
```

---

## Task 4: Stitching Helper (gridsize=1-safe)

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Test: `tests/test_grid_lines_workflow.py`

**Step 1: Add stitch_predictions helper (adapted from scripts/studies/grid_resolution_study.py)**

```python
import numpy as np


def stitch_predictions(predictions: np.ndarray, norm_Y_I: float, part: str) -> np.ndarray:
    nimgs = p.get("nimgs_test")
    outer_offset = p.get("outer_offset_test")
    N = p.get("N")
    nsegments = int(np.sqrt((predictions.size / nimgs) / (N**2)))

    if part == "amp":
        getpart = np.absolute
    elif part == "phase":
        getpart = np.angle
    else:
        getpart = lambda x: x

    img_recon = np.reshape(norm_Y_I * getpart(predictions), (-1, nsegments, nsegments, N, N, 1))
    bordersize = (N - outer_offset / 2) / 2
    borderleft = int(np.ceil(bordersize))
    borderright = int(np.floor(bordersize))
    img_recon = img_recon[:, :, :, borderleft:-borderright, borderleft:-borderright, :]
    tmp = img_recon.transpose(0, 1, 3, 2, 4, 5)
    stitched = tmp.reshape(-1, np.prod(tmp.shape[1:3]), np.prod(tmp.shape[1:3]), 1)
    return stitched
```

**Step 2: Add unit test for gridsize=1 and gridsize=2**

```python
import numpy as np
from ptycho import params as p
from ptycho.workflows.grid_lines_workflow import stitch_predictions


def test_stitch_predictions_gridsize1_and_2():
    p.set("N", 64)
    p.set("outer_offset_test", 20)

    # gridsize=1
    p.set("nimgs_test", 4)
    preds = np.random.randn(4, 64, 64, 1) + 1j * np.random.randn(4, 64, 64, 1)
    stitched = stitch_predictions(preds, norm_Y_I=1.0, part="amp")
    assert stitched.shape[-1] == 1

    # gridsize=2
    p.set("nimgs_test", 4)
    preds = np.random.randn(4, 64, 64, 4) + 1j * np.random.randn(4, 64, 64, 4)
    stitched = stitch_predictions(preds, norm_Y_I=1.0, part="amp")
    assert stitched.shape[-1] == 1
```

**Step 3: Run tests**

Run: `pytest tests/test_grid_lines_workflow.py::test_stitch_predictions_gridsize1_and_2 -v`
Expected: PASS

**Step 4: Commit**

```bash
git add ptycho/workflows/grid_lines_workflow.py tests/test_grid_lines_workflow.py
git commit -m "feat(workflow): add gridsize-safe stitching helper"
```

---

## Task 5: Training + Inference (PINN + Baseline, multi-arch)

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`

**Step 1: Add PINN training/inference helpers**

```python
from ptycho import train_pinn
from ptycho import model_manager


def train_pinn_model(train_data) -> Any:
    model, history = train_pinn.train(train_data)
    return model, history


def save_pinn_model(cfg: GridLinesConfig, model_id: str) -> None:
    out_dir = cfg.output_dir / "runs" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    model_manager.save(str(out_dir))
```

**Step 2: Add baseline training/inference helpers**

```python
from ptycho import baselines


def select_baseline_channels(X, Y_I, Y_phi) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if X.shape[-1] > 1:
        return X[..., :1], Y_I[..., :1], Y_phi[..., :1]
    return X, Y_I, Y_phi


def train_baseline_model(X_train, Y_I_train, Y_phi_train):
    Xb, YIb, Yphib = select_baseline_channels(X_train, Y_I_train, Y_phi_train)
    model, history = baselines.train(Xb, YIb, Yphib)
    return model, history


def run_baseline_inference(base_model, test_container):
    Xb, _, _ = select_baseline_channels(test_container.X, test_container.Y_I, test_container.Y_phi)
    base_amp, base_phase = base_model.predict(Xb)
    return base_amp + 1j * base_phase
```

**Step 3: Add inference + stitch path**

```python
from ptycho.evaluation import eval_reconstruction


def run_inference_and_metrics(test_data, pred_complex, norm_Y_I, YY_ground_truth, label: str):
    stitched_amp = stitch_predictions(pred_complex, norm_Y_I, part="amp")
    stitched_phase = stitch_predictions(pred_complex, norm_Y_I, part="phase")
    stitched_complex = stitched_amp * np.exp(1j * stitched_phase)
    metrics = eval_reconstruction(stitched_complex, YY_ground_truth, label=label)
    return stitched_amp, stitched_phase, metrics
```

**Step 4: Commit**

```bash
git add ptycho/workflows/grid_lines_workflow.py
git commit -m "feat(workflow): add training + inference helpers"
```

---

## Task 6: Orchestrator + Outputs

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`

**Step 1: Implement run_grid_lines_workflow**

```python
def run_grid_lines_workflow(cfg: GridLinesConfig) -> Dict[str, Any]:
    probe_guess = load_probe_guess(cfg.probe_npz)
    probe_scaled = scale_probe(probe_guess, cfg.N, cfg.probe_smoothing_sigma)

    cached = load_cached_dataset(cfg)
    if cached is None:
        config = configure_legacy_params(cfg, probe_scaled, architecture=cfg.architectures[0])
        sim = simulate_grid_data(cfg, probe_scaled, architecture=cfg.architectures[0])
        # Save datasets
        sim["train"]["probeGuess"] = probe_scaled
        sim["test"]["probeGuess"] = probe_scaled
        train_npz = save_split_npz(cfg, "train", sim["train"], config)
        test_npz = save_split_npz(cfg, "test", sim["test"], config)
        write_dataset_manifest(cfg)
    else:
        train_npz = cached["train"]
        test_npz = cached["test"]
        sim = None

    # Load data containers from cached NPZs when reused
    train_data = RawData.from_file(str(train_npz))
    test_data = RawData.from_file(str(test_npz))
    train_container = create_ptycho_data_container(train_data, config)
    test_container = create_ptycho_data_container(test_data, config)

    # Train + run PINN for each architecture
    for arch in cfg.architectures:
        model_id = f"pinn_{arch}"
        config = configure_legacy_params(cfg, probe_scaled, architecture=arch)
        pinn_model, pinn_hist = train_pinn_model(train_container)
        save_pinn_model(cfg, model_id)

        pinn_pred = pinn_model.predict([test_container.X * p.get("intensity_scale"),
                                        test_container.coords_nominal])
        pinn_amp, pinn_phase, pinn_metrics = run_inference_and_metrics(
            test_container, pinn_pred, test_container.norm_Y_I,
            test_container.YY_ground_truth, label=model_id
        )
        save_run_outputs(cfg, model_id, pinn_pred, pinn_amp, pinn_phase, pinn_metrics)

    # Train Baseline once
    base_model, base_hist = train_baseline_model(train_container.X, train_container.Y_I, train_container.Y_phi)
    base_pred = run_baseline_inference(base_model, test_container)
    base_amp, base_phase, base_metrics = run_inference_and_metrics(
        test_container, base_pred, test_container.norm_Y_I,
        test_container.YY_ground_truth, label="baseline"
    )
    save_run_outputs(cfg, "baseline", base_pred, base_amp, base_phase, base_metrics)

    return {
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
    }
```

**Step 2: Add per-run output writer**

```python
def run_dir(cfg: GridLinesConfig, model_id: str) -> Path:
    out_dir = cfg.output_dir / "runs" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_run_outputs(cfg: GridLinesConfig, model_id: str, pred_complex, amp, phase, metrics) -> None:
    out_dir = run_dir(cfg, model_id)
    np.savez(out_dir / "predictions.npz", pred_complex=pred_complex)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_comparison_pngs(out_dir, amp, phase)
```

Use Matplotlib to save 2x3 (amp/phase rows × GT/PINN/Baseline columns) for each run.

**Step 3: Commit**

```bash
git add ptycho/workflows/grid_lines_workflow.py
git commit -m "feat(workflow): implement end-to-end grid lines orchestrator"
```

---

## Task 7: Comparison Script (JSON Aggregation)

**Files:**
- Create: `scripts/studies/grid_lines_compare.py`
- Test: `tests/test_grid_lines_compare.py`

**Step 1: Implement JSON aggregator**

```python
# scripts/studies/grid_lines_compare.py
import argparse
import json
from pathlib import Path


def load_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    return json.loads(metrics_path.read_text()) if metrics_path.exists() else None


def aggregate_runs(runs_root: Path) -> dict:
    results = {}
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        metrics = load_metrics(run_dir)
        if metrics is not None:
            results[run_dir.name] = metrics
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    payload = aggregate_runs(args.runs_root)
    args.output_json.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
```

**Step 2: Add unit test**

```python
# tests/test_grid_lines_compare.py
import json
from pathlib import Path
from scripts.studies.grid_lines_compare import aggregate_runs


def test_aggregate_runs(tmp_path: Path):
    runs = tmp_path / "runs"
    (runs / "pinn_cnn").mkdir(parents=True)
    (runs / "baseline").mkdir(parents=True)
    (runs / "pinn_cnn" / "metrics.json").write_text(json.dumps({"ssim": 0.9}))
    (runs / "baseline" / "metrics.json").write_text(json.dumps({"ssim": 0.8}))

    payload = aggregate_runs(runs)
    assert payload["pinn_cnn"]["ssim"] == 0.9
    assert payload["baseline"]["ssim"] == 0.8
```

**Step 3: Run tests**

Run: `pytest tests/test_grid_lines_compare.py -v`  
Expected: PASS

**Step 4: Commit**

```bash
git add scripts/studies/grid_lines_compare.py tests/test_grid_lines_compare.py
git commit -m "feat(studies): add grid lines comparison script"
```

---

## Task 8: Smoke-Run Checklist (Manual)

**Run 1:** N=64, gridsize=1
```bash
python scripts/studies/grid_lines_workflow.py --N 64 --gridsize 1 --output-dir out/grid_lines_N64_gs1
```

**Run 2:** N=64, gridsize=2
```bash
python scripts/studies/grid_lines_workflow.py --N 64 --gridsize 2 --output-dir out/grid_lines_N64_gs2
```

**Run 3:** N=128, gridsize=1
```bash
python scripts/studies/grid_lines_workflow.py --N 128 --gridsize 1 --output-dir out/grid_lines_N128_gs1
```

**Run 4:** N=128, gridsize=2
```bash
python scripts/studies/grid_lines_workflow.py --N 128 --gridsize 2 --output-dir out/grid_lines_N128_gs2
```

**Expected outputs per run:**
- `datasets/N{N}/gs{gridsize}/train.npz`
- `datasets/N{N}/gs{gridsize}/test.npz`
- `pinn/wts.h5.zip` (via model_manager.save)
- `baseline/baseline.keras`
- `metrics.json` with SSIM/MS-SSIM for PINN + Baseline
- `visuals/compare_amp_phase.png`

---

## Testing Checklist

- Unit tests only (probe scaling, dataset path, stitching helper).
- Run: `pytest tests/test_grid_lines_workflow.py -v`
- **Record logs under `.artifacts/`** and link from `plans/active/GRID-LINES-WORKFLOW-001/summary.md` and `docs/fix_plan.md` (per artifact hygiene; no timestamped report dirs).

---

## Open Questions / Assumptions

- Assumes `data_preprocessing.generate_data()` returns `YY_ground_truth` that matches stitched output size for grid mode.
- Uses local `stitch_predictions()` instead of fixing `data_preprocessing.stitch_data()` (STITCH-GRIDSIZE-001 remains open).
- Baseline uses channel 0 only for gridsize>1 per requirement; no flattening.
- `train_pinn_model` assumes `train_data` container available; ensure `simulate_grid_data` returns `dataset` or `train_data` if needed.
