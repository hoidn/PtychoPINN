# Grid-Lines PtychoViT Backend Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `pinn_ptychovit` as a selectable model arm in the grid-lines study workflow with a source-backed NPZ->HDF5 adapter, strict compatibility validation, subprocess execution, and object-space canonical harmonized metrics.

**Architecture:** Keep a strict adapter boundary: grid-lines synthetic data remains authoritative in PtychoPINN NPZ, and an interop layer converts NPZ splits into PtychoViT-compatible HDF5 inputs. Wrapper orchestration must be non-breaking for existing `--architectures` flows, while adding explicit `--models/--model-n` selection for cross-backend orchestration. Evaluation is centralized in object space: all selected model reconstructions are aligned/resized to one canonical full-object GT grid before a single metrics pass.

**Tech Stack:** Python 3, NumPy, h5py, pytest, subprocess, existing `ptycho.workflows.grid_lines_workflow`, existing `ptycho.evaluation.eval_reconstruction`, `scripts/studies/grid_lines_compare_wrapper.py`.

**Non-negotiable guardrails:**
- Preserve existing wrapper behavior and flags (`--architectures`, existing model keys, existing `metrics.json`) unless explicitly deprecated with migration coverage.
- Do not invent PtychoViT data/checkpoint contracts; document and implement from explicit upstream source references first.
- Keep PtychoViT docs out of the top-level `docs/COMMANDS_REFERENCE.md` per current scope.

**Execution discipline:**
- Use `@superpowers:test-driven-development` per task.
- Use `@superpowers:verification-before-completion` before final claims.
- Keep doc updates in the same commits as behavior changes.

### Task 0: Contract Source Gate (Must Be Green Before Code Tasks 1-6)

**Files:**
- Modify: `docs/workflows/ptychovit.md`
- Create/Modify: `tests/test_docs_ptychovit_workflow.py`

**Step 1: Write failing doc tests**

```python
# tests/test_docs_ptychovit_workflow.py

import re

def test_ptychovit_doc_records_interop_contract_source():
    text = Path("docs/workflows/ptychovit.md").read_text()
    assert "Interop Contract Source" in text
    assert "Checkpoint Contract Source" in text
    assert "source_repo" in text
    assert "source_commit" in text
    assert "TBD" not in text
    assert re.search(r"source_commit:\\s*`?[0-9a-f]{7,40}`?", text)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_docs_ptychovit_workflow.py::test_ptychovit_doc_records_interop_contract_source -v`  
Expected: FAIL if source sections/fields are missing.

**Step 3: Write minimal documentation implementation**

Update `docs/workflows/ptychovit.md` with:
- `## Interop Contract Source`
- `## Checkpoint Contract Source`
- Source record fields:
  - `source_repo`
  - `source_commit`
  - `source_paths`
  - `validated_on`
- Explicit list of required input/output datasets and checkpoint filenames derived from those sources.
- Concrete source values (no placeholders) and commit hash pinning.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_docs_ptychovit_workflow.py::test_ptychovit_doc_records_interop_contract_source -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/workflows/ptychovit.md tests/test_docs_ptychovit_workflow.py
git commit -m "docs: add ptychovit contract source-of-truth sections"
```

### Task 1: Non-Breaking Model-Selection Contract + Per-Model N Mapping

**Files:**
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Write failing tests**

```python
# tests/test_grid_lines_compare_wrapper.py

def test_wrapper_keeps_architectures_backward_compat(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args
    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--architectures", "cnn,baseline,fno",
    ])
    assert args.architectures == ("cnn", "baseline", "fno")


def test_wrapper_accepts_models_and_model_n(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args
    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--models", "pinn_hybrid,pinn_ptychovit",
        "--model-n", "pinn_hybrid=128,pinn_ptychovit=256",
    ])
    assert args.models == ("pinn_hybrid", "pinn_ptychovit")
    assert args.model_n["pinn_hybrid"] == 128
    assert args.model_n["pinn_ptychovit"] == 256


def test_wrapper_rejects_ptychovit_non_256():
    from scripts.studies.grid_lines_compare_wrapper import validate_model_specs
    with pytest.raises(ValueError, match="pinn_ptychovit.*N=256"):
        validate_model_specs(
            models=("pinn_ptychovit",),
            model_n={"pinn_ptychovit": 128},
        )


def test_compute_required_ns_from_models_and_model_n():
    from scripts.studies.grid_lines_compare_wrapper import compute_required_ns
    required = compute_required_ns(
        models=("pinn_hybrid", "pinn_ptychovit"),
        model_n={"pinn_hybrid": 128, "pinn_ptychovit": 256},
        default_n=128,
    )
    assert required == [128, 256]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grid_lines_compare_wrapper.py -k "backward_compat or models_and_model_n or ptychovit_non_256 or compute_required_ns" -v`  
Expected: FAIL with missing args/functions.

**Step 3: Write minimal implementation**

```python
# scripts/studies/grid_lines_compare_wrapper.py

LEGACY_ARCH_TO_MODEL = {
    "cnn": "pinn",
    "baseline": "baseline",
    "fno": "pinn_fno",
    "hybrid": "pinn_hybrid",
    "stable_hybrid": "pinn_stable_hybrid",
    "fno_vanilla": "pinn_fno_vanilla",
    "hybrid_resnet": "pinn_hybrid_resnet",
}

SUPPORTED_MODEL_IDS = set(LEGACY_ARCH_TO_MODEL.values()) | {"pinn_ptychovit"}

def _parse_models(value: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in value.split(",") if x.strip())

def _parse_model_n(value: str) -> dict[str, int]:
    out = {}
    for chunk in value.split(","):
        name, raw_n = chunk.split("=", 1)
        out[name.strip()] = int(raw_n)
    return out

def validate_model_specs(models: tuple[str, ...], model_n: dict[str, int]) -> None:
    for model_id in models:
        if model_id not in SUPPORTED_MODEL_IDS:
            raise ValueError(f"Unsupported model '{model_id}'")
        if model_id in model_n and model_n[model_id] <= 0:
            raise ValueError(f"Invalid N for model '{model_id}'")
    if "pinn_ptychovit" in models and model_n.get("pinn_ptychovit", 256) != 256:
        raise ValueError("pinn_ptychovit currently supports only N=256")

def compute_required_ns(models: tuple[str, ...], model_n: dict[str, int], default_n: int) -> list[int]:
    return sorted({model_n.get(model_id, default_n) for model_id in models})

```

Implementation notes:
- Keep `--architectures` fully functional.
- Add `--models` and `--model-n` as additive options, not replacements.
- Resolve selected models from `--models` when present, otherwise derive from `--architectures`.
- Build datasets per unique N in `model_n` (defaulting missing entries to top-level `N`) and route each model to the matching dataset bundle.
- Keep a single authoritative full-object GT artifact (`recons/gt/recon.npz`) across all selected model-N arms.
- Task 1 tests must stay independent of Task 5/6 symbols (`run_grid_lines_ptychovit`, `evaluate_selected_models`).

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_grid_lines_compare_wrapper.py -k "backward_compat or models_and_model_n or ptychovit_non_256 or compute_required_ns" -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_compare_wrapper.py tests/test_grid_lines_compare_wrapper.py
git commit -m "feat: add non-breaking model selection and per-model N contract"
```

### Task 2: Dataset Builders For Multi-N Synthetic Runs (Shared Full-Object GT)

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Modify: `tests/test_grid_lines_workflow.py`

**Step 1: Write the failing test**

```python
# tests/test_grid_lines_workflow.py

def test_build_grid_lines_datasets_writes_train_test_npz(tmp_path):
    from ptycho.workflows.grid_lines_workflow import GridLinesConfig, build_grid_lines_datasets
    cfg = GridLinesConfig(N=128, gridsize=1, output_dir=tmp_path, probe_npz=tmp_path / "probe.npz")
    np.savez(cfg.probe_npz, probeGuess=(np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64))
    result = build_grid_lines_datasets(cfg)
    assert Path(result["train_npz"]).exists()
    assert Path(result["test_npz"]).exists()
    assert Path(result["gt_recon"]).exists()


def test_build_grid_lines_datasets_uses_shared_canonical_gt(tmp_path):
    from ptycho.workflows.grid_lines_workflow import GridLinesConfig, build_grid_lines_datasets
    cfg128 = GridLinesConfig(N=128, gridsize=1, output_dir=tmp_path, probe_npz=tmp_path / "probe128.npz")
    cfg256 = GridLinesConfig(N=256, gridsize=1, output_dir=tmp_path, probe_npz=tmp_path / "probe256.npz")
    np.savez(cfg128.probe_npz, probeGuess=(np.ones((128, 128)) + 1j * np.ones((128, 128))).astype(np.complex64))
    np.savez(cfg256.probe_npz, probeGuess=(np.ones((256, 256)) + 1j * np.ones((256, 256))).astype(np.complex64))
    out128 = build_grid_lines_datasets(cfg128, dataset_tag="N128", canonical_gt_label="gt")
    out256 = build_grid_lines_datasets(cfg256, dataset_tag="N256", canonical_gt_label="gt")
    assert out128["gt_recon"] == out256["gt_recon"]
    assert out128["gt_recon"].endswith("recons/gt/recon.npz")


def test_build_grid_lines_datasets_by_n_builds_each_required_n(monkeypatch, tmp_path):
    from ptycho.workflows.grid_lines_workflow import GridLinesConfig, build_grid_lines_datasets_by_n
    called = []

    def fake_build(cfg, dataset_tag=None):
        called.append((cfg.N, dataset_tag))
        return {"train_npz": "train.npz", "test_npz": "test.npz", "gt_recon": f"recons/gt_{dataset_tag}/recon.npz", "tag": dataset_tag}

    monkeypatch.setattr("ptycho.workflows.grid_lines_workflow.build_grid_lines_datasets", fake_build)
    base_cfg = GridLinesConfig(N=128, gridsize=1, output_dir=tmp_path, probe_npz=tmp_path / "probe.npz")
    out = build_grid_lines_datasets_by_n(base_cfg, required_ns=[256, 128, 256])
    assert sorted(out.keys()) == [128, 256]
    assert called == [(128, "N128"), (256, "N256")]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_grid_lines_workflow.py -k "build_grid_lines_datasets_writes_train_test_npz or uses_shared_canonical_gt or by_n_builds_each_required_n" -v`  
Expected: FAIL with missing function.

**Step 3: Write minimal implementation**

```python
# ptycho/workflows/grid_lines_workflow.py

from dataclasses import replace

def build_grid_lines_datasets(
    cfg: GridLinesConfig,
    dataset_tag: str | None = None,
    canonical_gt_label: str = "gt",
) -> Dict[str, str]:
    probe_guess = load_ideal_disk_probe(cfg.N) if cfg.probe_source == "ideal_disk" else load_probe_guess(cfg.probe_npz)
    probe_scaled = scale_probe(probe_guess, cfg.N, cfg.probe_smoothing_sigma, scale_mode=cfg.probe_scale_mode)
    probe_scaled = apply_probe_mask(probe_scaled, cfg.probe_mask_diameter)
    sim = simulate_grid_data(cfg, probe_scaled)
    config = configure_legacy_params(cfg, probe_scaled)
    sim["train"]["probeGuess"] = probe_scaled
    sim["test"]["probeGuess"] = probe_scaled
    train_npz = save_split_npz(cfg, "train", sim["train"], config)
    test_npz = save_split_npz(cfg, "test", sim["test"], config)
    tag = dataset_tag or f"N{cfg.N}"
    gt_path = cfg.output_dir / "recons" / canonical_gt_label / "recon.npz"
    gt_complex = np.squeeze(sim["test"]["YY_ground_truth"])
    if gt_path.exists():
        existing_gt = np.load(gt_path)["YY_pred"]
        if not np.allclose(np.squeeze(existing_gt), gt_complex, rtol=1e-6, atol=1e-6):
            raise ValueError("Canonical GT mismatch across N builds; enforce shared synthetic object identity/seed")
    else:
        gt_path = save_recon_artifact(cfg.output_dir, canonical_gt_label, gt_complex)
    return {"train_npz": str(train_npz), "test_npz": str(test_npz), "gt_recon": str(gt_path), "tag": tag}

def build_grid_lines_datasets_by_n(base_cfg: GridLinesConfig, required_ns: Iterable[int]) -> dict[int, Dict[str, str]]:
    bundles = {}
    for n in sorted(set(required_ns)):
        cfg_n = replace(base_cfg, N=n)
        bundles[n] = build_grid_lines_datasets(cfg_n, dataset_tag=f"N{n}", canonical_gt_label="gt")
    return bundles
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_grid_lines_workflow.py -k "build_grid_lines_datasets_writes_train_test_npz or uses_shared_canonical_gt or by_n_builds_each_required_n" -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho/workflows/grid_lines_workflow.py tests/test_grid_lines_workflow.py
git commit -m "feat: add dataset-only grid-lines builder for multi-N orchestration"
```

### Task 3: PtychoViT Interop Contract + NPZ->HDF5 Adapter (Source-Backed)

**Files:**
- Create: `ptycho/interop/ptychovit/__init__.py`
- Create: `ptycho/interop/ptychovit/contracts.py`
- Create: `ptycho/interop/ptychovit/convert.py`
- Modify: `docs/workflows/ptychovit.md`
- Create: `tests/test_ptychovit_adapter.py`

**Step 1: Write failing tests**

```python
# tests/test_ptychovit_adapter.py

def test_contract_defines_required_hdf5_keys():
    from ptycho.interop.ptychovit.contracts import REQUIRED_DP_KEYS, REQUIRED_PARA_KEYS
    assert "dp" in REQUIRED_DP_KEYS
    assert "object" in REQUIRED_PARA_KEYS
    assert "probe" in REQUIRED_PARA_KEYS
    assert "probe_position_x_m" in REQUIRED_PARA_KEYS
    assert "probe_position_y_m" in REQUIRED_PARA_KEYS


def test_convert_grid_lines_npz_to_ptychovit_hdf5(tmp_path):
    from ptycho.interop.ptychovit.convert import convert_npz_split_to_hdf5_pair
    out = convert_npz_split_to_hdf5_pair(...)
    assert out.dp_hdf5.exists()
    assert out.para_hdf5.exists()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ptychovit_adapter.py -v`  
Expected: FAIL with import errors.

**Step 3: Write minimal implementation**

```python
# ptycho/interop/ptychovit/contracts.py

REQUIRED_DP_KEYS = ("dp",)
REQUIRED_PARA_KEYS = ("object", "probe", "probe_position_x_m", "probe_position_y_m")

@dataclass(frozen=True)
class PtychoViTHdf5Pair:
    dp_hdf5: Path
    para_hdf5: Path
    object_name: str
```

```python
# ptycho/interop/ptychovit/convert.py

def convert_npz_split_to_hdf5_pair(npz_path: Path, out_dir: Path, object_name: str, pixel_size_m: float = 1.0) -> PtychoViTHdf5Pair:
    data = dict(np.load(npz_path, allow_pickle=True))
    diffraction_amp = np.asarray(data["diffraction"], dtype=np.float32)
    if diffraction_amp.ndim == 4:
        diffraction_amp = diffraction_amp[..., 0]
    dp = np.square(np.clip(diffraction_amp, a_min=0.0, a_max=None)).astype(np.float32)

    gt_complex = np.squeeze(np.asarray(data.get("YY_ground_truth", data["YY_full"]))).astype(np.complex64)
    probe = np.asarray(data["probeGuess"]).astype(np.complex64)
    coords = np.asarray(data["coords_nominal"])
    y = coords[:, 0] if coords.ndim == 2 else coords[:, 0, 0, 0]
    x = coords[:, 1] if coords.ndim == 2 else coords[:, 0, 1, 0]

    dp_path = out_dir / f"{object_name}_dp.hdf5"
    para_path = out_dir / f"{object_name}_para.hdf5"
    with h5py.File(dp_path, "w") as f:
        f.create_dataset("dp", data=dp)
    with h5py.File(para_path, "w") as f:
        f.create_dataset("object", data=gt_complex[np.newaxis, ...])
        f["object"].attrs["pixel_height_m"] = float(pixel_size_m)
        f["object"].attrs["pixel_width_m"] = float(pixel_size_m)
        f.create_dataset("probe", data=probe[np.newaxis, ...])
        f["probe"].attrs["pixel_height_m"] = float(pixel_size_m)
        f["probe"].attrs["pixel_width_m"] = float(pixel_size_m)
        f.create_dataset("probe_position_x_m", data=np.asarray(x, dtype=np.float64) * pixel_size_m)
        f.create_dataset("probe_position_y_m", data=np.asarray(y, dtype=np.float64) * pixel_size_m)
    return PtychoViTHdf5Pair(dp_hdf5=dp_path, para_hdf5=para_path, object_name=object_name)
```

Documentation requirement in same commit:
- Update `docs/workflows/ptychovit.md` with the exact dataset list and source citation fields from Task 0.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ptychovit_adapter.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho/interop/ptychovit/__init__.py ptycho/interop/ptychovit/contracts.py ptycho/interop/ptychovit/convert.py docs/workflows/ptychovit.md tests/test_ptychovit_adapter.py
git commit -m "feat: add source-backed ptychovit npz-to-hdf5 adapter contract"
```

### Task 4: Strict PtychoViT Compatibility Validator

**Files:**
- Create: `ptycho/interop/ptychovit/validate.py`
- Modify: `tests/test_ptychovit_adapter.py`

**Step 1: Write failing tests**

```python
# tests/test_ptychovit_adapter.py

def test_validate_hdf5_pair_rejects_missing_probe_positions(tmp_path):
    from ptycho.interop.ptychovit.validate import validate_hdf5_pair
    with pytest.raises(ValueError, match="probe_position"):
        validate_hdf5_pair(dp_path, para_path)


def test_validate_hdf5_pair_rejects_mismatched_position_lengths(tmp_path):
    from ptycho.interop.ptychovit.validate import validate_hdf5_pair
    with pytest.raises(ValueError, match="same length"):
        validate_hdf5_pair(dp_path, para_path)


def test_validate_hdf5_pair_rejects_scan_count_mismatch(tmp_path):
    from ptycho.interop.ptychovit.validate import validate_hdf5_pair
    with pytest.raises(ValueError, match="dp scan count"):
        validate_hdf5_pair(dp_path, para_path)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ptychovit_adapter.py -k "validate_hdf5_pair" -v`  
Expected: FAIL with missing function.

**Step 3: Write minimal implementation**

```python
# ptycho/interop/ptychovit/validate.py

def validate_hdf5_pair(dp_hdf5: Path, para_hdf5: Path) -> None:
    with h5py.File(dp_hdf5, "r") as dp, h5py.File(para_hdf5, "r") as para:
        if "dp" not in dp:
            raise ValueError("Missing required dataset 'dp'")
        for key in ("object", "probe", "probe_position_x_m", "probe_position_y_m"):
            if key not in para:
                raise ValueError(f"Missing required dataset '{key}'")
        x = np.asarray(para["probe_position_x_m"])
        y = np.asarray(para["probe_position_y_m"])
        dp_arr = np.asarray(dp["dp"])
        if x.shape[0] != y.shape[0]:
            raise ValueError("probe_position_x_m and probe_position_y_m must have same length")
        if dp_arr.shape[0] != x.shape[0]:
            raise ValueError("dp scan count must match probe position vector length")
        if dp_arr.ndim != 3:
            raise ValueError("dp must be rank-3 [N,H,W]")
        if not np.issubdtype(dp_arr.dtype, np.floating):
            raise ValueError("dp must be float dtype")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("probe positions must be finite")
        for ds_name in ("object", "probe"):
            ds = para[ds_name]
            if "pixel_height_m" not in ds.attrs or "pixel_width_m" not in ds.attrs:
                raise ValueError(f"{ds_name} attrs missing pixel size")
```

Validation policy:
- Key existence
- Position vector shape/length consistency
- `len(dp scans) == len(position vectors)` consistency
- Required pixel-size attrs on `object` and `probe`
- Finite-valued numeric position arrays
- Rank/dtype checks for `dp`

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ptychovit_adapter.py -k "validate_hdf5_pair" -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho/interop/ptychovit/validate.py tests/test_ptychovit_adapter.py
git commit -m "feat: add strict ptychovit hdf5 compatibility validator"
```

### Task 5: Subprocess PtychoViT Runner (Inference + Fine-Tune Entrypoints)

**Files:**
- Create: `scripts/studies/grid_lines_ptychovit_runner.py`
- Create: `scripts/studies/ptychovit_bridge_entrypoint.py`
- Create: `tests/test_grid_lines_ptychovit_runner.py`

**Step 1: Write failing tests**

```python
# tests/test_grid_lines_ptychovit_runner.py

def test_runner_invokes_subprocess_with_resolved_paths(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_ptychovit_runner import run_grid_lines_ptychovit
    captured = {}
    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_grid_lines_ptychovit(...)
    assert captured["cmd"][0] == "python"
    assert result["status"] == "ok"


def test_runner_returns_recon_npz_for_metrics_handoff(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_ptychovit_runner import run_grid_lines_ptychovit
    recon_path = tmp_path / "recons" / "pinn_ptychovit" / "recon.npz"
    recon_path.parent.mkdir(parents=True, exist_ok=True)
    recon_path.write_bytes(b"stub")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="", stderr=""))
    result = run_grid_lines_ptychovit(...)
    assert result["recon_npz"] == str(recon_path)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grid_lines_ptychovit_runner.py -v`  
Expected: FAIL with import error.

**Step 3: Write minimal implementation**

```python
# scripts/studies/grid_lines_ptychovit_runner.py

@dataclass
class PtychoViTRunnerConfig:
    ptychovit_repo: Path
    output_dir: Path
    train_dp: Path
    test_dp: Path
    model_n: int = 256
    mode: str = "inference"  # inference|finetune

def run_grid_lines_ptychovit(cfg: PtychoViTRunnerConfig) -> dict:
    if cfg.model_n != 256:
        raise ValueError("pinn_ptychovit currently supports only N=256")
    logs_dir = cfg.output_dir / "runs" / "pinn_ptychovit"
    logs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "scripts/studies/ptychovit_bridge_entrypoint.py",
        "--ptychovit-repo", str(cfg.ptychovit_repo),
        "--train-dp", str(cfg.train_dp),
        "--test-dp", str(cfg.test_dp),
        "--mode", cfg.mode,
        "--output-dir", str(logs_dir),
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    (logs_dir / "stdout.log").write_text(completed.stdout)
    (logs_dir / "stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"ptychovit subprocess failed (exit={completed.returncode})")
    recon_npz = cfg.output_dir / "recons" / "pinn_ptychovit" / "recon.npz"
    if not recon_npz.exists():
        raise RuntimeError(f"ptychovit subprocess succeeded but recon artifact missing: {recon_npz}")
    return {"status": "ok", "run_dir": str(logs_dir), "model_id": "pinn_ptychovit", "recon_npz": str(recon_npz)}
```

Runner handoff contract (mandatory):
- `run_grid_lines_ptychovit` must return a `recon_npz` path.
- Wrapper must consume `recon_npz` into `recon_paths["pinn_ptychovit"]` before calling `evaluate_selected_models`.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_grid_lines_ptychovit_runner.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_ptychovit_runner.py scripts/studies/ptychovit_bridge_entrypoint.py tests/test_grid_lines_ptychovit_runner.py
git commit -m "feat: add subprocess ptychovit runner for studies"
```

### Task 6: Canonical Full-Object-Grid Harmonized Metrics For Selected Models

**Files:**
- Create: `ptycho/image/harmonize.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Write failing tests**

```python
# tests/test_grid_lines_compare_wrapper.py

def test_wrapper_writes_metrics_by_model_for_selected_models(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare
    run_grid_lines_compare(...)
    metrics = json.loads((tmp_path / "metrics_by_model.json").read_text())
    assert "pinn_hybrid" in metrics
    assert "pinn_ptychovit" in metrics


def test_harmonized_metrics_run_on_canonical_gt_grid(monkeypatch, tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import evaluate_selected_models
    out = evaluate_selected_models(...)
    assert out["pinn_hybrid"]["reference_shape"] == [392, 392]  # example: matches GT object grid
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grid_lines_compare_wrapper.py -k "metrics_by_model or canonical_gt_grid" -v`  
Expected: FAIL with missing file/functions.

**Step 3: Write minimal implementation**

```python
# ptycho/image/harmonize.py

from scipy.ndimage import zoom

def resize_complex_to_shape(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    z = np.squeeze(arr).astype(np.complex64)
    if z.ndim != 2:
        raise ValueError("resize_complex_to_shape expects 2D complex input")
    if z.shape == target_hw:
        return z
    zoom_factors = (target_hw[0] / z.shape[0], target_hw[1] / z.shape[1])
    real = zoom(z.real, zoom_factors, order=3)
    imag = zoom(z.imag, zoom_factors, order=3)
    return (real + 1j * imag).astype(np.complex64)
```

```python
# scripts/studies/grid_lines_compare_wrapper.py

def evaluate_selected_models(recon_paths: dict[str, Path], gt_path: Path) -> dict:
    gt = np.load(gt_path)["YY_pred"]
    gt_ref = np.squeeze(gt).astype(np.complex64)
    target_hw = tuple(gt_ref.shape)
    out = {}
    for model_id, recon_path in recon_paths.items():
        pred = np.load(recon_path)["YY_pred"]
        pred_ref = resize_complex_to_shape(pred, target_hw)
        metrics = eval_reconstruction(pred_ref[..., None], gt_ref[..., None], label=model_id)
        out[model_id] = {"reference_shape": [target_hw[0], target_hw[1]], "metrics": metrics}
    return out
```

Compatibility requirement:
- Continue writing legacy `metrics.json` for current users.
- Also write new `metrics_by_model.json` for selected-model orchestration.
- Build `recon_paths` exclusively from runner return contracts (`recon_npz`) and the single canonical full-object GT artifact (`recons/gt/recon.npz`).
- Canonicalization target is GT object grid shape (real-space object), not diffraction patch size `N`.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_grid_lines_compare_wrapper.py -k "metrics_by_model or canonical_gt_grid" -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho/image/harmonize.py scripts/studies/grid_lines_compare_wrapper.py tests/test_grid_lines_compare_wrapper.py
git commit -m "feat: add canonical object-grid harmonized metrics for selected model arms"
```

### Task 7: Documentation Publication (No Top-Level Commands Reference Changes)

**Files:**
- Modify: `docs/index.md`
- Modify: `scripts/studies/README.md`
- Modify: `docs/workflows/ptychovit.md`
- Modify: `tests/test_docs_ptychovit_workflow.py`

**Step 1: Write failing tests**

```python
# tests/test_docs_ptychovit_workflow.py

def test_docs_index_links_ptychovit_workflow():
    text = Path("docs/index.md").read_text()
    assert "workflows/ptychovit.md" in text

def test_studies_readme_points_to_workflow_doc():
    text = Path("scripts/studies/README.md").read_text()
    assert "docs/workflows/ptychovit.md" in text
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_docs_ptychovit_workflow.py -v`  
Expected: FAIL before discoverability updates.

**Step 3: Write minimal docs implementation**

- `docs/index.md`: add an entry for `workflows/ptychovit.md`.
- `scripts/studies/README.md`: add a short pointer under grid-lines wrapper docs.
- `docs/workflows/ptychovit.md`: ensure sections cover restore/training/fine-tune/inference runbook and troubleshooting.
- Do not modify `docs/COMMANDS_REFERENCE.md` in this initiative.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_docs_ptychovit_workflow.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/index.md scripts/studies/README.md docs/workflows/ptychovit.md tests/test_docs_ptychovit_workflow.py
git commit -m "docs: publish ptychovit workflow in index and studies guide"
```

### Task 8: End-to-End Verification Bundle

**Files:**
- Modify: `docs/plans/2026-02-10-grid-lines-ptychovit-backend-integration.md` (append evidence links)
- Create: `.artifacts/ptychovit_integration/README.md` (gitignored; artifact pointer only)

**Step 1: Run targeted tests**

Run:
- `pytest tests/test_grid_lines_compare_wrapper.py -v`
- `pytest tests/test_grid_lines_workflow.py -k datasets -v`
- `pytest tests/test_ptychovit_adapter.py -v`
- `pytest tests/test_grid_lines_ptychovit_runner.py -v`
- `pytest tests/test_docs_ptychovit_workflow.py -v`

Expected: PASS.

**Step 2: Run dry integration command**

Run:
- `python scripts/studies/grid_lines_compare_wrapper.py --N 128 --gridsize 1 --output-dir tmp/ptychovit_smoke --architectures hybrid --models pinn_hybrid,pinn_ptychovit --model-n pinn_hybrid=128,pinn_ptychovit=256 --set-phi`

Expected:
- `metrics_by_model.json`
- `metrics.json` (legacy compatibility)
- `recons/gt/recon.npz` (single canonical full-object GT artifact)
- Harmonization target is GT object grid shape from `recons/gt/recon.npz` (independent of diffraction `N`)
- selected model recon folders and run logs

**Step 3: Archive logs and note paths**

Run:
- `mkdir -p .artifacts/ptychovit_integration`
- copy pytest logs and smoke-run logs

Expected: reproducible evidence inventory.

**Step 4: Commit plan evidence update**

```bash
git add docs/plans/2026-02-10-grid-lines-ptychovit-backend-integration.md
git commit -m "chore: record verification evidence for ptychovit backend integration"
```

### Task 9: Full Inference + Comparison Execution (Non-Smoke)

**Files:**
- Modify: `docs/plans/2026-02-10-grid-lines-ptychovit-backend-integration.md` (append evidence links)
- Create: `.artifacts/ptychovit_integration/full_inference_comparison/README.md` (gitignored; artifact pointer only)

**Step 1: Execute a full selected-model run (not smoke-sized)**

Run:
- `python scripts/studies/grid_lines_compare_wrapper.py --N 128 --gridsize 1 --output-dir tmp/ptychovit_full_infer --architectures hybrid --models pinn_hybrid,pinn_ptychovit --model-n pinn_hybrid=128,pinn_ptychovit=256 --set-phi --nimgs-train 8 --nimgs-test 8 --torch-epochs 120`

Expected:
- Both model arms execute end-to-end (no skipped model).
- `tmp/ptychovit_full_infer/metrics_by_model.json` exists and includes `pinn_hybrid` and `pinn_ptychovit`.
- `tmp/ptychovit_full_infer/recons/pinn_hybrid/recon.npz` exists.
- `tmp/ptychovit_full_infer/recons/pinn_ptychovit/recon.npz` exists.
- `tmp/ptychovit_full_infer/recons/gt/recon.npz` exists.

**Step 2: Validate metric payload integrity**

Run:
- `python - <<'PY'\nimport json, math\nfrom pathlib import Path\nm = json.loads(Path('tmp/ptychovit_full_infer/metrics_by_model.json').read_text())\nfor mid in ('pinn_hybrid','pinn_ptychovit'):\n    assert mid in m, f'missing model: {mid}'\n    metrics = m[mid]['metrics']\n    for key in ('mae','mse','psnr','ssim','ms_ssim','frc50'):\n        assert key in metrics, f'{mid} missing metric {key}'\n        vals = metrics[key]\n        for v in vals:\n            assert v is None or (isinstance(v,(int,float)) and math.isfinite(v)), f'non-finite {mid}:{key}:{v}'\nprint('metrics payload OK')\nPY`

Expected: script exits 0 with `metrics payload OK`.

**Step 3: Validate visual sanity artifacts**

Run:
- `test -f tmp/ptychovit_full_infer/visuals/compare_amp_phase.png`
- `test -f tmp/ptychovit_full_infer/visuals/amp_phase_pinn_ptychovit.png`
- `python - <<'PY'\nimport numpy as np\nfrom pathlib import Path\np = Path('tmp/ptychovit_full_infer/recons/pinn_ptychovit/recon.npz')\nd = np.load(p)\namp = d['amp']\nphase = d['phase']\nassert np.isfinite(amp).all() and np.isfinite(phase).all()\nassert float(amp.std()) > 0.0, 'amplitude collapsed'\nassert float(phase.std()) > 0.0, 'phase collapsed'\nprint('visual sanity OK')\nPY`

Expected: visual files exist and script prints `visual sanity OK`.

**Step 4: Archive full-inference evidence**

Run:
- `mkdir -p .artifacts/ptychovit_integration/full_inference_comparison`
- copy command logs, `metrics_by_model.json`, and visual sanity outputs into artifact folder (or record durable external path in README).

Expected: reproducible evidence index for full inference + comparison.

**Step 5: Commit plan evidence update**

```bash
git add docs/plans/2026-02-10-grid-lines-ptychovit-backend-integration.md
git commit -m "chore: record full inference and comparison evidence"
```

### Task 10: Full Fine-Tuning Execution + Post-Tune Comparison

**Files:**
- Modify: `docs/plans/2026-02-10-grid-lines-ptychovit-backend-integration.md` (append evidence links)
- Create: `.artifacts/ptychovit_integration/full_finetune/README.md` (gitignored; artifact pointer only)

**Step 1: Execute full PtychoViT fine-tune run**

Run:
- `python scripts/studies/ptychovit_bridge_entrypoint.py --ptychovit-repo <path-to-ptychovit> --mode finetune --train-dp tmp/ptychovit_full_infer/interop/train_dp.hdf5 --test-dp tmp/ptychovit_full_infer/interop/test_dp.hdf5 --output-dir tmp/ptychovit_full_finetune/runs/pinn_ptychovit --resume-from-checkpoint false`

Expected:
- Fine-tune run exits 0.
- Checkpoint artifacts exist under `tmp/ptychovit_full_finetune/runs/pinn_ptychovit/`:
  - `best_model.pth`
  - `checkpoint_model.pth`
  - `checkpoint.state`

**Step 2: Run post-fine-tune inference using fine-tuned checkpoint**

Run:
- `python scripts/studies/ptychovit_bridge_entrypoint.py --ptychovit-repo <path-to-ptychovit> --mode inference --train-dp tmp/ptychovit_full_infer/interop/train_dp.hdf5 --test-dp tmp/ptychovit_full_infer/interop/test_dp.hdf5 --output-dir tmp/ptychovit_full_finetune/infer --checkpoint tmp/ptychovit_full_finetune/runs/pinn_ptychovit/best_model.pth`

Expected:
- `tmp/ptychovit_full_finetune/infer/recons/pinn_ptychovit/recon.npz` exists.
- inference logs are persisted (stdout/stderr).

**Step 3: Run post-tune comparison against selected model set**

Run:
- `python scripts/studies/grid_lines_compare_wrapper.py --N 128 --gridsize 1 --output-dir tmp/ptychovit_full_posttune_compare --architectures hybrid --models pinn_hybrid,pinn_ptychovit --model-n pinn_hybrid=128,pinn_ptychovit=256 --set-phi --nimgs-train 8 --nimgs-test 8 --torch-epochs 120`

Expected:
- `tmp/ptychovit_full_posttune_compare/metrics_by_model.json` exists.
- includes both `pinn_hybrid` and `pinn_ptychovit`.

**Step 4: Record pre/post fine-tune metric deltas for PtychoViT**

Run:
- `python - <<'PY'\nimport json\nfrom pathlib import Path\npre = json.loads(Path('tmp/ptychovit_full_infer/metrics_by_model.json').read_text())['pinn_ptychovit']['metrics']\npost = json.loads(Path('tmp/ptychovit_full_posttune_compare/metrics_by_model.json').read_text())['pinn_ptychovit']['metrics']\nsummary = {\n  'ms_ssim_amp_pre': pre['ms_ssim'][0],\n  'ms_ssim_amp_post': post['ms_ssim'][0],\n  'mae_amp_pre': pre['mae'][0],\n  'mae_amp_post': post['mae'][0],\n}\nPath('tmp/ptychovit_full_posttune_compare/ptychovit_pre_post_delta.json').write_text(json.dumps(summary, indent=2))\nprint(json.dumps(summary, indent=2))\nPY`

Expected:
- `ptychovit_pre_post_delta.json` exists for audit.
- values are finite.

**Step 5: Archive full fine-tune evidence and commit plan update**

Run:
- `mkdir -p .artifacts/ptychovit_integration/full_finetune`
- copy checkpoints/logs/comparison metrics or record durable external path in README.

Then:

```bash
git add docs/plans/2026-02-10-grid-lines-ptychovit-backend-integration.md
git commit -m "chore: record full fine-tune and post-tune comparison evidence"
```

## Notes On Non-Goals (YAGNI)
- No physical-unit harmonization in v1 (pixel-space canonicalization only).
- No support for PtychoViT at `N!=256` in v1.
- No invasive edits to external PtychoViT repository internals.

## Expected Outputs
- `output_dir/metrics_by_model.json` (selected models)
- `output_dir/metrics.json` (legacy compatibility)
- `output_dir/recons/<model_id>/recon.npz`
- `output_dir/runs/<model_id>/` logs and manifests
- `docs/workflows/ptychovit.md` as the authoritative backend workflow guide
- `.artifacts/ptychovit_integration/full_inference_comparison/` evidence index
- `.artifacts/ptychovit_integration/full_finetune/` evidence index

## Verification Evidence (2026-02-09)

- Pytest logs:
  - `.artifacts/ptychovit_integration/pytest_grid_lines_compare_wrapper.log`
  - `.artifacts/ptychovit_integration/pytest_grid_lines_workflow_datasets.log`
  - `.artifacts/ptychovit_integration/pytest_ptychovit_adapter.log`
  - `.artifacts/ptychovit_integration/pytest_grid_lines_ptychovit_runner.log`
  - `.artifacts/ptychovit_integration/pytest_docs_ptychovit_workflow.log`
- Smoke command log:
  - `.artifacts/ptychovit_integration/smoke_grid_lines_compare_wrapper.log`
- Smoke artifacts:
  - `tmp/ptychovit_smoke/metrics_by_model.json`
  - `tmp/ptychovit_smoke/metrics.json`
  - `tmp/ptychovit_smoke/recons/gt/recon.npz`
  - `tmp/ptychovit_smoke/recons/pinn_hybrid/recon.npz`
  - `tmp/ptychovit_smoke/recons/pinn_ptychovit/recon.npz`
  - `tmp/ptychovit_smoke/runs/pinn_hybrid/stdout.log`
  - `tmp/ptychovit_smoke/runs/pinn_ptychovit/stdout.log`

## Verification Evidence (2026-02-10 Task 9/10)

- Full inference + comparison evidence index:
  - `.artifacts/ptychovit_integration/full_inference_comparison/README.md`
- Full inference logs:
  - `.artifacts/ptychovit_integration/full_inference_comparison/full_infer_command.log`
  - `.artifacts/ptychovit_integration/full_inference_comparison/metrics_payload_check.log`
  - `.artifacts/ptychovit_integration/full_inference_comparison/visual_sanity_check.log`
- Full inference artifacts:
  - `tmp/ptychovit_full_infer/metrics_by_model.json`
  - `tmp/ptychovit_full_infer/metrics.json`
  - `tmp/ptychovit_full_infer/recons/gt/recon.npz`
  - `tmp/ptychovit_full_infer/recons/pinn_hybrid/recon.npz`
  - `tmp/ptychovit_full_infer/recons/pinn_ptychovit/recon.npz`

- Full fine-tune + post-tune evidence index:
  - `.artifacts/ptychovit_integration/full_finetune/README.md`
- Full fine-tune logs:
  - `.artifacts/ptychovit_integration/full_finetune/interop_resize_to_256.log`
  - `.artifacts/ptychovit_integration/full_finetune/finetune_command.log`
  - `.artifacts/ptychovit_integration/full_finetune/inference_with_checkpoint.log`
  - `.artifacts/ptychovit_integration/full_finetune/posttune_compare_command.log`
  - `.artifacts/ptychovit_integration/full_finetune/pre_post_delta.log`
- Full fine-tune artifacts:
  - `tmp/ptychovit_full_finetune/runs/pinn_ptychovit/best_model.pth`
  - `tmp/ptychovit_full_finetune/runs/pinn_ptychovit/checkpoint_model.pth`
  - `tmp/ptychovit_full_finetune/runs/pinn_ptychovit/checkpoint.state`
  - `tmp/ptychovit_full_finetune/infer/recons/pinn_ptychovit/recon.npz`
  - `tmp/ptychovit_full_posttune_compare/metrics_by_model.json`
  - `tmp/ptychovit_full_posttune_compare/ptychovit_pre_post_delta.json`
