# Grid-Lines Torch External Objects Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow grid-lines Torch study wrappers to train/test on external objects (for example `fly64`) instead of synthetic lines, while keeping behavior explicit and minimizing architecture drift.

**Architecture:** Add one explicit dataset source switch in study wrappers (`synthetic_lines` vs `external_raw_npz`), then route both through a shared dataset-builder module. Reuse the existing generalization/comparison data-loading/grouping path for external datasets (RawData + grouped data generation), and add a dedicated Torch reassembly mode for external coordinate-based datasets so metrics/recons do not depend on grid-lines-only stitching metadata.

**Tech Stack:** Python 3.11, argparse, pathlib, NumPy NPZ I/O, `ptycho.workflows.components`, `ptycho.raw_data.RawData`, `ptycho.metadata.MetadataManager`, TensorFlow reassembly helper, pytest

---

### Task 1: Define Wrapper CLI Contract for External Objects (RED)

**Files:**
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`
- Test: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Write the failing test**

Add tests:

```python
def test_parse_args_accepts_external_raw_npz_mode(tmp_path):
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    args = parse_args([
        "--N", "64",
        "--gridsize", "1",
        "--output-dir", str(tmp_path),
        "--dataset-source", "external_raw_npz",
        "--train-data", "datasets/fly64/fly001_64_train_converted.npz",
        "--test-data", "datasets/fly64/fly001_64_train_converted.npz",
        "--models", "pinn_hybrid_resnet",
    ])
    assert args.dataset_source == "external_raw_npz"
    assert args.train_data.name.endswith(".npz")
    assert args.test_data.name.endswith(".npz")
```

```python
def test_parse_args_external_raw_requires_train_and_test_data(tmp_path):
    import pytest
    from scripts.studies.grid_lines_compare_wrapper import parse_args

    with pytest.raises(SystemExit):
        parse_args([
            "--N", "64",
            "--gridsize", "1",
            "--output-dir", str(tmp_path),
            "--dataset-source", "external_raw_npz",
            "--models", "pinn_hybrid_resnet",
        ])
```

```python
def test_external_raw_rejects_tf_and_ptychovit_models(tmp_path):
    import pytest
    from scripts.studies.grid_lines_compare_wrapper import run_grid_lines_compare

    with pytest.raises(ValueError, match="external_raw_npz.*Torch model IDs"):
        run_grid_lines_compare(
            N=64,
            gridsize=1,
            output_dir=tmp_path,
            probe_npz=Path("dummy_probe.npz"),
            architectures=("cnn",),
            models=("pinn",),
            dataset_source="external_raw_npz",
            train_data=Path("datasets/fly64/fly001_64_train_converted.npz"),
            test_data=Path("datasets/fly64/fly001_64_train_converted.npz"),
        )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_grid_lines_compare_wrapper.py -k "external_raw_npz_mode or external_raw" -v`
Expected: FAIL (unknown args and/or missing validation)

**Step 3: Write minimal implementation**

In `parse_args` add:
- `--dataset-source` with choices `synthetic_lines`, `external_raw_npz` (default `synthetic_lines`)
- `--train-data`, `--test-data` as `Path` args
- validation: both required when `dataset_source == external_raw_npz`

In `run_grid_lines_compare` add guardrails:
- external mode supports only Torch model IDs:
  - `pinn_fno`, `pinn_hybrid`, `pinn_stable_hybrid`, `pinn_fno_vanilla`, `pinn_hybrid_resnet`
- reject TF (`pinn`, `baseline`) and `pinn_ptychovit`
- reject multi-N overrides in external mode (single N only for phase 1)

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_grid_lines_compare_wrapper.py -k "external_raw_npz_mode or external_raw" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_compare_wrapper.py tests/test_grid_lines_compare_wrapper.py
git commit -m "feat(studies): add external_raw_npz dataset-source CLI contract"
```

---

### Task 2: Create Shared Dataset Builder for Synthetic vs External Sources (RED)

**Files:**
- Create: `scripts/studies/grid_study_dataset_builder.py`
- Create: `tests/studies/test_grid_study_dataset_builder.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Test: `tests/studies/test_grid_study_dataset_builder.py`

**Step 1: Write the failing test**

Add tests for shared builder:

```python
def test_build_synthetic_delegates_to_grid_lines_builder(monkeypatch, tmp_path):
    # monkeypatch build_grid_lines_datasets_by_n
    # assert required_ns are forwarded and train/test paths returned
```

```python
def test_build_external_raw_generates_grouped_train_test_npz(tmp_path):
    # create minimal raw train/test npz with xcoords/ycoords/diffraction/probeGuess/objectGuess
    # call build_datasets(... dataset_source="external_raw_npz")
    # assert returned train/test files exist and contain required keys:
    # diffraction, Y_I, Y_phi, coords_nominal, coords_true, coords_offsets, YY_full
```

```python
def test_build_external_raw_fails_without_object_ground_truth(tmp_path):
    # raw npz without objectGuess and without Y
    # expect ValueError mentioning required GT/object context
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/studies/test_grid_study_dataset_builder.py -v`
Expected: FAIL (module missing)

**Step 3: Write minimal implementation**

Implement in `scripts/studies/grid_study_dataset_builder.py`:

```python
@dataclass
class GridStudyDatasetBundle:
    train_npz: Path
    test_npz: Path
    gt_recon: Path
    tag: str


def build_datasets(*, dataset_source, cfg, required_ns, train_data=None, test_data=None,
                   n_groups=512, n_subsample=None, neighbor_count=7, subsample_seed=None) -> dict[int, dict[str, str]]:
    ...
```

Implementation requirements:
- `synthetic_lines`: delegate to `ptycho.workflows.grid_lines_workflow.build_grid_lines_datasets_by_n`
- `external_raw_npz`:
  - load train/test via `ptycho.workflows.components.load_data`
  - group via `RawData.generate_grouped_data` using `N`, `gridsize`, `neighbor_count`, `n_groups`, `n_subsample`
  - convert grouped data to container via `ptycho.loader.load(...)`
  - write grouped split NPZs under `output_dir/datasets/N{N}/gs{gridsize}/`
  - write metadata with `MetadataManager.save_with_metadata` and include:
    - `coords_type="relative"`
    - `dataset_source="external_raw_npz"`
    - grouping params (`n_groups`, `n_subsample`, `neighbor_count`, `subsample_seed`)
  - emit canonical GT recon from `objectGuess` (or fail if not available)

**Step 4: Run test to verify it passes**

Run: `pytest tests/studies/test_grid_study_dataset_builder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_study_dataset_builder.py tests/studies/test_grid_study_dataset_builder.py scripts/studies/grid_lines_compare_wrapper.py
git commit -m "feat(studies): add shared dataset builder for synthetic and external raw npz"
```

---

### Task 3: Add Torch Runner Reassembly Mode for External Coordinate-Based Datasets (RED)

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write the failing test**

Add tests:

```python
def test_position_reassembly_mode_uses_coords_offsets(monkeypatch, synthetic_npz, tmp_path):
    # configure TorchRunnerConfig(reassembly_mode="position")
    # patch tf_helper.reassemble_position to assert called with coords_offsets
```

```python
def test_position_reassembly_mode_requires_coords_offsets(tmp_path):
    # dataset missing coords_offsets
    # expect ValueError with actionable message
```

```python
def test_grid_lines_mode_keeps_existing_stitching_path(monkeypatch, synthetic_npz, tmp_path):
    # reassembly_mode="grid_lines"
    # assert _stitch_for_metrics still used
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py -k "reassembly_mode or coords_offsets" -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In `TorchRunnerConfig` add:

```python
reassembly_mode: str = "grid_lines"  # "grid_lines" | "position"
```

Add helper in runner:

```python
def _reassemble_with_coords_offsets(pred_complex: np.ndarray, test_data: Dict[str, np.ndarray], M: int = 20) -> np.ndarray:
    # validate coords_offsets present
    # normalize expected shape for hh.reassemble_position
    # return complex stitched image
```

In `run_grid_lines_torch` metrics block:
- if `cfg.reassembly_mode == "position"`:
  - reassemble via `coords_offsets`
  - evaluate against `YY_ground_truth` / `YY_full` / `objectGuess` (first available)
- else preserve existing `grid_lines` stitching behavior

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py -k "reassembly_mode or coords_offsets" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "feat(studies): add position reassembly mode for external torch datasets"
```

---

### Task 4: Integrate Shared Builder and Reassembly Mode into Compare Wrapper (RED)

**Files:**
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`
- Test: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Write the failing test**

Add integration tests:

```python
def test_external_raw_mode_uses_shared_builder_not_synthetic_builder(monkeypatch, tmp_path):
    # assert build_grid_lines_datasets_by_n is NOT called
    # shared builder is called
```

```python
def test_external_raw_mode_sets_torch_reassembly_mode_position(monkeypatch, tmp_path):
    # capture TorchRunnerConfig passed into run_grid_lines_torch
    # assert cfg.reassembly_mode == "position"
```

```python
def test_external_raw_mode_generates_metrics_by_model(monkeypatch, tmp_path):
    # run with models=("pinn_hybrid_resnet",)
    # assert metrics_by_model.json and metrics_table.tex created
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_grid_lines_compare_wrapper.py -k "external_raw_mode" -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In wrapper:
- replace direct dataset creation calls with `grid_study_dataset_builder.build_datasets(...)`
- feed external args to builder when `dataset_source == external_raw_npz`
- when constructing `TorchRunnerConfig`:

```python
reassembly_mode = "position" if dataset_source == "external_raw_npz" else "grid_lines"
```

- keep metrics harmonization path (`evaluate_selected_models`) unchanged

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_grid_lines_compare_wrapper.py -k "external_raw_mode" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_compare_wrapper.py tests/test_grid_lines_compare_wrapper.py
 git commit -m "feat(studies): wire compare wrapper through shared dataset builder and torch reassembly mode"
```

---

### Task 5: Add External Object Support to FNO Hyperparameter Study Wrapper (RED)

**Files:**
- Modify: `scripts/studies/fno_hyperparam_study.py`
- Modify: `tests/test_fno_hyperparam_study.py`
- Test: `tests/test_fno_hyperparam_study.py`

**Step 1: Write the failing test**

Add tests:

```python
def test_run_sweep_accepts_external_raw_npz(monkeypatch, tmp_path):
    # pass dataset_source="external_raw_npz", train/test paths
    # assert shared builder called and sweep still writes CSV
```

```python
def test_parse_args_exposes_external_source_flags(tmp_path):
    # assert parser accepts --dataset-source/--train-data/--test-data
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_fno_hyperparam_study.py -k "external_raw_npz or dataset-source" -v`
Expected: FAIL

**Step 3: Write minimal implementation**

In `fno_hyperparam_study.py`:
- add CLI args:
  - `--dataset-source` (`synthetic_lines` default)
  - `--train-data`, `--test-data`
- replace `_ensure_dataset` direct grid-lines workflow calls with shared builder
- pass `reassembly_mode="position"` to `TorchRunnerConfig` when external source is used

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_fno_hyperparam_study.py -k "external_raw_npz or dataset-source" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/fno_hyperparam_study.py tests/test_fno_hyperparam_study.py
 git commit -m "feat(studies): add external raw dataset source to fno hyperparam study"
```

---

### Task 6: Document Clear Capability Matrix to Avoid Confusion

**Files:**
- Modify: `scripts/studies/README.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `docs/COMMANDS_REFERENCE.md`
- Test: `tests/test_docs_ptychovit_workflow.py` (and existing doc checks if present)

**Step 1: Write/update failing doc test (if needed)**

If doc tests assert command examples or flags, update/add assertions for:
- `--dataset-source external_raw_npz`
- `--train-data`, `--test-data`

**Step 2: Run doc-related tests to confirm failure**

Run: `pytest tests/test_docs_ptychovit_workflow.py -v`
Expected: FAIL (if assertions added)

**Step 3: Write minimal documentation updates**

Add a concise matrix:
- `synthetic_lines`: TF + Torch + PtychoViT
- `external_raw_npz` (phase 1): Torch model IDs only

Add fly64 example:

```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 --gridsize 1 \
  --output-dir outputs/grid_lines_fly64_smoke \
  --dataset-source external_raw_npz \
  --train-data datasets/fly64/fly001_64_train_converted.npz \
  --test-data datasets/fly64/fly001_64_train_converted.npz \
  --models pinn_hybrid_resnet \
  --nepochs 10 --batch-size 16 --seed 3
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_docs_ptychovit_workflow.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/studies/README.md docs/workflows/pytorch.md docs/COMMANDS_REFERENCE.md tests/test_docs_ptychovit_workflow.py
 git commit -m "docs(studies): document external raw dataset mode and capability matrix"
```

---

### Task 7: Verification Bundle and Smoke Proof

**Files:**
- Optional Create: `.artifacts/studies/grid_lines_external_raw_smoke/README.md`
- Modify: `docs/studies/index.md` (if this index tracks study wrappers)
- Test: targeted pytest selectors

**Step 1: Prepare verification checklist artifact**

Create a short checklist recording:
- command used
- input datasets
- output artifact paths

**Step 2: Run targeted tests**

Run:

```bash
pytest tests/studies/test_grid_study_dataset_builder.py -v
pytest tests/test_grid_lines_compare_wrapper.py -k "external_raw_mode" -v
pytest tests/torch/test_grid_lines_torch_runner.py -k "reassembly_mode or coords_offsets" -v
pytest tests/test_fno_hyperparam_study.py -k "external_raw_npz or dataset-source" -v
```

Expected: PASS

**Step 3: Run end-to-end smoke command**

```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 --gridsize 1 \
  --output-dir outputs/grid_lines_external_raw_fly64_smoke \
  --dataset-source external_raw_npz \
  --train-data datasets/fly64/fly001_64_train_converted.npz \
  --test-data datasets/fly64/fly001_64_train_converted.npz \
  --models pinn_hybrid_resnet \
  --nepochs 3 --batch-size 16 --seed 3
```

Expected artifacts:
- `outputs/grid_lines_external_raw_fly64_smoke/metrics.json`
- `outputs/grid_lines_external_raw_fly64_smoke/metrics_by_model.json`
- `outputs/grid_lines_external_raw_fly64_smoke/metrics_table.tex`
- `outputs/grid_lines_external_raw_fly64_smoke/recons/pinn_hybrid_resnet/recon.npz`
- `outputs/grid_lines_external_raw_fly64_smoke/recons/gt/recon.npz`

**Step 4: Record proof paths**

Store command log and artifact listing in `.artifacts/studies/grid_lines_external_raw_smoke/README.md`.

**Step 5: Commit**

```bash
git add docs/studies/index.md .artifacts/studies/grid_lines_external_raw_smoke/README.md
 git commit -m "chore(studies): add verification evidence for external raw grid-lines torch mode"
```

---

## Rollout Notes

- Phase 1 intentionally limits external mode to Torch model IDs to avoid mixing incompatible assumptions from TF grid-lines simulation and PtychoViT bridge contracts.
- This plan avoids duplicating generalization/comparison data loading logic by reusing `ptycho.workflows.components.load_data` + grouped-data generation.
- The key anti-debt boundary is explicit source selection (`--dataset-source`), not implicit behavior based on file presence.
