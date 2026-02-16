# FLY001 N=128 External Study Reproducibility Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the `N=128` fly001 external study reproducible and discoverable by adding a canonicalization/split pipeline, registering the study runbook, and updating dataset/study documentation.

**Architecture:** Add one deterministic dataset-prep script that converts raw fly001-128 NPZ into canonical NPZ and creates disjoint top/bottom splits plus a manifest. Use those split files as the fixed inputs for a dedicated study launcher under `.artifacts/studies/...`. Register the launcher in `docs/studies/index.md` and add dataset-level docs linked from `docs/index.md` and command recipes.

**Tech Stack:** Python 3.11, NumPy NPZ I/O, existing `transpose_rename_convert_tool` semantics, pytest, shell launcher scripts, Markdown docs.

---

## Preflight Constraints

- Use `python` from PATH (PYTHON-ENV-001).
- Do not modify stable physics/model files (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Follow TDD with small commits (`@superpowers:test-driven-development`, `@superpowers:verification-before-completion`).
- Keep artifacts lightweight in git; store executable runbook in `.artifacts/studies/...` and document outputs in docs.

### Task 1: Build Deterministic N=128 Prep Script (Canonicalize + Top/Bottom Split + Manifest)

**Files:**
- Create: `scripts/studies/prepare_fly001_128_external_split.py`
- Create: `tests/studies/test_prepare_fly001_128_external_split.py`

**Step 1: Write the failing test**

```python
# tests/studies/test_prepare_fly001_128_external_split.py
from pathlib import Path
import json
import numpy as np

from scripts.studies.prepare_fly001_128_external_split import prepare_dataset


def _write_raw_npz(path: Path):
    n = 10
    N = 128
    np.savez(
        path,
        diff3d=np.arange(n * N * N, dtype=np.uint16).reshape(n, N, N),
        xcoords=np.linspace(1.0, 2.0, n),
        ycoords=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64),
        xcoords_start=np.linspace(1.0, 2.0, n),
        ycoords_start=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64),
        scan_index=np.zeros(n, dtype=np.int64),
        probeGuess=np.ones((N, N), dtype=np.complex64),
        objectGuess=np.ones((462, 461), dtype=np.complex64),
    )


def test_prepare_dataset_writes_canonical_and_disjoint_splits(tmp_path):
    raw = tmp_path / "fly001_128_train.npz"
    out = tmp_path / "fly001_128"
    _write_raw_npz(raw)

    result = prepare_dataset(raw_npz=raw, output_dir=out)

    canonical = np.load(result["canonical_npz"], allow_pickle=True)
    top = np.load(result["train_npz"], allow_pickle=True)
    bottom = np.load(result["test_npz"], allow_pickle=True)

    assert "diffraction" in canonical.files
    assert canonical["diffraction"].dtype == np.float32
    assert "diff3d" not in canonical.files

    y_top = top["ycoords"]
    y_bottom = bottom["ycoords"]
    assert y_top.min() >= result["split_threshold"]
    assert y_bottom.max() < result["split_threshold"]

    manifest = json.loads(Path(result["manifest_json"]).read_text())
    assert manifest["source_file"] == str(raw)
    assert manifest["n_total"] == 10
    assert manifest["n_train"] + manifest["n_test"] == 10
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/studies/test_prepare_fly001_128_external_split.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing `prepare_dataset`.

**Step 3: Write minimal implementation**

```python
# scripts/studies/prepare_fly001_128_external_split.py
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import numpy as np


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonicalize(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for k, v in data.items():
        nk = "diffraction" if k == "diff3d" else k
        vv = v.astype(np.float32) if getattr(v, "dtype", None) == np.uint16 else v
        out[nk] = vv
    return out


def _split_payload(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    n = data["xcoords"].shape[0]
    out = {}
    for k, v in data.items():
        if hasattr(v, "shape") and v.shape and v.shape[0] == n:
            out[k] = v[mask]
        else:
            out[k] = v
    return out


def prepare_dataset(*, raw_npz: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(raw_npz, allow_pickle=True) as d:
        raw = {k: d[k] for k in d.files}

    canonical = _canonicalize(raw)
    y = np.asarray(canonical["ycoords"], dtype=np.float64)
    threshold = float((y.min() + y.max()) / 2.0)

    train_mask = y >= threshold
    test_mask = ~train_mask

    canonical_npz = output_dir / "fly001_128_train_converted.npz"
    train_npz = output_dir / "fly001_128_top_half_converted.npz"
    test_npz = output_dir / "fly001_128_bottom_half_converted.npz"
    manifest_json = output_dir / "manifest.json"

    np.savez_compressed(canonical_npz, **canonical)
    np.savez_compressed(train_npz, **_split_payload(canonical, train_mask))
    np.savez_compressed(test_npz, **_split_payload(canonical, test_mask))

    manifest = {
        "source_file": str(raw_npz),
        "source_sha256": _sha256(raw_npz),
        "canonical_npz": str(canonical_npz),
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        "split_axis": "ycoords",
        "split_threshold": threshold,
        "n_total": int(y.size),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2))

    return {
        "canonical_npz": str(canonical_npz),
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        "manifest_json": str(manifest_json),
        "split_threshold": threshold,
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/studies/test_prepare_fly001_128_external_split.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/studies/test_prepare_fly001_128_external_split.py scripts/studies/prepare_fly001_128_external_split.py
git commit -m "feat(studies): add deterministic fly001 N128 prep and split script"
```

### Task 2: Add CLI for Prep Script and Safety Guards

**Files:**
- Modify: `scripts/studies/prepare_fly001_128_external_split.py`
- Modify: `tests/studies/test_prepare_fly001_128_external_split.py`

**Step 1: Write the failing test**

```python
import json
from pathlib import Path
from scripts.studies.prepare_fly001_128_external_split import main


def test_cli_writes_manifest_with_required_fields(tmp_path, monkeypatch):
    raw = tmp_path / "raw.npz"
    # fixture writer reused from previous test
    _write_raw_npz(raw)
    out = tmp_path / "out"

    argv = [
        "prepare_fly001_128_external_split.py",
        "--input-npz", str(raw),
        "--output-dir", str(out),
    ]
    monkeypatch.setattr("sys.argv", argv)
    main()

    manifest = json.loads((out / "manifest.json").read_text())
    for key in ["source_sha256", "canonical_npz", "train_npz", "test_npz", "n_train", "n_test"]:
        assert key in manifest
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/studies/test_prepare_fly001_128_external_split.py -k cli -v`
Expected: FAIL (no CLI `main` yet).

**Step 3: Write minimal implementation**

```python
# add argparse main() to script
# - --input-npz (required)
# - --output-dir (required)
# - prints resulting manifest path
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/studies/test_prepare_fly001_128_external_split.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/studies/test_prepare_fly001_128_external_split.py scripts/studies/prepare_fly001_128_external_split.py
git commit -m "feat(studies): add CLI and manifest guarantees for fly001 N128 prep"
```

### Task 3: Add Reproducible N=128 Study Launcher + Studies Index Entry

**Files:**
- Create: `.artifacts/studies/grid_lines_external_fly001_n128_top_train_bottom_test_e40/run_study.sh`
- Modify: `docs/studies/index.md`
- Create: `tests/studies/test_studies_index_entries.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_study_index_registers_fly001_n128_external_runbook():
    index = Path("docs/studies/index.md").read_text()
    assert "grid-lines-external-fly001-n128-top-train-bottom-test-e40" in index
    script = Path(".artifacts/studies/grid_lines_external_fly001_n128_top_train_bottom_test_e40/run_study.sh")
    assert script.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/studies/test_studies_index_entries.py -v`
Expected: FAIL (entry/script missing).

**Step 3: Write minimal implementation**

```bash
# .artifacts/studies/grid_lines_external_fly001_n128_top_train_bottom_test_e40/run_study.sh
# - call build_datasets(dataset_source="external_raw_npz", required_ns=[128])
# - use train_data=~/Documents/128_res/fly001_128_top_half_converted.npz
# - use test_data=~/Documents/128_res/fly001_128_bottom_half_converted.npz
# - set N=128, n_groups=4096, epochs=40, reassembly_mode="position"
# - run hybrid_resnet and cnn
# - finalize compare outputs
```

```markdown
<!-- docs/studies/index.md -->
### `grid-lines-external-fly001-n128-top-train-bottom-test-e40`
- Purpose: External raw NPZ study on fly001 N=128 with disjoint top-half train / bottom-half test.
- Script: `.artifacts/studies/grid_lines_external_fly001_n128_top_train_bottom_test_e40/run_study.sh`
- Output directory: `outputs/grid_lines_external_fly001_n128_top_train_bottom_test_n4096_e40_seed3_cnn_hybrid_resnet`
- Dataset manifest: `datasets/fly001_128/manifest.json`
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/studies/test_studies_index_entries.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/studies/test_studies_index_entries.py docs/studies/index.md .artifacts/studies/grid_lines_external_fly001_n128_top_train_bottom_test_e40/run_study.sh
git commit -m "docs(studies): register fly001 N128 external runbook"
```

### Task 4: Add Dataset Documentation + Documentation Hub Discoverability

**Files:**
- Create: `docs/FLY001_128_DATASET_GUIDE.md`
- Modify: `docs/index.md`
- Modify: `docs/COMMANDS_REFERENCE.md`
- Modify: `tests/studies/test_studies_index_entries.py`

**Step 1: Write the failing test**

```python
def test_docs_index_links_fly001_128_dataset_guide():
    idx = Path("docs/index.md").read_text()
    assert "FLY001 N=128 Dataset Guide" in idx
    assert Path("docs/FLY001_128_DATASET_GUIDE.md").exists()


def test_commands_reference_contains_fly001_128_recipe():
    text = Path("docs/COMMANDS_REFERENCE.md").read_text()
    assert "fly001_128_top_half_converted.npz" in text
    assert "fly001_128_bottom_half_converted.npz" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/studies/test_studies_index_entries.py -k "fly001_128 or commands_reference" -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

```markdown
# docs/FLY001_128_DATASET_GUIDE.md
- Raw file format summary for `~/Documents/128_res/fly001_128_train.npz`
- Canonicalization rule (`diff3d` -> `diffraction`, uint16 -> float32)
- Deterministic split policy (`ycoords >= threshold` train, `< threshold` test)
- Paths produced in `datasets/fly001_128/`
- Manifest schema and example
```

```markdown
# docs/index.md
- Add Datasets entry:
  - [FLY001 N=128 Dataset Guide](FLY001_128_DATASET_GUIDE.md)
```

```markdown
# docs/COMMANDS_REFERENCE.md
- Add end-to-end recipe:
  1) `prepare_fly001_128_external_split.py`
  2) run `.artifacts/.../run_study.sh`
```

**Step 4: Run tests to verify pass**

Run: `pytest tests/studies/test_studies_index_entries.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/studies/test_studies_index_entries.py docs/FLY001_128_DATASET_GUIDE.md docs/index.md docs/COMMANDS_REFERENCE.md
git commit -m "docs: add fly001 N128 dataset guide and command recipes"
```

### Task 5: Verification Bundle and Final Validation

**Files:**
- Modify: `docs/studies/index.md` (final command/output corrections if needed)

**Step 1: Write final verification checklist (failing criterion = any missing artifact)**

```bash
# check required files
ls docs/FLY001_128_DATASET_GUIDE.md \
   docs/studies/index.md \
   .artifacts/studies/grid_lines_external_fly001_n128_top_train_bottom_test_e40/run_study.sh \
   scripts/studies/prepare_fly001_128_external_split.py
```

**Step 2: Run test suite for touched functionality**

Run:
- `pytest tests/studies/test_prepare_fly001_128_external_split.py -v`
- `pytest tests/studies/test_studies_index_entries.py -v`
- `pytest tests/studies/test_grid_study_dataset_builder.py -v`

Expected: PASS.

**Step 3: Run dataset prep CLI smoke on local sample path**

Run:

```bash
python scripts/studies/prepare_fly001_128_external_split.py \
  --input-npz ~/Documents/128_res/fly001_128_train.npz \
  --output-dir datasets/fly001_128
```

Expected: writes canonical + split NPZs + `datasets/fly001_128/manifest.json`.

**Step 4: Optional study smoke (short)**

Run (optional quick validation): edit launcher to `nepochs=1`, `n_groups=64` and execute once.
Expected: output tree created, metrics emitted.

**Step 5: Commit**

```bash
git add docs/studies/index.md
git commit -m "chore(studies): finalize fly001 N128 reproducibility verification"
```

## Notes for Execution

- The full 40-epoch N=128 study is computationally heavy; use a short smoke first, then full run.
- Keep train/test disjoint by construction (top vs bottom split); avoid using identical train/test paths.
- Do not rely on `scan_index` as unique identity for fly001/fly64 provenance checks; use coordinate matching.
