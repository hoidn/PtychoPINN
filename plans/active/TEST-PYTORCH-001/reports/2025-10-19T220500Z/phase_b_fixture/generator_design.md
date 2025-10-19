# Fixture Generator Design Specification

**Initiative**: TEST-PYTORCH-001 — PyTorch integration workflow regression
**Phase**: B2.A (Generator Design + Stub)
**Date**: 2025-10-19T220500Z
**Status**: Design Document (Implementation Pending)

---

## 1. Purpose & Requirements

### 1.1 Objective

Create a deterministic, reproducible fixture generator that transforms the canonical Run1084 dataset into a minimal, DATA-001 compliant NPZ file suitable for PyTorch integration regression testing.

### 1.2 Acceptance Criteria Reference

Per `fixture_scope.md` §3, the generator must produce:

- **Shape**: `diffraction` as `(N=64, H=64, W=64)` (canonical `(N, H, W)` order)
- **Dtypes**: `float32` diffraction, `complex64` probe/object
- **Subset**: First 64 scan positions (deterministic, no seeding required)
- **Contract**: Full compliance with `specs/data_contracts.md` §1
- **Runtime**: Enable <25s CPU-only training (2 epochs, proven budget)

---

## 2. Input Specification

### 2.1 Source Dataset

**Path**: `datasets/Run1084_recon3_postPC_shrunk_3.npz`

**Expected Structure** (per Phase B1 dataset probe):
```
diffraction:   (64, 64, 1087)  dtype=float64    # LEGACY FORMAT (H, W, N)
objectGuess:   (227, 226)      dtype=complex128
probeGuess:    (64, 64)        dtype=complex128
xcoords:       (1087,)         dtype=float64
ycoords:       (1087,)         dtype=float64
xcoords_start: (1087,)         dtype=float64    # Optional, preserved if present
ycoords_start: (1087,)         dtype=float64    # Optional, preserved if present
```

### 2.2 Transformation Requirements

| Operation | Input | Output | Rationale |
|-----------|-------|--------|-----------|
| **Axis Reorder** | `diffraction: (H, W, N)` | `(N, H, W)` | DATA-001 canonical format (avoid FORMAT-001 heuristic reliance) |
| **Dtype Downcast** | `diffraction: float64` | `float32` | DATA-001 contract, memory efficiency |
| **Dtype Downcast** | `objectGuess: complex128` | `complex64` | DATA-001 contract |
| **Dtype Downcast** | `probeGuess: complex128` | `complex64` | DATA-001 contract |
| **Subset Extraction** | All arrays indexed by N=1087 | First `n_subset=64` positions | Deterministic, reproducible, proven runtime budget |
| **Preserve Coordinates** | `xcoords, ycoords` at `float64` | Same | Precision acceptable per DATA-001 |

---

## 3. Output Specification

### 3.1 Primary Artifact

**Filename**: `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz`

**Schema**:
```python
{
    'diffraction':   (64, 64, 64),   dtype=float32,     # Canonical (N, H, W)
    'objectGuess':   (227, 226),     dtype=complex64,   # Unchanged shape
    'probeGuess':    (64, 64),       dtype=complex64,   # Unchanged shape
    'xcoords':       (64,),          dtype=float64,     # First 64 positions
    'ycoords':       (64,),          dtype=float64,     # First 64 positions
    # Optional keys preserved if in source:
    'xcoords_start': (64,),          dtype=float64,
    'ycoords_start': (64,),          dtype=float64,
}
```

### 3.2 Metadata Sidecar

**Filename**: `tests/fixtures/pytorch_integration/minimal_dataset_v1.json`

**Content** (example):
```json
{
  "version": "v1",
  "created": "2025-10-19T22:05:00Z",
  "generator_script": "scripts/tools/make_pytorch_integration_fixture.py",
  "generator_commit": "<commit_sha>",
  "source_dataset": "datasets/Run1084_recon3_postPC_shrunk_3.npz",
  "subset_strategy": "first_n_positions",
  "subset_size": 64,
  "transformations": [
    "diffraction: transpose (H,W,N) → (N,H,W)",
    "diffraction: dtype float64 → float32",
    "objectGuess: dtype complex128 → complex64",
    "probeGuess: dtype complex128 → complex64",
    "coordinates: subset indices [0:64]"
  ],
  "sha256_checksum": "<computed_hash>",
  "validation_notes": "Compliant with specs/data_contracts.md §1; tested via tests/torch/test_fixture_pytorch_integration.py"
}
```

---

## 4. Algorithm Pseudocode

```python
import numpy as np
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

def generate_fixture(source_path, output_path, n_subset=64):
    """
    Generate minimal PyTorch integration fixture from canonical dataset.

    Args:
        source_path: Path to Run1084_recon3_postPC_shrunk_3.npz
        output_path: Path to output fixture NPZ
        n_subset: Number of scan positions to extract (default: 64)
    """
    # Step 1: Load source dataset
    source = np.load(source_path)

    # Step 2: Extract and transform diffraction (CRITICAL)
    # Input: (H=64, W=64, N=1087) float64
    # Output: (N=64, H=64, W=64) float32
    diffraction_legacy = source['diffraction']
    diffraction_canonical = diffraction_legacy.transpose(2, 0, 1)  # (H,W,N) → (N,H,W)
    diffraction_subset = diffraction_canonical[:n_subset, :, :].astype(np.float32)

    # Step 3: Downcast probe/object
    objectGuess = source['objectGuess'].astype(np.complex64)
    probeGuess = source['probeGuess'].astype(np.complex64)

    # Step 4: Subset coordinates (deterministic: first n_subset)
    xcoords = source['xcoords'][:n_subset]
    ycoords = source['ycoords'][:n_subset]

    # Step 5: Preserve optional keys if present
    optional_keys = {}
    for key in ['xcoords_start', 'ycoords_start']:
        if key in source:
            optional_keys[key] = source[key][:n_subset]

    # Step 6: Assemble output dictionary
    output_data = {
        'diffraction': diffraction_subset,
        'objectGuess': objectGuess,
        'probeGuess': probeGuess,
        'xcoords': xcoords,
        'ycoords': ycoords,
        **optional_keys
    }

    # Step 7: Save NPZ with compression
    np.savez_compressed(output_path, **output_data)

    # Step 8: Compute SHA256 checksum
    checksum = compute_sha256(output_path)

    # Step 9: Generate metadata sidecar
    metadata = {
        "version": "v1",
        "created": datetime.now(timezone.utc).isoformat(),
        "generator_script": "scripts/tools/make_pytorch_integration_fixture.py",
        "generator_commit": get_git_commit(),  # Helper to fetch current commit SHA
        "source_dataset": str(source_path),
        "subset_strategy": "first_n_positions",
        "subset_size": n_subset,
        "transformations": [
            "diffraction: transpose (H,W,N) → (N,H,W)",
            "diffraction: dtype float64 → float32",
            "objectGuess: dtype complex128 → complex64",
            "probeGuess: dtype complex128 → complex64",
            f"coordinates: subset indices [0:{n_subset}]"
        ],
        "sha256_checksum": checksum,
        "validation_notes": "Compliant with specs/data_contracts.md §1"
    }

    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return output_path, metadata_path, checksum

def compute_sha256(file_path):
    """Compute SHA256 checksum of NPZ file."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_git_commit():
    """Fetch current git commit SHA."""
    import subprocess
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            text=True
        ).strip()
    except:
        return "unknown"
```

---

## 5. CLI Interface Design

### 5.1 Command Signature

```bash
python scripts/tools/make_pytorch_integration_fixture.py \
    --source datasets/Run1084_recon3_postPC_shrunk_3.npz \
    --output tests/fixtures/pytorch_integration/minimal_dataset_v1.npz \
    --subset-size 64 \
    --metadata-out tests/fixtures/pytorch_integration/minimal_dataset_v1.json
```

### 5.2 Argparse Specification

```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate minimal PyTorch integration fixture from canonical dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default 64-position fixture
  python scripts/tools/make_pytorch_integration_fixture.py \\
      --source datasets/Run1084_recon3_postPC_shrunk_3.npz \\
      --output tests/fixtures/pytorch_integration/minimal_dataset_v1.npz

  # Custom subset size
  python scripts/tools/make_pytorch_integration_fixture.py \\
      --source datasets/Run1084_recon3_postPC_shrunk_3.npz \\
      --output tests/fixtures/pytorch_integration/minimal_dataset_v2.npz \\
      --subset-size 128

See plans/active/TEST-PYTORCH-001/reports/2025-10-19T220500Z/phase_b_fixture/generator_design.md
        """
    )

    parser.add_argument(
        '--source',
        type=Path,
        required=True,
        help='Path to source NPZ dataset (e.g., Run1084_recon3_postPC_shrunk_3.npz)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Path to output fixture NPZ (will create parent directories)'
    )

    parser.add_argument(
        '--subset-size',
        type=int,
        default=64,
        help='Number of scan positions to extract (default: 64)'
    )

    parser.add_argument(
        '--metadata-out',
        type=Path,
        default=None,
        help='Path to metadata JSON sidecar (default: <output>.json)'
    )

    return parser.parse_args()
```

---

## 6. Validation Strategy (Phase B2.B RED Test)

The fixture generator implementation will be validated via `tests/torch/test_fixture_pytorch_integration.py::test_fixture_outputs_match_contract`, which asserts:

### 6.1 Shape Validation
```python
assert fixture['diffraction'].shape == (64, 64, 64), "Diffraction must be (N, H, W)"
assert fixture['objectGuess'].shape == (227, 226), "Object shape preserved"
assert fixture['probeGuess'].shape == (64, 64), "Probe shape preserved"
assert fixture['xcoords'].shape == (64,), "Coordinates subset to n_subset"
```

### 6.2 Dtype Validation
```python
assert fixture['diffraction'].dtype == np.float32, "DATA-001 compliance"
assert fixture['objectGuess'].dtype == np.complex64, "DATA-001 compliance"
assert fixture['probeGuess'].dtype == np.complex64, "DATA-001 compliance"
```

### 6.3 Metadata Validation
```python
metadata_path = fixture_path.with_suffix('.json')
assert metadata_path.exists(), "Metadata sidecar must exist"
metadata = json.load(open(metadata_path))
assert metadata['subset_size'] == 64
assert metadata['sha256_checksum'] == compute_sha256(fixture_path)
```

### 6.4 Normalization Validation
```python
assert np.max(fixture['diffraction']) < 10.0, "Amplitude normalization preserved"
assert np.min(fixture['diffraction']) >= 0.0, "Non-negative amplitudes"
```

---

## 7. Error Handling & Edge Cases

### 7.1 Input Validation

```python
def validate_source(source_path):
    """Validate source dataset conforms to expected structure."""
    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_path}")

    source = np.load(source_path)
    required_keys = ['diffraction', 'objectGuess', 'probeGuess', 'xcoords', 'ycoords']
    missing = [k for k in required_keys if k not in source.keys()]
    if missing:
        raise KeyError(f"Source missing required keys: {missing}")

    # Verify diffraction shape matches expected legacy format
    if source['diffraction'].ndim != 3:
        raise ValueError(f"Expected 3D diffraction array, got shape {source['diffraction'].shape}")
```

### 7.2 Subset Size Constraints

```python
def validate_subset_size(n_subset, n_total):
    """Ensure subset size is valid."""
    if n_subset < 16:
        raise ValueError(f"Subset size {n_subset} < 16; insufficient for grouping tests")
    if n_subset > n_total:
        raise ValueError(f"Subset size {n_subset} exceeds dataset size {n_total}")
```

---

## 8. Implementation Notes

### 8.1 Phase B2.A Deliverable (This Loop)

For **TDD RED phase**, this loop will:
1. ✓ Author this design document (`generator_design.md`)
2. ✓ Create generator script stub (`make_pytorch_integration_fixture.py`) with:
   - Full argparse interface
   - `main()` function raising `NotImplementedError`
   - Module docstring referencing this design doc
3. Next loop (B2.B): Write failing test asserting fixture contract

### 8.2 Phase B2.C Deliverable (Future Loop)

Implement the full generator logic per §4 pseudocode to make the test GREEN.

---

## 9. References

### 9.1 Project Documents
- `specs/data_contracts.md` §1 — Canonical NPZ format requirements
- `docs/workflows/pytorch.md` §§4–8 — RawData loading expectations, CONFIG-001
- `docs/findings.md` — FORMAT-001 (legacy transpose heuristic), POLICY-001 (torch required)
- `plans/active/TEST-PYTORCH-001/reports/2025-10-19T215300Z/phase_b_fixture/fixture_scope.md` — Acceptance criteria

### 9.2 Source Code
- `ptycho_torch/data.py` — PyTorch dataloader with FORMAT-001 auto-transpose
- `ptycho/raw_data.py` — RawData.from_file() NPZ ingestion contract
- `tests/torch/test_integration_workflow_torch.py` — Target integration test consuming fixture

---

**Design Complete**. Ready for Phase B2.A stub implementation.
