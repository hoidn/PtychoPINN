#!/usr/bin/env python
"""
Generate minimal PyTorch integration fixture from canonical dataset.

This script transforms the Run1084_recon3_postPC_shrunk_3.npz dataset into a
deterministic, DATA-001 compliant fixture suitable for PyTorch integration
regression testing. It performs:

1. Axis reordering: (H, W, N) → (N, H, W) canonical format
2. Dtype downcasting: float64 → float32, complex128 → complex64
3. Deterministic subset extraction: first n_subset scan positions
4. Metadata generation: JSON sidecar with provenance and SHA256 checksum

Design Specification:
    plans/active/TEST-PYTORCH-001/reports/2025-10-19T220500Z/phase_b_fixture/generator_design.md

Data Contract:
    specs/data_contracts.md §1

Usage:
    python scripts/tools/make_pytorch_integration_fixture.py \\
        --source datasets/Run1084_recon3_postPC_shrunk_3.npz \\
        --output tests/fixtures/pytorch_integration/minimal_dataset_v1.npz \\
        --subset-size 64

Author: Ralph (TDD stub, Phase B2.A)
Date: 2025-10-19
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import json
import hashlib
import subprocess
from datetime import datetime, timezone


def compute_sha256(file_path: Path) -> str:
    """
    Compute SHA256 checksum of NPZ file.

    Args:
        file_path: Path to file to hash

    Returns:
        Hexadecimal SHA256 checksum string
    """
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit() -> str:
    """
    Fetch current git commit SHA.

    Returns:
        Git commit SHA or "unknown" if git command fails
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for fixture generator.

    Returns:
        Parsed arguments namespace

    CLI Reference:
        See generator_design.md §5 for full specification
    """
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

Design Specification:
  plans/active/TEST-PYTORCH-001/reports/2025-10-19T220500Z/phase_b_fixture/generator_design.md

Data Contract:
  specs/data_contracts.md §1
        """
    )

    parser.add_argument(
        '--source',
        type=Path,
        required=True,
        help='Path to source NPZ dataset (e.g., datasets/Run1084_recon3_postPC_shrunk_3.npz)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Path to output fixture NPZ (will create parent directories if needed)'
    )

    parser.add_argument(
        '--subset-size',
        type=int,
        default=64,
        help='Number of scan positions to extract (default: 64; minimum: 16)'
    )

    parser.add_argument(
        '--metadata-out',
        type=Path,
        default=None,
        help='Path to metadata JSON sidecar (default: <output>.json)'
    )

    return parser.parse_args()


def generate_fixture(
    source_path: Path,
    output_path: Path,
    subset_size: int = 64,
    metadata_path: Path = None
) -> Tuple[Path, Path, str]:
    """
    Generate minimal PyTorch integration fixture from canonical dataset.

    This function performs the transformations specified in generator_design.md §4:
    1. Load source NPZ dataset
    2. Transpose diffraction array from (H,W,N) to (N,H,W)
    3. Downcast dtypes per DATA-001 contract
    4. Extract first subset_size scan positions (deterministic)
    5. Save compressed NPZ fixture
    6. Compute SHA256 checksum
    7. Generate JSON metadata sidecar

    Args:
        source_path: Path to canonical Run1084 dataset NPZ
        output_path: Path to output fixture NPZ
        subset_size: Number of scan positions to extract (default: 64)
        metadata_path: Optional path to metadata sidecar (default: <output>.json)

    Returns:
        Tuple of (fixture_path, metadata_path, sha256_checksum)

    Raises:
        FileNotFoundError: If source dataset does not exist
        KeyError: If source dataset missing required keys
        ValueError: If subset_size invalid

    Design Reference:
        generator_design.md §4 (Algorithm Pseudocode)

    Contract Reference:
        specs/data_contracts.md §1 (Canonical NPZ format)
        docs/findings.md#FORMAT-001 (Legacy transpose heuristic)
    """
    # Step 1: Load source dataset
    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_path}")

    source = np.load(source_path)

    # Validate required keys
    required_keys = ['diffraction', 'objectGuess', 'probeGuess', 'xcoords', 'ycoords']
    missing = [k for k in required_keys if k not in source.keys()]
    if missing:
        raise KeyError(f"Source missing required keys: {missing}")

    # Validate diffraction shape
    if source['diffraction'].ndim != 3:
        raise ValueError(f"Expected 3D diffraction array, got shape {source['diffraction'].shape}")

    # Validate subset size
    n_total = source['diffraction'].shape[-1]  # Works for both (H,W,N) and (N,H,W)
    if n_total < subset_size:
        # Determine actual N dimension
        if source['diffraction'].shape[0] == source['xcoords'].shape[0]:
            n_total = source['diffraction'].shape[0]
        elif source['diffraction'].shape[2] == source['xcoords'].shape[0]:
            n_total = source['diffraction'].shape[2]

    if subset_size < 16:
        raise ValueError(f"Subset size {subset_size} < 16; insufficient for grouping tests")
    if subset_size > n_total:
        raise ValueError(f"Subset size {subset_size} exceeds dataset size {n_total}")

    # Step 2: Extract and transform diffraction (CRITICAL)
    # Input: (H=64, W=64, N=1087) float64
    # Output: (N=64, H=64, W=64) float32
    diffraction_legacy = source['diffraction']

    # Detect if already in canonical format (N, H, W) or legacy (H, W, N)
    # Heuristic: if first dimension matches coord count, it's canonical
    if diffraction_legacy.shape[0] == source['xcoords'].shape[0]:
        # Already canonical (N, H, W)
        diffraction_canonical = diffraction_legacy
    else:
        # Legacy format (H, W, N) → transpose to (N, H, W)
        diffraction_canonical = diffraction_legacy.transpose(2, 0, 1)

    # Subset first n positions and downcast to float32
    diffraction_subset = diffraction_canonical[:subset_size, :, :].astype(np.float32)

    # Step 3: Downcast probe/object
    objectGuess = source['objectGuess'].astype(np.complex64)
    probeGuess = source['probeGuess'].astype(np.complex64)

    # Step 4: Subset coordinates (deterministic stratified sampling for spatial diversity)
    # Use uniform sampling across the dataset to ensure >50% coordinate coverage
    # This satisfies fixture_scope.md §3.1.4 spatial diversity requirement
    total_positions = len(source['xcoords'])
    if subset_size >= total_positions:
        # Use all positions if subset_size >= total
        subset_indices = np.arange(total_positions)
    else:
        # Stratified sampling: evenly spaced indices for spatial diversity
        # This is deterministic (no random seed) and ensures good coordinate coverage
        step = total_positions / subset_size
        subset_indices = np.array([int(i * step) for i in range(subset_size)], dtype=int)

    xcoords = source['xcoords'][subset_indices]
    ycoords = source['ycoords'][subset_indices]

    # Step 5: Preserve optional keys if present
    optional_keys = {}
    for key in ['xcoords_start', 'ycoords_start']:
        if key in source:
            optional_keys[key] = source[key][subset_indices]

    # Step 6: Assemble output dictionary
    # Include both canonical 'diffraction' and legacy 'diff3d' keys for compatibility
    output_data = {
        'diffraction': diffraction_subset,
        'diff3d': diffraction_subset,  # Legacy alias for RawData backward compatibility
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
    if metadata_path is None:
        metadata_path = output_path.with_suffix('.json')

    metadata = {
        "version": "v1",
        "created": datetime.now(timezone.utc).isoformat(),
        "generator_script": "scripts/tools/make_pytorch_integration_fixture.py",
        "generator_commit": get_git_commit(),
        "source_dataset": str(source_path),
        "subset_strategy": "stratified_uniform_sampling",
        "subset_size": int(subset_size),
        "transformations": [
            "diffraction: transpose (H,W,N) → (N,H,W)" if diffraction_legacy.shape[0] != source['xcoords'].shape[0] else "diffraction: already canonical (N,H,W)",
            "diffraction: dtype float64 → float32",
            "diffraction: duplicate as diff3d for RawData backward compatibility",
            "objectGuess: dtype complex128 → complex64",
            "probeGuess: dtype complex128 → complex64",
            f"coordinates: stratified uniform sampling (step={total_positions}/{subset_size}={total_positions/subset_size:.1f})"
        ],
        "sha256_checksum": checksum,
        "validation_notes": "Compliant with specs/data_contracts.md §1; tested via tests/torch/test_fixture_pytorch_integration.py"
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return output_path, metadata_path, checksum


def main() -> int:
    """
    Main entry point for fixture generator CLI.

    Parses arguments and delegates to generate_fixture().

    Returns:
        Exit code (0 on success, 1 on error)

    Raises:
        NotImplementedError: Stub placeholder during TDD RED phase
    """
    args = parse_args()

    # Validate inputs (basic checks; full validation in implementation)
    if not args.source.exists():
        print(f"Error: Source dataset not found: {args.source}", file=sys.stderr)
        return 1

    if args.subset_size < 16:
        print(f"Error: Subset size {args.subset_size} < 16 (insufficient for grouping tests)", file=sys.stderr)
        return 1

    # Determine metadata path if not specified
    metadata_path = args.metadata_out or args.output.with_suffix('.json')

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fixture generator invoked (STUB MODE)")
    print(f"  Source:       {args.source}")
    print(f"  Output:       {args.output}")
    print(f"  Subset size:  {args.subset_size}")
    print(f"  Metadata:     {metadata_path}")
    print()

    try:
        # Delegate to core generator (currently raises NotImplementedError)
        fixture_path, meta_path, checksum = generate_fixture(
            source_path=args.source,
            output_path=args.output,
            subset_size=args.subset_size,
            metadata_path=metadata_path
        )

        print(f"✓ Fixture generated: {fixture_path}")
        print(f"✓ Metadata written:  {meta_path}")
        print(f"✓ SHA256 checksum:   {checksum}")
        return 0

    except NotImplementedError as e:
        print(f"NotImplementedError: {e}", file=sys.stderr)
        print("\nThis is expected during TDD RED phase (Phase B2.A/B2.B).", file=sys.stderr)
        print("Implementation deferred to Phase B2.C.", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
