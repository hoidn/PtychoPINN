"""Test suite for generate_patches_tool.py with metadata preservation.

This module tests the patch generation tool's ability to preserve and extend
metadata through NPZ transformations.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from scripts.tools.generate_patches_tool import generate_patches
from ptycho.metadata import MetadataManager


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return {
        "schema_version": "1.0.0",
        "creation_info": {
            "timestamp": "2025-11-07T13:05:00Z",
            "script": "test_fixture",
            "hostname": "test-host",
            "ptychopinn_version": "2.0.0"
        },
        "physics_parameters": {
            "nphotons": 1e9,
            "gridsize": 1,
            "N": 64
        },
        "data_transformations": [
            {
                "tool": "transpose_rename_convert",
                "timestamp": "2025-11-07T13:00:00Z",
                "operation": "canonicalize",
                "parameters": {"flipx": False, "flipy": False}
            }
        ]
    }


@pytest.fixture
def sample_npz_with_metadata(tmp_path, sample_metadata):
    """Create a sample NPZ file with metadata for patch generation testing."""
    npz_path = tmp_path / "input_with_metadata.npz"

    # Create sample data arrays that match what generate_patches expects
    n_scans = 20
    probe_size = 32
    object_size = 128

    data_dict = {
        'diffraction': np.random.rand(n_scans, probe_size, probe_size).astype(np.float32),
        'xcoords': np.random.rand(n_scans).astype(np.float64) * 1e-6,
        'ycoords': np.random.rand(n_scans).astype(np.float64) * 1e-6,
        'probeGuess': (np.random.rand(probe_size, probe_size) +
                       1j * np.random.rand(probe_size, probe_size)).astype(np.complex64),
        'objectGuess': (np.random.rand(object_size, object_size) +
                        1j * np.random.rand(object_size, object_size)).astype(np.complex64),
        'scan_index': np.arange(n_scans, dtype=np.int32)
    }

    # Save with metadata
    MetadataManager.save_with_metadata(str(npz_path), data_dict, sample_metadata)

    return npz_path


def test_generate_patches_preserves_metadata(tmp_path, sample_npz_with_metadata, sample_metadata):
    """Test that generate_patches preserves and extends metadata.

    This is the RED test - it should fail initially because the tool
    does not yet handle metadata.
    """
    output_path = tmp_path / "output_with_patches.npz"

    # Run the patch generation tool
    generate_patches(
        input_path=sample_npz_with_metadata,
        output_path=output_path,
        patch_size=64,
        k_neighbors=5,
        nsamples=1
    )

    # Load output and check metadata
    output_data, output_metadata = MetadataManager.load_with_metadata(str(output_path))

    # Metadata should be preserved
    assert output_metadata is not None, "Metadata was lost during patch generation"

    # Original physics parameters should be intact
    assert output_metadata['physics_parameters']['nphotons'] == sample_metadata['physics_parameters']['nphotons']
    assert output_metadata['physics_parameters']['N'] == sample_metadata['physics_parameters']['N']

    # Should have transformation history from previous step
    transformations = output_metadata.get('data_transformations', [])
    assert len(transformations) >= 2, "Transformation history not preserved"

    # Check that the original canonicalize transformation is preserved
    assert transformations[0]['tool'] == 'transpose_rename_convert'

    # Latest transformation should be from generate_patches
    latest_transform = transformations[-1]
    assert latest_transform['tool'] == 'generate_patches'
    assert latest_transform['operation'] == 'generate_patches'
    assert 'patch_size' in latest_transform['parameters']
    assert latest_transform['parameters']['patch_size'] == 64
    assert 'k_neighbors' in latest_transform['parameters']
    assert latest_transform['parameters']['k_neighbors'] == 5

    # Data should contain Y patches
    assert 'Y' in output_data, "Y patches were not generated"
    # Note: generate_patches creates 4D Y arrays (N, H, W, 1) by design
    # These get squeezed to 3D by transpose_rename_convert in Phase C workflow
    assert output_data['Y'].ndim == 4, "Y patches from generate_patches are 4D"
    assert output_data['Y'].shape[-1] == 1, "Last dimension should be singleton for squeezing"


def test_generate_patches_without_metadata(tmp_path):
    """Test that the tool handles NPZ files without metadata gracefully."""
    # Create NPZ without metadata
    input_path = tmp_path / "input_no_metadata.npz"

    n_scans = 15
    probe_size = 32
    object_size = 128

    data_dict = {
        'diffraction': np.random.rand(n_scans, probe_size, probe_size).astype(np.float32),
        'xcoords': np.random.rand(n_scans).astype(np.float64) * 1e-6,
        'ycoords': np.random.rand(n_scans).astype(np.float64) * 1e-6,
        'probeGuess': (np.random.rand(probe_size, probe_size) +
                       1j * np.random.rand(probe_size, probe_size)).astype(np.complex64),
        'objectGuess': (np.random.rand(object_size, object_size) +
                        1j * np.random.rand(object_size, object_size)).astype(np.complex64),
        'scan_index': np.arange(n_scans, dtype=np.int32)
    }
    np.savez_compressed(input_path, **data_dict)

    output_path = tmp_path / "output_no_metadata.npz"

    # Should not fail
    generate_patches(
        input_path=input_path,
        output_path=output_path,
        patch_size=64,
        k_neighbors=5,
        nsamples=1
    )

    # Load output
    output_data, output_metadata = MetadataManager.load_with_metadata(str(output_path))

    # Metadata should be created with transformation record
    assert output_metadata is not None
    assert 'data_transformations' in output_metadata
    assert len(output_metadata['data_transformations']) == 1

    # Latest transformation should be from generate_patches
    assert output_metadata['data_transformations'][0]['tool'] == 'generate_patches'

    # Data should contain Y patches
    assert 'Y' in output_data
