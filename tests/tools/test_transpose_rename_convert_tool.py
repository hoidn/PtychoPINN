"""Test suite for transpose_rename_convert_tool.py with metadata preservation.

This module tests the canonicalization tool's ability to preserve and extend
metadata through NPZ transformations.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from scripts.tools.transpose_rename_convert_tool import transpose_rename_convert
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
        "data_transformations": []
    }


@pytest.fixture
def sample_npz_with_metadata(tmp_path, sample_metadata):
    """Create a sample NPZ file with metadata for testing."""
    npz_path = tmp_path / "input_with_metadata.npz"

    # Create sample data arrays
    data_dict = {
        'diff3d': np.random.rand(10, 32, 32).astype(np.float32),
        'xcoords': np.random.rand(10).astype(np.float64),
        'ycoords': np.random.rand(10).astype(np.float64),
        'probeGuess': np.random.rand(32, 32).astype(np.complex64),
        'objectGuess': np.random.rand(64, 64).astype(np.complex64),
        'Y': np.random.rand(10, 64, 64, 1).astype(np.complex64)  # 4D to test squeeze
    }

    # Save with metadata
    MetadataManager.save_with_metadata(str(npz_path), data_dict, sample_metadata)

    return npz_path


def test_canonicalize_preserves_metadata(tmp_path, sample_npz_with_metadata, sample_metadata):
    """Test that transpose_rename_convert preserves and extends metadata.

    This is the RED test - it should fail initially because the tool
    does not yet handle metadata.
    """
    output_path = tmp_path / "output_canonicalized.npz"

    # Run the canonicalization tool
    transpose_rename_convert(
        in_file=sample_npz_with_metadata,
        out_file=output_path,
        flipx=False,
        flipy=False
    )

    # Load output and check metadata
    output_data, output_metadata = MetadataManager.load_with_metadata(str(output_path))

    # Metadata should be preserved
    assert output_metadata is not None, "Metadata was lost during canonicalization"

    # Original physics parameters should be intact
    assert output_metadata['physics_parameters']['nphotons'] == sample_metadata['physics_parameters']['nphotons']
    assert output_metadata['physics_parameters']['N'] == sample_metadata['physics_parameters']['N']

    # Should have added a transformation record
    transformations = output_metadata.get('data_transformations', [])
    assert len(transformations) > 0, "No transformation record added"

    # Latest transformation should be from transpose_rename_convert
    latest_transform = transformations[-1]
    assert latest_transform['tool'] == 'transpose_rename_convert'
    assert latest_transform['operation'] == 'canonicalize'
    assert 'flipx' in latest_transform['parameters']
    assert 'flipy' in latest_transform['parameters']

    # Data should still be correctly transformed
    assert 'diffraction' in output_data, "diff3d was not renamed to diffraction"
    assert output_data['Y'].ndim == 3, "Y array was not squeezed to 3D"


def test_canonicalize_without_metadata(tmp_path):
    """Test that the tool handles NPZ files without metadata gracefully."""
    # Create NPZ without metadata
    input_path = tmp_path / "input_no_metadata.npz"
    data_dict = {
        'diff3d': np.random.rand(5, 32, 32).astype(np.float32),
        'xcoords': np.random.rand(5).astype(np.float64),
        'ycoords': np.random.rand(5).astype(np.float64)
    }
    np.savez_compressed(input_path, **data_dict)

    output_path = tmp_path / "output_no_metadata.npz"

    # Should not fail
    transpose_rename_convert(
        in_file=input_path,
        out_file=output_path,
        flipx=True,
        flipy=False
    )

    # Load output
    output_data, output_metadata = MetadataManager.load_with_metadata(str(output_path))

    # Metadata should be created with transformation record
    assert output_metadata is not None
    assert 'data_transformations' in output_metadata
    assert len(output_metadata['data_transformations']) == 1

    # Data should still be transformed
    assert 'diffraction' in output_data
    assert np.array_equal(output_data['xcoords'], -data_dict['xcoords'])  # flipx=True
