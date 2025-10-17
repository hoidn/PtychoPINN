"""
Unit tests for PyTorch dataloader DATA-001 compliance.

This module tests the PyTorch memory-mapped dataloader's ability to load
canonical NPZ datasets with the `diffraction` key and maintain backward
compatibility with legacy `diff3d` key.

Test Coverage:
1. Canonical diffraction key loading (DATA-001 spec compliance)
2. Legacy diff3d key backward compatibility
3. Error handling when neither key exists

References:
- specs/data_contracts.md §1 — Canonical NPZ schema
- docs/findings.md#DATA-001 — diffraction key requirement
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T223200Z/dataloader_triage.md
"""

import sys
import unittest
import numpy as np
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class TestDataloaderCanonicalKeySupport(unittest.TestCase):
    """
    Unit tests for PyTorch dataloader canonical NPZ key support.

    Status: RED PHASE (pre-implementation)
    Expected to fail until ptycho_torch/dataloader.py implements canonical key preference.
    """

    def setUp(self):
        """Create temporary directory and fixture NPZ files."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.data_path = Path(self.test_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.test_dir.cleanup()

    def _create_minimal_npz(self, filename, use_canonical_key=True):
        """
        Helper to create minimal NPZ fixture for dataloader testing.

        Args:
            filename: NPZ file path
            use_canonical_key: If True, uses 'diffraction' key; else uses 'diff3d'

        Returns:
            Path to created NPZ file
        """
        # Create minimal required arrays per DATA-001 spec
        n_images = 10
        H, W = 64, 64

        # Diffraction patterns (amplitude, float32)
        diff_key = 'diffraction' if use_canonical_key else 'diff3d'
        diffraction = np.random.rand(n_images, H, W).astype(np.float32) * 0.5

        # Coordinates
        xcoords = np.random.rand(n_images).astype(np.float64) * 100
        ycoords = np.random.rand(n_images).astype(np.float64) * 100

        # Probe and object
        probeGuess = np.random.rand(H, W).astype(np.complex64)
        objectGuess = np.random.rand(H*2, W*2).astype(np.complex64)

        # Ground truth patches (for supervised mode)
        Y = np.random.rand(n_images, H, W).astype(np.complex64)

        # Scan index
        scan_index = np.arange(n_images, dtype=np.int32)

        npz_data = {
            diff_key: diffraction,
            'xcoords': xcoords,
            'ycoords': ycoords,
            'probeGuess': probeGuess,
            'objectGuess': objectGuess,
            'Y': Y,
            'scan_index': scan_index,
        }

        filepath = self.data_path / filename
        np.savez(str(filepath), **npz_data)
        return filepath

    def test_loads_canonical_diffraction(self):
        """
        Test dataloader can load NPZ files with canonical 'diffraction' key.

        Red Phase: Expected to fail because npz_headers() only searches for 'diff3d'.
        Green Phase: Should pass after implementing canonical key preference.
        """
        # Arrange
        from ptycho_torch.dataloader import npz_headers

        canonical_npz = self._create_minimal_npz("canonical_dataset.npz", use_canonical_key=True)

        # Act & Assert
        try:
            shape, xcoords, ycoords = npz_headers(canonical_npz)

            # Verify shape extraction worked
            self.assertEqual(len(shape), 3, "Diffraction shape should be 3D (n_images, H, W)")
            self.assertEqual(shape[0], 10, "Expected 10 images in fixture")
            self.assertEqual(shape[1], 64, "Expected H=64")
            self.assertEqual(shape[2], 64, "Expected W=64")

            # Verify coordinates loaded
            self.assertEqual(len(xcoords), 10, "Expected 10 x-coordinates")
            self.assertEqual(len(ycoords), 10, "Expected 10 y-coordinates")

        except ValueError as e:
            self.fail(
                f"npz_headers() failed to load canonical 'diffraction' key. "
                f"Error: {e}. This indicates DATA-001 non-compliance."
            )

    def test_backward_compat_legacy_diff3d(self):
        """
        Test dataloader maintains backward compatibility with legacy 'diff3d' key.

        Should pass in both red and green phases (legacy fallback required).
        """
        # Arrange
        from ptycho_torch.dataloader import npz_headers

        legacy_npz = self._create_minimal_npz("legacy_dataset.npz", use_canonical_key=False)

        # Act & Assert
        try:
            shape, xcoords, ycoords = npz_headers(legacy_npz)

            self.assertEqual(len(shape), 3)
            self.assertEqual(shape[0], 10)

        except ValueError as e:
            self.fail(
                f"npz_headers() failed to load legacy 'diff3d' key. "
                f"Backward compatibility broken. Error: {e}"
            )

    def test_error_when_no_diffraction_key(self):
        """
        Test dataloader raises clear error when neither canonical nor legacy key exists.

        Should pass after implementing key preference logic.
        """
        # Arrange
        from ptycho_torch.dataloader import npz_headers

        # Create NPZ without diffraction data
        invalid_npz = self.data_path / "invalid_dataset.npz"
        np.savez(
            str(invalid_npz),
            xcoords=np.arange(10, dtype=np.float64),
            ycoords=np.arange(10, dtype=np.float64),
            probeGuess=np.ones((64, 64), dtype=np.complex64),
            objectGuess=np.ones((128, 128), dtype=np.complex64),
        )

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            npz_headers(invalid_npz)

        # Verify error message mentions both keys
        error_msg = str(context.exception).lower()
        self.assertTrue(
            'diffraction' in error_msg or 'diff3d' in error_msg,
            f"Error message should mention missing diffraction keys. Got: {context.exception}"
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
