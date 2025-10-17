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


class TestDataloaderFormatAutoTranspose(unittest.TestCase):
    """
    Unit tests for automatic legacy format detection and transposition.

    Tests the _get_diffraction_stack() function's ability to detect and fix
    legacy (H, W, N) format datasets, ensuring they're transposed to DATA-001
    compliant (N, H, W) format.

    Status: GREEN PHASE (post-implementation)
    """

    def setUp(self):
        """Create temporary directory for test NPZ files."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.data_path = Path(self.test_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.test_dir.cleanup()

    def test_auto_transposes_legacy_hwn_format(self):
        """
        Test that legacy (H, W, N) format is automatically transposed to (N, H, W).

        Regression test for INTEGRATE-PYTORCH-001-DATALOADER-INDEXING:
        Dataset Run1084_recon3_postPC_shrunk_3.npz has shape (64, 64, 1087),
        which caused IndexError when nn_indices contained values > 64.
        """
        # Arrange: Create NPZ with legacy (H, W, N) format
        from ptycho_torch.dataloader import _get_diffraction_stack

        H, W, N = 64, 64, 100
        legacy_diffraction = np.random.rand(H, W, N).astype(np.float32)

        npz_path = self.data_path / "legacy_format.npz"
        np.savez(str(npz_path), diffraction=legacy_diffraction)

        # Act
        result = _get_diffraction_stack(npz_path)

        # Assert
        expected_shape = (N, H, W)
        self.assertEqual(
            result.shape, expected_shape,
            f"Legacy (H,W,N) format should be transposed to (N,H,W). "
            f"Got {result.shape}, expected {expected_shape}"
        )

    def test_preserves_canonical_nwh_format(self):
        """
        Test that canonical (N, H, W) format is preserved without transposition.
        """
        # Arrange: Create NPZ with canonical (N, H, W) format
        from ptycho_torch.dataloader import _get_diffraction_stack

        N, H, W = 100, 64, 64
        canonical_diffraction = np.random.rand(N, H, W).astype(np.float32)

        npz_path = self.data_path / "canonical_format.npz"
        np.savez(str(npz_path), diffraction=canonical_diffraction)

        # Act
        result = _get_diffraction_stack(npz_path)

        # Assert
        self.assertEqual(
            result.shape, (N, H, W),
            "Canonical (N,H,W) format should be preserved as-is"
        )

    def test_handles_edge_case_square_dataset(self):
        """
        Test edge case where N ≈ H ≈ W (ambiguous format).

        In this case, the heuristic should preserve the original format
        since there's no clear signal of legacy format.
        """
        # Arrange: Create NPZ where all dimensions are similar
        from ptycho_torch.dataloader import _get_diffraction_stack

        # Shape (65, 64, 64) — last dim NOT much larger, should be preserved
        N, H, W = 65, 64, 64
        ambiguous_diffraction = np.random.rand(N, H, W).astype(np.float32)

        npz_path = self.data_path / "ambiguous_format.npz"
        np.savez(str(npz_path), diffraction=ambiguous_diffraction)

        # Act
        result = _get_diffraction_stack(npz_path)

        # Assert: Should preserve original since last dim not >> first dims
        self.assertEqual(
            result.shape, (N, H, W),
            "Ambiguous format should be preserved (no auto-transpose)"
        )

    def test_works_with_diff3d_legacy_key(self):
        """
        Test auto-transpose also works with legacy 'diff3d' key.
        """
        # Arrange: Create NPZ with diff3d key in legacy format
        from ptycho_torch.dataloader import _get_diffraction_stack

        H, W, N = 64, 64, 150
        legacy_diffraction = np.random.rand(H, W, N).astype(np.float32)

        npz_path = self.data_path / "diff3d_legacy.npz"
        np.savez(str(npz_path), diff3d=legacy_diffraction)

        # Act
        result = _get_diffraction_stack(npz_path)

        # Assert
        self.assertEqual(
            result.shape, (N, H, W),
            "Legacy format should be transposed even with 'diff3d' key"
        )

    def test_real_dataset_dimensions(self):
        """
        Test with actual problematic dataset dimensions from Run1084.

        This is the exact case that triggered the IndexError:
        - Original shape: (64, 64, 1087)
        - Should transpose to: (1087, 64, 64)
        """
        # Arrange: Simulate Run1084_recon3_postPC_shrunk_3.npz dimensions
        from ptycho_torch.dataloader import _get_diffraction_stack

        H, W, N = 64, 64, 1087
        legacy_diffraction = np.random.rand(H, W, N).astype(np.float32)

        npz_path = self.data_path / "run1084_sim.npz"
        np.savez(str(npz_path), diffraction=legacy_diffraction)

        # Act
        result = _get_diffraction_stack(npz_path)

        # Assert
        self.assertEqual(
            result.shape, (1087, 64, 64),
            "Run1084-style dataset should be transposed to (1087, 64, 64)"
        )

        # Verify we can index with nn_indices that would have caused the crash
        test_indices = [0, 367, 722, 1086]  # Values that appeared in error log
        try:
            _ = result[test_indices]
        except IndexError:
            self.fail(
                f"Should be able to index transposed array with indices {test_indices}, "
                f"but got IndexError. This means the fix didn't resolve the bug."
            )

    def test_npz_headers_also_transposes_shape(self):
        """
        Test that npz_headers() returns transposed shape for legacy format.

        This is CRITICAL because npz_headers() is used to pre-allocate memory maps.
        If the shape isn't transposed here, the memory map will have wrong dimensions
        even if _get_diffraction_stack() transposes the data.

        Regression test for the RuntimeError: tensor size mismatch during assignment.
        """
        # Arrange: Create NPZ with legacy (H, W, N) format
        from ptycho_torch.dataloader import npz_headers

        H, W, N = 64, 64, 1087
        legacy_diffraction = np.random.rand(H, W, N).astype(np.float32)
        xcoords = np.random.rand(N).astype(np.float64) * 100
        ycoords = np.random.rand(N).astype(np.float64) * 100

        npz_path = self.data_path / "legacy_for_headers.npz"
        np.savez(str(npz_path), diffraction=legacy_diffraction, xcoords=xcoords, ycoords=ycoords)

        # Act
        shape, coords_x, coords_y = npz_headers(npz_path)

        # Assert: Shape should be transposed
        self.assertEqual(
            shape, (1087, 64, 64),
            f"npz_headers() must transpose legacy format for memory map allocation. "
            f"Got {shape}, expected (1087, 64, 64)"
        )

        # Verify coordinates loaded correctly
        self.assertEqual(len(coords_x), N)
        self.assertEqual(len(coords_y), N)


if __name__ == '__main__':
    unittest.main(verbosity=2)
