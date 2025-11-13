"""
Unit tests for PyTorch CLI probe size inference (INTEGRATE-PYTORCH-001-PROBE-SIZE).

This module tests the PyTorch train.py CLI's ability to automatically infer
probe size (DataConfig.N) from NPZ metadata, eliminating hardcoded defaults
that cause tensor shape mismatches.

Test Coverage:
1. NPZ probe shape extraction utility function
2. DataConfig.N correctly derived from probeGuess metadata
3. Integration with existing npz_headers pattern

Red Phase Target: Tests should fail because utility function doesn't exist yet.

References:
- specs/data_contracts.md §1 — probeGuess schema requirement
- ptycho_torch/train.py:420 — Current hardcoded DataConfig(N=128)
- ptycho_torch/dataloader.py:29-83 — Existing npz_headers() pattern to extend
- docs/fix_plan.md#INTEGRATE-PYTORCH-001-PROBE-SIZE
- input.md Do Now #1
"""

import sys
import unittest
import tempfile
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class TestNPZProbeSizeExtraction(unittest.TestCase):
    """
    Unit tests for NPZ probe size metadata extraction utility.

    Phase: RED (TDD cycle per input.md)
    Expected to fail until ptycho_torch/train.py implements _infer_probe_size().
    """

    def setUp(self):
        """Create temporary directory and fixture NPZ files."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.data_path = Path(self.test_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.test_dir.cleanup()

    def _create_minimal_training_npz(self, filename, probe_size=64):
        """
        Helper to create minimal training NPZ fixture with specified probe size.

        Args:
            filename: NPZ file path
            probe_size: Square probe dimensions (N x N)

        Returns:
            Path to created NPZ file
        """
        n_images = 20
        H, W = probe_size, probe_size

        # Canonical diffraction key per DATA-001
        diffraction = np.random.rand(n_images, H, W).astype(np.float32) * 0.5

        # Coordinates
        xcoords = np.random.rand(n_images).astype(np.float64) * 100
        ycoords = np.random.rand(n_images).astype(np.float64) * 100

        # Probe with specified size (critical for this test)
        probeGuess = np.random.rand(H, W).astype(np.complex64)

        # Object must be larger than probe
        objectGuess = np.random.rand(H*2, W*2).astype(np.complex64)

        # Ground truth patches
        Y = np.random.rand(n_images, H, W).astype(np.complex64)

        # Scan index
        scan_index = np.arange(n_images, dtype=np.int32)

        npz_data = {
            'diffraction': diffraction,
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

    def test_infer_probe_size_from_npz(self):
        """
        Test utility function extracts probe size from NPZ metadata without loading full arrays.

        Red Phase: Fails because _infer_probe_size() function doesn't exist in train.py yet.
        Green Phase: Should pass after implementing NPZ metadata reader using zipfile approach.

        This test validates the core requirement: derive N from probeGuess.shape[0]
        efficiently (no full array load) before DataConfig instantiation.
        """
        # Create NPZ with 64x64 probe
        npz_file = self._create_minimal_training_npz("test_train.npz", probe_size=64)

        # Import the utility function (should fail in RED phase)
        try:
            from ptycho_torch.train import _infer_probe_size
        except ImportError:
            self.fail("_infer_probe_size() function not found in ptycho_torch.train module. "
                     "Implement this utility to extract probe size from NPZ metadata.")

        # Call the utility
        inferred_N = _infer_probe_size(str(npz_file))

        # CRITICAL ASSERTION: N should match actual probe dimensions
        self.assertEqual(
            inferred_N, 64,
            f"_infer_probe_size() should return probeGuess.shape[0]=64, but got {inferred_N}"
        )

    def test_infer_probe_size_128(self):
        """
        Test utility correctly handles 128x128 probe (different from 64 default).

        This ensures the function works for probe sizes other than the common 64x64.
        """
        npz_file = self._create_minimal_training_npz("test_128.npz", probe_size=128)

        try:
            from ptycho_torch.train import _infer_probe_size
        except ImportError:
            self.skip("_infer_probe_size() not implemented yet")

        inferred_N = _infer_probe_size(str(npz_file))
        self.assertEqual(inferred_N, 128)

    def test_infer_probe_size_rectangular(self):
        """
        Test utility handles rectangular probes by using first dimension.

        Some edge cases may have non-square probes (e.g., 64x32).
        Utility should use probe.shape[0] to determine N.
        """
        n_images = 10
        probe_h, probe_w = 64, 32  # Rectangular

        # Create NPZ with rectangular probe
        npz_file = self.data_path / "rect_probe.npz"
        np.savez(
            str(npz_file),
            diffraction=np.random.rand(n_images, probe_h, probe_w).astype(np.float32) * 0.5,
            xcoords=np.random.rand(n_images).astype(np.float64) * 100,
            ycoords=np.random.rand(n_images).astype(np.float64) * 100,
            probeGuess=np.random.rand(probe_h, probe_w).astype(np.complex64),
            objectGuess=np.random.rand(probe_h*2, probe_w*2).astype(np.complex64),
            Y=np.random.rand(n_images, probe_h, probe_w).astype(np.complex64),
            scan_index=np.arange(n_images, dtype=np.int32),
        )

        try:
            from ptycho_torch.train import _infer_probe_size
        except ImportError:
            self.skip("_infer_probe_size() not implemented yet")

        inferred_N = _infer_probe_size(str(npz_file))
        self.assertEqual(
            inferred_N, probe_h,
            f"For rectangular probe ({probe_h}x{probe_w}), should use shape[0]={probe_h}"
        )

    def test_infer_probe_size_missing_probe(self):
        """
        Test utility returns None (or default) when probeGuess key missing.

        This allows graceful fallback to default N=64 when NPZ doesn't provide probe.
        CLI code can then decide whether to use default or raise error.
        """
        n_images = 10
        H, W = 64, 64

        # Create NPZ without probeGuess
        npz_file = self.data_path / "no_probe.npz"
        np.savez(
            str(npz_file),
            diffraction=np.random.rand(n_images, H, W).astype(np.float32) * 0.5,
            xcoords=np.random.rand(n_images).astype(np.float64) * 100,
            ycoords=np.random.rand(n_images).astype(np.float64) * 100,
            objectGuess=np.random.rand(H*2, W*2).astype(np.complex64),
            scan_index=np.arange(n_images, dtype=np.int32),
        )

        try:
            from ptycho_torch.train import _infer_probe_size
        except ImportError:
            self.skip("_infer_probe_size() not implemented yet")

        inferred_N = _infer_probe_size(str(npz_file))

        # Should return None to signal "use default"
        self.assertIsNone(
            inferred_N,
            "When probeGuess missing, _infer_probe_size() should return None for fallback to default"
        )

    def test_infer_probe_size_real_dataset(self):
        """
        Test utility on actual project dataset (integration smoke test).

        This validates the fix works for the canonical test dataset used in
        test_integration_workflow_torch.py where the original mismatch occurred.
        """
        real_dataset = project_root / "datasets" / "Run1084_recon3_postPC_shrunk_3.npz"

        if not real_dataset.exists():
            self.skipTest(f"Real dataset not found: {real_dataset}")

        try:
            from ptycho_torch.train import _infer_probe_size
        except ImportError:
            self.skip("_infer_probe_size() not implemented yet")

        inferred_N = _infer_probe_size(str(real_dataset))

        # This dataset has 64x64 probe (per investigation subagent output)
        self.assertEqual(
            inferred_N, 64,
            f"Real dataset should have N=64 based on actual probeGuess shape, got {inferred_N}"
        )


if __name__ == '__main__':
    # Run with verbose output for TDD debugging
    unittest.main(verbosity=2)
