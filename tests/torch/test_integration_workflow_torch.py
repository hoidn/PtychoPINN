"""
Phase E2.B1 PyTorch Integration Workflow Tests (INTEGRATE-PYTORCH-001)

This module provides end-to-end integration tests for the PyTorch backend,
mirroring the TensorFlow integration test structure in tests/test_integration_workflow.py.

Test Coverage:
1. PyTorch train → save → load → infer workflow using subprocess calls
2. Artifact persistence (Lightning checkpoint, model bundle)
3. Reconstruction output validation
4. CONFIG-001 compliance (params.cfg synchronization)

Implementation Status: RED PHASE (Phase E2.B) — These tests document the expected
PyTorch workflow behavior before implementation. They will fail until Phase E2.C
wiring completes.

References:
- Phase E plan: plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md §E2
- TensorFlow baseline: tests/test_integration_workflow.py
- TEST-PYTORCH-001 plan: plans/pytorch_integration_test_plan.md
- Fixture inventory: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T213500Z/phase_e_fixture_sync.md
"""

import sys
import unittest
import subprocess
import tempfile
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class TestPyTorchIntegrationWorkflow(unittest.TestCase):
    """
    Integration tests for PyTorch backend end-to-end workflows.

    Phase: E2.B (Red Test Phase)
    Status: EXPECTED TO FAIL until Phase E2.C implementation

    These tests validate the PyTorch train→save→load→infer cycle using
    subprocess calls to mirror real user workflows across separate processes.
    """

    def setUp(self):
        """Create temporary directory for test outputs."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_path = Path(self.test_dir.name)
        print(f"\nCreated temporary directory for PyTorch test run: {self.output_path}")

    def tearDown(self):
        """Clean up temporary directory after test."""
        self.test_dir.cleanup()
        print(f"Cleaned up temporary directory: {self.output_path}")

    def test_pytorch_train_save_load_infer_cycle(self):
        """
        Tests the complete PyTorch train → save → load → infer workflow.

        This validates the PyTorch model persistence layer by simulating a real
        user workflow across separate processes, mirroring the TensorFlow integration test.

        Phase: E2.B1 (Red Test)
        Expected Behavior (Phase E2.C implementation target):
        1. Training subprocess creates Lightning checkpoint
        2. Checkpoint artifact saved to <output_dir>/checkpoints/ or <output_dir>/wts.pt
        3. Inference subprocess loads checkpoint and generates reconstructions
        4. Output images created in inference output directory

        Current Status: FAILING — ptycho_torch training/inference scripts not yet
        integrated with subprocess harness and backend dispatcher.
        """
        # --- 1. Define Paths ---
        data_file = project_root / "datasets" / "Run1084_recon3_postPC_shrunk_3.npz"
        training_output_dir = self.output_path / "training_outputs"
        inference_output_dir = self.output_path / "pytorch_output"

        # --- 2. Training Step (PyTorch) ---
        print("--- Running PyTorch Training Step (subprocess) ---")

        # NOTE: This command reflects the expected CLI interface after Phase E2.C implementation.
        # Currently, ptycho_torch/train.py does not support these exact flags.
        # The test documents the target API contract.
        train_command = [
            sys.executable, "-m", "ptycho_torch.train",
            "--train_data_file", str(data_file),
            "--test_data_file", str(data_file),
            "--output_dir", str(training_output_dir),
            "--max_epochs", "2",
            "--n_images", "64",
            "--gridsize", "1",
            "--batch_size", "4",
            "--device", "cpu",
            "--disable_mlflow",  # Suppress MLflow for CI (flag to be added per TEST-PYTORCH-001)
        ]

        # Expected to fail in Phase E2.B (red phase) because:
        # - ptycho_torch.train may not support these CLI flags yet
        # - Backend dispatcher not wired to route PyTorch workflows
        # - CONFIG-001 gate may not be enforced
        train_result = subprocess.run(train_command, capture_output=True, text=True)

        self.assertEqual(
            train_result.returncode, 0,
            f"PyTorch training script failed with stdout:\n{train_result.stdout}\nstderr:\n{train_result.stderr}"
        )

        # Check for PyTorch checkpoint artifact
        # Expected format: Lightning checkpoint or custom .pt bundle
        # Phase E2.C implementation should define exact artifact name
        checkpoint_candidates = [
            training_output_dir / "checkpoints" / "last.ckpt",  # Lightning default
            training_output_dir / "wts.pt",  # Custom bundle format
            training_output_dir / "model.pt",  # Alternative naming
        ]

        checkpoint_found = any(p.exists() for p in checkpoint_candidates)
        self.assertTrue(
            checkpoint_found,
            f"No PyTorch checkpoint found in {training_output_dir}. Searched: {[str(p) for p in checkpoint_candidates]}"
        )

        # --- 3. Inference Step (PyTorch) ---
        print("--- Running PyTorch Inference Step (subprocess) ---")

        # NOTE: ptycho_torch/inference.py does not exist yet (per TEST-PYTORCH-001 §Open Questions).
        # This test documents the expected CLI interface for Phase E2.C implementation.
        inference_command = [
            sys.executable, "-m", "ptycho_torch.inference",
            "--model_path", str(training_output_dir),
            "--test_data", str(data_file),
            "--output_dir", str(inference_output_dir),
            "--n_images", "32",
            "--device", "cpu",
        ]

        # Expected to fail in Phase E2.B because:
        # - ptycho_torch.inference module does not exist yet
        # - Inference helper needs to be authored (TEST-PYTORCH-001 §Next Steps #3)
        infer_result = subprocess.run(inference_command, capture_output=True, text=True)

        self.assertEqual(
            infer_result.returncode, 0,
            f"PyTorch inference script failed with stdout:\n{infer_result.stdout}\nstderr:\n{infer_result.stderr}"
        )

        # --- 4. Assertions on Output ---
        print("--- Verifying PyTorch Outputs ---")

        # Expected output artifact names (to be defined in Phase E2.C)
        recon_amp_path = inference_output_dir / "reconstructed_amplitude.png"
        recon_phase_path = inference_output_dir / "reconstructed_phase.png"

        self.assertTrue(recon_amp_path.exists(), "PyTorch reconstructed amplitude image not created.")
        self.assertTrue(recon_phase_path.exists(), "PyTorch reconstructed phase image not created.")

        self.assertGreater(os.path.getsize(recon_amp_path), 1000, "Amplitude image too small or empty.")
        self.assertGreater(os.path.getsize(recon_phase_path), 1000, "Phase image too small or empty.")

        print("✅ PyTorch full workflow integration test passed.")


    @unittest.skip("Deferred to Phase E2.D parity verification")
    def test_pytorch_tf_output_parity(self):
        """
        Tests that PyTorch and TensorFlow produce comparable reconstructions.

        Phase: E2.D (Parity Verification)
        Deferred: This test requires both backends to be functional.
        Implementation should compare:
        - Reconstruction metrics (PSNR, SSIM, FRC)
        - Output array shapes and dtypes
        - Runtime performance

        References:
        - Phase E2.D plan: phase_e_integration.md §E2.D1-D3
        - Parity thresholds: To be defined in phase_e_parity_summary.md
        """
        self.skipTest("Parity comparison requires Phase E2.C implementation + E2.D metrics harness")


if __name__ == '__main__':
    # Run with verbose output for debugging
    unittest.main(verbosity=2)
