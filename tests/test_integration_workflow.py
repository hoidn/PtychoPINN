# file: tests/test_integration_workflow.py

import unittest
import pytest
import subprocess
import sys
import tempfile
from pathlib import Path
import os

# Add project root to path to ensure scripts can find the ptycho module
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

pytestmark = [pytest.mark.integration, pytest.mark.tf_integration]

class TestFullWorkflow(unittest.TestCase):

    def setUp(self):
        """Create a temporary directory for all test outputs."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_path = Path(self.test_dir.name)
        print(f"\nCreated temporary directory for test run: {self.output_path}")

    def tearDown(self):
        """Clean up the temporary directory after the test."""
        self.test_dir.cleanup()
        print(f"Cleaned up temporary directory: {self.output_path}")

    def test_train_save_load_infer_cycle(self):
        """
        Tests the complete train -> save -> load -> infer workflow.
        This is the ultimate validation of the model save/restore cycle as it
        simulates a real user workflow across separate processes.
        """
        # --- 1. Define Paths ---
        data_file = project_root / "ptycho" / "datasets" / "Run1084_recon3_postPC_shrunk_3.npz"
        training_output_dir = self.output_path / "training_outputs"
        inference_output_dir = self.output_path / "lcls_output"
        
        # --- 2. Training Step ---
        print("--- Running Training Step (subprocess) ---")
        train_command = [
            sys.executable, str(project_root / "scripts" / "training" / "train.py"),
            "--train_data_file", str(data_file),
            "--test_data_file", str(data_file),
            "--output_dir", str(training_output_dir),
            "--nepochs", "2",
            "--n_images", "64",
            "--gridsize", "1", # Explicitly set for clarity
            "--quiet"
        ]
        
        train_result = subprocess.run(train_command, capture_output=True, text=True)
        
        self.assertEqual(train_result.returncode, 0, 
                         f"Training script failed with stdout:\n{train_result.stdout}\nstderr:\n{train_result.stderr}")
        
        model_artifact_path = training_output_dir / "wts.h5.zip"
        self.assertTrue(model_artifact_path.exists(), "Model artifact was not saved after training.")
        
        # --- 3. Inference Step ---
        print("--- Running Inference Step (subprocess) ---")
        inference_command = [
            sys.executable, str(project_root / "scripts" / "inference" / "inference.py"),
            "--model_path", str(training_output_dir),
            "--test_data", str(data_file),
            "--output_dir", str(inference_output_dir),
            "--n_images", "32"
        ]
        
        infer_result = subprocess.run(inference_command, capture_output=True, text=True)
        
        self.assertEqual(infer_result.returncode, 0,
                         f"Inference script failed with stdout:\n{infer_result.stdout}\nstderr:\n{infer_result.stderr}")
        
        # --- 4. Assertions on Output ---
        print("--- Verifying Outputs ---")
        recon_amp_path = inference_output_dir / "reconstructed_amplitude.png"
        recon_phase_path = inference_output_dir / "reconstructed_phase.png"
        
        self.assertTrue(recon_amp_path.exists(), "Reconstructed amplitude image was not created.")
        self.assertTrue(recon_phase_path.exists(), "Reconstructed phase image was not created.")
        
        self.assertGreater(os.path.getsize(recon_amp_path), 1000, "Amplitude image is too small or empty.")
        self.assertGreater(os.path.getsize(recon_phase_path), 1000, "Phase image is too small or empty.")
        print("✅ Full workflow integration test passed.")

if __name__ == '__main__':
    unittest.main()
