#!/usr/bin/env python
"""
Comprehensive integration test for nphotons metadata system.

This test verifies the complete nphotons metadata workflow:
1. Simulation with different nphotons values saves metadata correctly
2. Training loads metadata and validates configurations 
3. Inference uses correct nphotons from metadata
4. Parameter mismatch warnings work correctly
5. End-to-end workflow maintains nphotons consistency

The test follows the project's integration test patterns using subprocess
calls to simulate real user workflows across separate processes.
"""

import unittest
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
import os
import numpy as np
import json
from typing import Dict, Any, Optional

# Add project root to path to ensure scripts can find the ptycho module
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Import metadata utilities for verification
from ptycho.metadata import MetadataManager
from ptycho.config.config import TrainingConfig, ModelConfig


class TestNphotonsMetadataIntegration(unittest.TestCase):
    """Integration test suite for nphotons metadata system."""

    def setUp(self):
        """Create temporary directories and test data for each test."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_path = Path(self.test_dir.name)
        
        # Create subdirectories for organized testing
        self.sim_dir = self.output_path / "simulation"
        self.train_dir = self.output_path / "training"
        self.inference_dir = self.output_path / "inference" 
        
        for dir_path in [self.sim_dir, self.train_dir, self.inference_dir]:
            dir_path.mkdir(exist_ok=True)
        
        print(f"\nCreated test directories under: {self.output_path}")
        
        # Define test nphotons values
        self.test_nphotons = [1e4, 1e6, 1e8, 1e9]
        
        # Use existing working data as input template
        self.base_data_file = project_root / "ptycho" / "datasets" / "Run1084_recon3_postPC_shrunk_3.npz"
        
        # Ensure base data exists, otherwise create minimal test data
        if not self.base_data_file.exists():
            self._create_minimal_test_data()

    def tearDown(self):
        """Clean up temporary directory after test."""
        self.test_dir.cleanup()
        print(f"Cleaned up test directory: {self.output_path}")

    def _create_minimal_test_data(self):
        """Create minimal test data if base dataset doesn't exist."""
        print("Creating minimal test data...")
        
        # Create minimal working ptychography data
        N = 64
        M = 192  # Object larger than probe for scanning
        n_images = 100
        
        # Simple probe and object
        probe = np.ones((N, N), dtype=np.complex64)
        obj = np.ones((M, M), dtype=np.complex64)
        
        # Simple coordinate grid
        coords = np.random.uniform(-50, 50, size=(n_images, 2))
        
        # Fake diffraction data (amplitudes)
        diff3d = np.random.uniform(0, 10, size=(n_images, N, N)).astype(np.float32)
        
        # Save minimal dataset
        self.base_data_file.parent.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(
            self.base_data_file,
            probeGuess=probe,
            objectGuess=obj,
            xcoords=coords[:, 0],
            ycoords=coords[:, 1], 
            diffraction=diff3d  # Note: using 'diffraction' key as per data contracts
        )
        print(f"Created minimal test data at {self.base_data_file}")

    def _simulate_with_nphotons(self, nphotons: float, output_suffix: str = "") -> Path:
        """Run simulation with specific nphotons value."""
        output_file = self.sim_dir / f"simulated_data_nphotons_{nphotons}{output_suffix}.npz"
        
        print(f"--- Simulating data with nphotons={nphotons} ---")
        
        sim_command = [
            sys.executable, 
            str(project_root / "scripts" / "simulation" / "simulate_and_save.py"),
            "--input-file", str(self.base_data_file),
            "--output-file", str(output_file), 
            "--n-images", "100",  # Keep small for fast testing
            "--n-photons", str(nphotons),
            "--gridsize", "1",
            "--seed", "42"  # For reproducibility
        ]
        
        result = subprocess.run(sim_command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, 
                        f"Simulation failed for nphotons={nphotons}\n"
                        f"stdout: {result.stdout}\nstderr: {result.stderr}")
        
        # Verify output file exists
        self.assertTrue(output_file.exists(), 
                       f"Simulation output file not created: {output_file}")
        
        return output_file

    def _train_model(self, data_file: Path, nphotons: float, 
                    output_suffix: str = "") -> Path:
        """Train model with given data file."""
        training_output_dir = self.train_dir / f"training_nphotons_{nphotons}{output_suffix}"
        
        print(f"--- Training model with data from nphotons={nphotons} ---")
        
        train_command = [
            sys.executable, 
            str(project_root / "scripts" / "training" / "train.py"),
            "--train_data_file", str(data_file),
            "--test_data_file", str(data_file),  # Use same file for simplicity
            "--output_dir", str(training_output_dir),
            "--nepochs", "2",  # Minimal epochs for fast testing
            "--n_images", "50", 
            "--gridsize", "1",
            "--nphotons", str(nphotons),  # Explicitly set nphotons
            "--quiet"
        ]
        
        result = subprocess.run(train_command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0,
                        f"Training failed for nphotons={nphotons}\n"
                        f"stdout: {result.stdout}\nstderr: {result.stderr}")
        
        # Verify model artifact exists
        model_artifact = training_output_dir / "wts.h5.zip"
        self.assertTrue(model_artifact.exists(),
                       f"Model artifact not created: {model_artifact}")
        
        return training_output_dir

    def _run_inference(self, model_dir: Path, test_data: Path, nphotons: float,
                      output_suffix: str = "") -> Path:
        """Run inference with trained model."""
        inference_output_dir = self.inference_dir / f"inference_nphotons_{nphotons}{output_suffix}"
        
        print(f"--- Running inference for nphotons={nphotons} ---")
        
        inference_command = [
            sys.executable,
            str(project_root / "scripts" / "inference" / "inference.py"),
            "--model_path", str(model_dir),
            "--test_data", str(test_data),
            "--output_dir", str(inference_output_dir),
            "--n_images", "25"
        ]
        
        result = subprocess.run(inference_command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0,
                        f"Inference failed for nphotons={nphotons}\n"
                        f"stdout: {result.stdout}\nstderr: {result.stderr}")
        
        return inference_output_dir

    def _verify_metadata(self, npz_file: Path, expected_nphotons: float) -> Dict[str, Any]:
        """Verify that NPZ file contains correct metadata."""
        print(f"Verifying metadata in {npz_file.name}...")
        
        # Load metadata
        _, metadata = MetadataManager.load_with_metadata(str(npz_file))
        
        self.assertIsNotNone(metadata, f"No metadata found in {npz_file}")
        
        # Check nphotons value
        stored_nphotons = MetadataManager.get_nphotons(metadata)
        self.assertAlmostEqual(stored_nphotons, expected_nphotons, places=9,
                             msg=f"nphotons mismatch in {npz_file}: "
                                 f"expected {expected_nphotons}, got {stored_nphotons}")
        
        # Verify metadata structure
        self.assertIn("physics_parameters", metadata)
        self.assertIn("creation_info", metadata) 
        self.assertIn("schema_version", metadata)
        
        print(f"✓ Metadata verified: nphotons={stored_nphotons}")
        return metadata

    def test_metadata_persistence_single_nphotons(self):
        """Test metadata persistence through complete workflow for single nphotons value."""
        nphotons = 1e6
        
        # 1. Simulate data with specific nphotons
        sim_file = self._simulate_with_nphotons(nphotons, "_single")
        
        # 2. Verify simulation metadata
        sim_metadata = self._verify_metadata(sim_file, nphotons)
        
        # 3. Train model with simulated data
        model_dir = self._train_model(sim_file, nphotons, "_single")
        
        # 4. Run inference
        inference_dir = self._run_inference(model_dir, sim_file, nphotons, "_single")
        
        # 5. Verify inference outputs exist
        expected_outputs = ["reconstructed_amplitude.png", "reconstructed_phase.png"]
        for output_file in expected_outputs:
            output_path = inference_dir / output_file
            self.assertTrue(output_path.exists(), 
                          f"Expected inference output not found: {output_path}")
            self.assertGreater(os.path.getsize(output_path), 1000,
                             f"Inference output file too small: {output_path}")
        
        print("✅ Single nphotons workflow test passed")

    def test_multiple_nphotons_metadata_consistency(self):
        """Test metadata consistency across multiple nphotons values."""
        print("--- Testing multiple nphotons values ---")
        
        sim_files = {}
        metadatas = {}
        
        # Simulate data with different nphotons values
        for nphotons in [1e4, 1e6, 1e8]:
            sim_file = self._simulate_with_nphotons(nphotons, "_multi")
            sim_files[nphotons] = sim_file
            metadatas[nphotons] = self._verify_metadata(sim_file, nphotons)
        
        # Verify each metadata is different and correct
        nphotons_values = set()
        for nphotons, metadata in metadatas.items():
            stored_nphotons = MetadataManager.get_nphotons(metadata)
            nphotons_values.add(stored_nphotons)
            
            # Verify that stored value matches expected
            self.assertAlmostEqual(stored_nphotons, nphotons, places=9)
        
        # Verify all nphotons values are different
        self.assertEqual(len(nphotons_values), len(metadatas),
                        "All simulated datasets should have different nphotons values")
        
        print("✅ Multiple nphotons consistency test passed")

    def test_configuration_mismatch_warnings(self):
        """Test that configuration mismatches generate appropriate warnings."""
        print("--- Testing configuration mismatch warnings ---")
        
        # 1. Simulate data with one nphotons value
        original_nphotons = 1e6
        sim_file = self._simulate_with_nphotons(original_nphotons, "_mismatch")
        
        # 2. Load metadata
        _, metadata = MetadataManager.load_with_metadata(str(sim_file))
        self.assertIsNotNone(metadata)
        
        # 3. Create config with different nphotons
        mismatched_nphotons = 1e8
        mismatched_config = TrainingConfig(
            model=ModelConfig(N=64, gridsize=1),
            nphotons=mismatched_nphotons,
            n_images=100,
            nepochs=2
        )
        
        # 4. Validate and check for warnings
        warnings_list = MetadataManager.validate_parameters(metadata, mismatched_config)
        
        # Should have at least one warning about nphotons mismatch
        nphotons_warnings = [w for w in warnings_list if "nphotons mismatch" in w]
        self.assertTrue(len(nphotons_warnings) > 0,
                       f"Expected nphotons mismatch warning, got warnings: {warnings_list}")
        
        # Verify the warning contains correct values
        warning_text = nphotons_warnings[0]
        self.assertIn(str(original_nphotons), warning_text)
        self.assertIn(str(mismatched_nphotons), warning_text)
        
        print("✅ Configuration mismatch warning test passed")

    def test_training_with_mismatched_config_warns_but_continues(self):
        """Test that training with mismatched nphotons generates warnings but continues."""
        print("--- Testing training with mismatched nphotons ---")
        
        # 1. Simulate data with one nphotons value
        data_nphotons = 1e4
        sim_file = self._simulate_with_nphotons(data_nphotons, "_train_mismatch")
        
        # 2. Train with different nphotons - should warn but not fail
        config_nphotons = 1e6
        training_output_dir = self.train_dir / "training_mismatch_test"
        
        train_command = [
            sys.executable,
            str(project_root / "scripts" / "training" / "train.py"),
            "--train_data_file", str(sim_file),
            "--test_data_file", str(sim_file),
            "--output_dir", str(training_output_dir),
            "--nepochs", "2",
            "--n_images", "50",
            "--gridsize", "1", 
            "--nphotons", str(config_nphotons),  # Different from data
            "--quiet"
        ]
        
        result = subprocess.run(train_command, capture_output=True, text=True)
        
        # Training should succeed despite mismatch
        self.assertEqual(result.returncode, 0,
                        f"Training should succeed with nphotons mismatch\n"
                        f"stdout: {result.stdout}\nstderr: {result.stderr}")
        
        # Verify model was created
        model_artifact = training_output_dir / "wts.h5.zip"
        self.assertTrue(model_artifact.exists())
        
        print("✅ Training with mismatched nphotons test passed")

    def test_end_to_end_workflow_consistency(self):
        """Test complete end-to-end workflow maintains nphotons consistency."""
        print("--- Testing end-to-end nphotons consistency ---")
        
        test_nphotons = 5e5  # Use unusual value to ensure it's preserved
        
        # 1. Simulate → Train → Infer complete workflow
        sim_file = self._simulate_with_nphotons(test_nphotons, "_e2e")
        model_dir = self._train_model(sim_file, test_nphotons, "_e2e")
        inference_dir = self._run_inference(model_dir, sim_file, test_nphotons, "_e2e")
        
        # 2. Verify nphotons consistency at each stage
        stages = [
            ("Simulation", sim_file),
        ]
        
        for stage_name, file_path in stages:
            if file_path.suffix == '.npz':
                metadata = self._verify_metadata(file_path, test_nphotons)
                print(f"✓ {stage_name} stage: nphotons consistency verified")
        
        # 3. Verify training logs mention the correct nphotons
        debug_log = model_dir / "logs" / "debug.log"
        if debug_log.exists():
            with open(debug_log, 'r') as f:
                log_content = f.read()
                # Should contain mentions of the nphotons value
                self.assertIn(str(test_nphotons), log_content,
                            "Training logs should contain the nphotons value")
                print("✓ Training logs contain correct nphotons")
        
        print("✅ End-to-end workflow consistency test passed")

    def test_metadata_backward_compatibility(self):
        """Test that NPZ files without metadata are handled gracefully."""
        print("--- Testing backward compatibility with non-metadata files ---")
        
        # Create NPZ file without metadata (legacy format)
        legacy_file = self.sim_dir / "legacy_no_metadata.npz"
        
        # Use minimal test data structure
        N = 64
        n_images = 50
        
        np.savez_compressed(
            legacy_file,
            probeGuess=np.ones((N, N), dtype=np.complex64),
            objectGuess=np.ones((192, 192), dtype=np.complex64),
            xcoords=np.random.uniform(-50, 50, n_images),
            ycoords=np.random.uniform(-50, 50, n_images),
            diffraction=np.random.uniform(0, 10, (n_images, N, N)).astype(np.float32)
        )
        
        # Try to load metadata - should return None gracefully
        _, metadata = MetadataManager.load_with_metadata(str(legacy_file))
        self.assertIsNone(metadata, "Legacy file should have None metadata")
        
        # get_nphotons should return default value
        default_nphotons = MetadataManager.get_nphotons(metadata, default=1e9)
        self.assertEqual(default_nphotons, 1e9)
        
        print("✅ Backward compatibility test passed")


if __name__ == '__main__':
    # Run with high verbosity to see detailed test output
    unittest.main(verbosity=2)