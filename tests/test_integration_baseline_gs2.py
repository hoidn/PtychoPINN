import unittest
import subprocess
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

class TestBaselineGridsize2Integration(unittest.TestCase):
    """Integration test for baseline model with gridsize=2."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp(prefix="test_baseline_gs2_")
        self.output_dir = Path(self.test_dir) / "output"
        
    def tearDown(self):
        """Clean up test artifacts."""
        # Clean up the temporary directory
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_baseline_gridsize2_end_to_end(self):
        """Test that baseline model runs successfully with gridsize=2."""
        
        # Check if test data exists
        test_data_path = project_root / "datasets" / "fly" / "fly001_transposed.npz"
        if not test_data_path.exists():
            self.skipTest(f"Test data not found at {test_data_path}")
        
        # Build the command
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "run_baseline.py"),
            "--train_data_file", str(test_data_path),
            "--test_data_file", str(test_data_path),
            "--gridsize", "2",
            "--n_groups", "128",  # Use small number for quick test
            "--nepochs", "2",  # Quick test with few epochs
            "--output_dir", str(self.output_dir),
            "--quiet"  # Suppress verbose output
        ]
        
        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )
        
        # Check for successful completion
        self.assertEqual(result.returncode, 0,
                        f"Baseline script failed with return code {result.returncode}\n"
                        f"stdout: {result.stdout}\n"
                        f"stderr: {result.stderr}")
        
        # Verify output files were created
        # The script creates a timestamped subdirectory, so we need to find it
        timestamped_dirs = [d for d in self.output_dir.iterdir() if d.is_dir() and 'baseline_gs' in d.name]
        self.assertTrue(len(timestamped_dirs) > 0, 
                       f"No timestamped baseline directory found in {self.output_dir}")
        
        # Use the first (and should be only) timestamped directory
        baseline_dir = timestamped_dirs[0]
        
        # Check for files that the baseline script actually creates
        expected_files = [
            baseline_dir / "baseline_model.h5",
            baseline_dir / "amp_recon.png",  # This is what save_recons() actually creates
            baseline_dir / "phi_recon.png"   # This too
        ]
        
        for expected_file in expected_files:
            self.assertTrue(expected_file.exists(),
                          f"Expected output file not created: {expected_file}")
        
        # Check that no errors about channel mismatch appear in output
        self.assertNotIn("shape mismatch", result.stderr.lower())
        self.assertNotIn("incompatible", result.stderr.lower())
        self.assertNotIn("ValueError", result.stderr)
        
        # Verify the log mentions flattening for gridsize=2
        log_file = baseline_dir / "logs" / "debug.log"
        if log_file.exists():
            with open(log_file, 'r') as f:
                log_content = f.read()
                self.assertIn("Flattening 4 channels to independent samples", log_content,
                            "Log should mention channel flattening for gridsize=2")

if __name__ == '__main__':
    unittest.main()