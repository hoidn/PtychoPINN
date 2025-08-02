#!/usr/bin/env python3
"""Apply the efficient gridsize > 1 sampling patch and run simulation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Apply the patch before importing anything else
from ptycho.raw_data_efficient import patch_raw_data_class
patch_raw_data_class()

# Now run the simulation
if __name__ == "__main__":
    import subprocess
    
    # Get command line arguments
    args = sys.argv[1:]
    
    if not args:
        print("Usage: python fix_gridsize_performance.py <simulate_and_save.py arguments>")
        print("\nExample:")
        print("python fix_gridsize_performance.py \\")
        print("    --input-file probe_study_lines_data/simulated_data.npz \\")
        print("    --probe-file probe_study_FULL_gs2/default_probe.npy \\")
        print("    --output-file probe_study_FULL_gs2/gs2_default/simulated_data.npz \\")
        print("    --n-images 5000 --gridsize 2")
        sys.exit(1)
    
    # Run the simulation script with the patch applied
    cmd = [sys.executable, "scripts/simulation/simulate_and_save.py"] + args
    print(f"Running with efficient sampling: {' '.join(cmd)}")
    subprocess.run(cmd)