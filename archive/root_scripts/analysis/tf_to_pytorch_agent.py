#!/usr/bin/env python3
"""
Agent Implementation for TF to PyTorch Training Equivalence
Configures and executes PyTorch training script to be functionally equivalent to TensorFlow command.
"""

import os
import sys
import logging
import shutil
import numpy as np
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Phase 0: Dataset Selection and Verification
    logger.info("Phase 0: Dataset Selection and Verification")
    
    # 0.1. Define Available Datasets
    DATASET_OPTIONS = [
        './datasets/fly64/fly001_64_train.npz',
        './datasets/fly64/fly001_64_train_converted.npz',
        './datasets/fly/fly001.npz'
    ]
    logger.info(f"Available dataset options: {DATASET_OPTIONS}")
    
    # 0.2. Select and Verify Active Dataset
    SOURCE_DATA_FILE = None
    for path in DATASET_OPTIONS:
        if os.path.exists(path):
            SOURCE_DATA_FILE = path
            logger.info(f"Selected dataset: {SOURCE_DATA_FILE}")
            break
    
    if SOURCE_DATA_FILE is None:
        logger.error("No valid source dataset was found!")
        sys.exit(1)
    
    # Phase 1: Environment Setup and Parameter Extraction
    logger.info("Phase 1: Environment Setup and Parameter Extraction")
    
    # 1.1. Parse Source Command
    GRID_SIZE = 2
    EXPERIMENT_NAME = "gridsize2_training_run"
    logger.info(f"Grid size: {GRID_SIZE}, Experiment name: {EXPERIMENT_NAME}")
    
    # 1.2. Define Target Paths (Dynamically)
    BASENAME = os.path.splitext(os.path.basename(SOURCE_DATA_FILE))[0]
    TARGET_PTYCHO_DIR = f'datasets/{BASENAME}_ptycho_agent'
    TARGET_PROBE_DIR = f'datasets/{BASENAME}_probes_agent'
    TARGET_DATA_FILE = os.path.join(TARGET_PTYCHO_DIR, os.path.basename(SOURCE_DATA_FILE))
    
    logger.info(f"Target basename: {BASENAME}")
    logger.info(f"Target ptycho dir: {TARGET_PTYCHO_DIR}")
    logger.info(f"Target probe dir: {TARGET_PROBE_DIR}")
    logger.info(f"Target data file: {TARGET_DATA_FILE}")
    
    # Phase 2: Data Preparation
    logger.info("Phase 2: Data Preparation")
    
    # 2.1. Create target directories
    os.makedirs(TARGET_PTYCHO_DIR, exist_ok=True)
    os.makedirs(TARGET_PROBE_DIR, exist_ok=True)
    logger.info(f"Created directories: {TARGET_PTYCHO_DIR}, {TARGET_PROBE_DIR}")
    
    # 2.2. Copy source dataset to target ptycho directory
    if not os.path.exists(TARGET_DATA_FILE):
        shutil.copy2(SOURCE_DATA_FILE, TARGET_DATA_FILE)
        logger.info(f"Copied {SOURCE_DATA_FILE} to {TARGET_DATA_FILE}")
    else:
        logger.info(f"Target data file already exists: {TARGET_DATA_FILE}")
    
    # 2.3. Load and inspect source dataset structure
    with np.load(SOURCE_DATA_FILE) as data:
        keys = list(data.keys())
        logger.info(f"Dataset keys: {keys}")
        for key in keys:
            shape = data[key].shape
            dtype = data[key].dtype
            logger.info(f"  {key}: shape={shape}, dtype={dtype}")
    
    # 2.4. Use existing probe
    EXISTING_PROBE_FILE = './datasets/probes/fly001.npz'
    if os.path.exists(EXISTING_PROBE_FILE):
        TARGET_PROBE_FILE = os.path.join(TARGET_PROBE_DIR, 'fly001.npz')
        if not os.path.exists(TARGET_PROBE_FILE):
            shutil.copy2(EXISTING_PROBE_FILE, TARGET_PROBE_FILE)
            logger.info(f"Copied existing probe to {TARGET_PROBE_FILE}")
        else:
            logger.info(f"Target probe file already exists: {TARGET_PROBE_FILE}")
        
        with np.load(EXISTING_PROBE_FILE) as probe_data:
            probe_keys = list(probe_data.keys())
            logger.info(f"Probe keys: {probe_keys}")
            for key in probe_keys:
                shape = probe_data[key].shape
                dtype = probe_data[key].dtype
                logger.info(f"  {key}: shape={shape}, dtype={dtype}")
    else:
        logger.error(f"Existing probe file not found: {EXISTING_PROBE_FILE}")
        sys.exit(1)
    
    # Phase 3: PyTorch Training Configuration and Execution
    logger.info("Phase 3: PyTorch Training Configuration and Execution")
    
    # 3.1. Locate PyTorch training script
    PYTORCH_TRAIN_SCRIPT = './ptycho_torch/train.py'
    if not os.path.exists(PYTORCH_TRAIN_SCRIPT):
        logger.error(f"PyTorch training script not found: {PYTORCH_TRAIN_SCRIPT}")
        sys.exit(1)
    logger.info(f"Found PyTorch training script: {PYTORCH_TRAIN_SCRIPT}")
    
    # 3.2. Configure training parameters to match TF command
    # The PyTorch script takes --ptycho_dir and --probe_dir as arguments
    training_command = [
        'python', PYTORCH_TRAIN_SCRIPT,
        '--ptycho_dir', TARGET_PTYCHO_DIR,
        '--probe_dir', TARGET_PROBE_DIR
    ]
    logger.info(f"Training command: {' '.join(training_command)}")
    
    # 3.3. Execute PyTorch training script
    try:
        logger.info("Starting PyTorch training...")
        result = subprocess.run(training_command, 
                              capture_output=True, 
                              text=True, 
                              cwd=os.getcwd())
        
        if result.returncode == 0:
            logger.info("PyTorch training completed successfully!")
            logger.info(f"Training stdout:\n{result.stdout}")
        else:
            logger.error(f"PyTorch training failed with return code {result.returncode}")
            logger.error(f"Training stderr:\n{result.stderr}")
            logger.error(f"Training stdout:\n{result.stdout}")
            
    except Exception as e:
        logger.error(f"Failed to execute PyTorch training: {str(e)}")
        sys.exit(1)
    
    return {
        'source_data_file': SOURCE_DATA_FILE,
        'grid_size': GRID_SIZE,
        'experiment_name': EXPERIMENT_NAME,
        'basename': BASENAME,
        'target_ptycho_dir': TARGET_PTYCHO_DIR,
        'target_probe_dir': TARGET_PROBE_DIR,
        'target_data_file': TARGET_DATA_FILE,
        'target_probe_file': TARGET_PROBE_FILE,
        'training_command': ' '.join(training_command),
        'training_result': result.returncode
    }

if __name__ == "__main__":
    result = main()
    print("=" * 80)
    print("TF to PyTorch Training Equivalence Agent - COMPLETED!")
    print("=" * 80)
    print(f"Selected dataset: {result['source_data_file']}")
    print(f"Grid size: {result['grid_size']}")
    print(f"Experiment name: {result['experiment_name']}")
    print(f"Target ptycho dir: {result['target_ptycho_dir']}")
    print(f"Target probe dir: {result['target_probe_dir']}")
    print(f"Training command executed: {result['training_command']}")
    print(f"Training result: {'SUCCESS' if result['training_result'] == 0 else 'FAILED'}")
    print("=" * 80)