#!/usr/bin/env python3
"""
Test script to trace the full pipeline from TrainingConfig through to grouped data generation.
This will help identify where n_groups might be getting lost or overwritten.
"""

import sys
import os
sys.path.append('/home/ollie/Documents/PtychoPINN')

import numpy as np
from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
from ptycho.workflows.components import load_data, create_ptycho_data_container
from ptycho.loader import RawData

def trace_full_pipeline():
    """Trace the full pipeline to see where n_groups gets modified."""
    
    print("=" * 80)
    print("FULL PIPELINE TRACE - TRACKING N_GROUPS")
    print("=" * 80)
    
    # Step 1: Create TrainingConfig
    print("\n1. CREATING TRAINING CONFIG")
    print("-" * 40)
    
    # Create ModelConfig first
    model_config = ModelConfig(N=64, gridsize=1)
    
    config = TrainingConfig(
        model=model_config,
        n_subsample=128,
        n_groups=1024,
        neighbor_count=7,
        train_data_file="datasets/fly/fly001_transposed.npz",  # Use a known good file
        output_dir="test_pipeline_output",
        nepochs=1  # Just one epoch for testing
    )
    
    print(f"Initial config.n_groups = {config.n_groups}")
    print(f"Initial config.n_subsample = {config.n_subsample}")
    print(f"Initial config.neighbor_count = {config.neighbor_count}")
    
    # Step 2: Update legacy params (this is what training scripts do)
    print("\n2. UPDATING LEGACY PARAMS")
    print("-" * 40)
    
    from ptycho import params
    update_legacy_dict(params.cfg, config)
    
    print(f"After update_legacy_params:")
    print(f"  params.get('n_groups') = {params.get('n_groups')}")
    print(f"  params.get('n_subsample') = {params.get('n_subsample')}")
    print(f"  params.get('neighbor_count') = {params.get('neighbor_count')}")
    print(f"  config.n_groups still = {config.n_groups}")
    
    # Step 3: Load raw data
    print("\n3. LOADING RAW DATA")
    print("-" * 40)
    
    print(f"Calling load_data with:")
    print(f"  file_path = {config.train_data_file}")
    print(f"  n_subsample = {config.n_subsample}")
    print(f"  config.n_groups = {config.n_groups}")
    
    try:
        raw_data = load_data(config.train_data_file, n_subsample=config.n_subsample)
        print(f"Successfully loaded raw data")
        print(f"  raw_data.diff3d shape = {raw_data.diff3d.shape}")
        print(f"  raw_data.xcoords shape = {raw_data.xcoords.shape}")
        print(f"  raw_data.ycoords shape = {raw_data.ycoords.shape}")
        
        # Check if raw_data has any n_groups info
        if hasattr(raw_data, 'n_groups'):
            print(f"  raw_data.n_groups = {raw_data.n_groups}")
        else:
            print(f"  raw_data has no n_groups attribute")
            
    except Exception as e:
        print(f"ERROR in load_data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Create PtychoDataContainer
    print("\n4. CREATING PTYCHO DATA CONTAINER")
    print("-" * 40)
    
    print(f"Calling create_ptycho_data_container with:")
    print(f"  config.n_groups = {config.n_groups}")
    print(f"  config.n_subsample = {config.n_subsample}")
    print(f"  config.neighbor_count = {config.neighbor_count}")
    
    try:
        # Let's also check what parameters are passed to create_ptycho_data_container
        print(f"Parameters being passed to create_ptycho_data_container:")
        print(f"  raw_data = {type(raw_data)}")
        print(f"  config = {type(config)}")
        
        # Check params dict values right before the call
        print(f"Current params dict values:")
        print(f"  params.get('n_groups') = {params.get('n_groups')}")
        print(f"  params.get('n_subsample') = {params.get('n_subsample')}")
        
        ptycho_data = create_ptycho_data_container(raw_data, config)
        
        print(f"Successfully created PtychoDataContainer")
        print(f"  ptycho_data.X shape = {ptycho_data.X.shape}")
        print(f"  ptycho_data.Y_I shape = {ptycho_data.Y_I.shape}")
        print(f"  ptycho_data.Y_phi shape = {ptycho_data.Y_phi.shape}")
        
        if hasattr(ptycho_data, 'n_groups'):
            print(f"  ptycho_data.n_groups = {ptycho_data.n_groups}")
        else:
            print(f"  ptycho_data has no n_groups attribute")
            
        # Check the actual number of training samples
        n_train_samples = ptycho_data.X.shape[0]
        print(f"  Actual number of training samples = {n_train_samples}")
        
        if hasattr(ptycho_data, 'groups_info'):
            print(f"  ptycho_data.groups_info = {ptycho_data.groups_info}")
            
    except Exception as e:
        print(f"ERROR in create_ptycho_data_container: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Let's also check if we can trace into generate_grouped_data
    print("\n5. TRACING INTO GENERATE_GROUPED_DATA")
    print("-" * 40)
    
    try:
        from ptycho.loader import generate_grouped_data
        
        # Let's see what parameters would be passed to generate_grouped_data
        # We need to look at the source to see how it's called
        print("Looking for calls to generate_grouped_data...")
        
        # Check current params state
        print(f"Current params state:")
        for key in ['n_groups', 'n_subsample', 'neighbor_count', 'gridsize']:
            print(f"  params.get('{key}') = {params.get(key)}")
            
        # Let's see what the actual data dimensions are
        print(f"\nActual data dimensions:")
        print(f"  Training diffraction shape: {raw_data.train_data['diffraction'].shape}")
        print(f"  Number of images available: {raw_data.train_data['diffraction'].shape[0]}")
        
        # Calculate what n_groups should be based on available data
        total_images = raw_data.train_data['diffraction'].shape[0]
        requested_n_groups = config.n_groups
        print(f"  Requested n_groups: {requested_n_groups}")
        print(f"  Available images: {total_images}")
        
        if requested_n_groups > total_images:
            print(f"  WARNING: Requested n_groups ({requested_n_groups}) > available images ({total_images})")
            print(f"  This might cause n_groups to be clamped to {total_images}")
        
    except Exception as e:
        print(f"ERROR in generate_grouped_data tracing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("PIPELINE TRACE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    trace_full_pipeline()