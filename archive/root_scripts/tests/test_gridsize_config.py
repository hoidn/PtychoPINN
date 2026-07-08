#!/usr/bin/env python3
"""
Test script to diagnose gridsize configuration issue.

The problem: config file specifies gridsize: 2 but debug output shows gridsize: 1.
This script traces through the configuration setup process to identify where the override happens.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from ptycho.config.config import ModelConfig, TrainingConfig
from ptycho.workflows.components import setup_configuration, load_yaml_config, parse_arguments

def test_yaml_loading():
    """Test 1: Check if YAML file loads correctly."""
    print("=== Test 1: YAML Loading ===")
    
    yaml_path = "configs/gridsize2_minimal.yaml"
    try:
        yaml_config = load_yaml_config(yaml_path)
        print(f"YAML config loaded: {yaml_config}")
        
        # Check if gridsize is in the YAML
        if 'model' in yaml_config:
            model_config = yaml_config['model']
            print(f"Model config from YAML: {model_config}")
            if 'gridsize' in model_config:
                print(f"  gridsize from YAML: {model_config['gridsize']}")
            else:
                print("  ERROR: gridsize not found in YAML model config!")
        else:
            print("  ERROR: 'model' key not found in YAML config!")
            
    except Exception as e:
        print(f"ERROR loading YAML: {e}")
        return False
        
    return yaml_config

def test_default_modelconfig():
    """Test 2: Check default ModelConfig values."""
    print("\n=== Test 2: Default ModelConfig ===")
    
    default_model = ModelConfig()
    print(f"Default ModelConfig: {default_model}")
    print(f"Default gridsize: {default_model.gridsize}")
    
    # Test creating ModelConfig with explicit gridsize
    custom_model = ModelConfig(gridsize=2)
    print(f"Custom ModelConfig(gridsize=2): {custom_model}")
    print(f"Custom gridsize: {custom_model.gridsize}")
    
    return default_model, custom_model

def test_argument_parsing():
    """Test 3: Check what CLI arguments are created."""
    print("\n=== Test 3: CLI Argument Parsing ===")
    
    # Create a minimal argument namespace (simulating no CLI args provided)
    args = argparse.Namespace()
    args.config = "configs/gridsize2_minimal.yaml"
    
    # Set all the fields that would be created by parse_arguments
    # First, add all TrainingConfig fields (except model)
    from dataclasses import fields
    
    for field in fields(TrainingConfig):
        if field.name != 'model':
            setattr(args, field.name, None)  # None means not provided by CLI
    
    # Add all ModelConfig fields
    for field in fields(ModelConfig):
        setattr(args, field.name, None)  # None means not provided by CLI
    
    print(f"Simulated args object: {vars(args)}")
    
    return args

def test_setup_configuration():
    """Test 4: Test the full setup_configuration process."""
    print("\n=== Test 4: Full Configuration Setup ===")
    
    # Create simulated args (no CLI overrides)
    args = test_argument_parsing()
    yaml_path = "configs/gridsize2_minimal.yaml"
    
    try:
        config = setup_configuration(args, yaml_path)
        print(f"Final TrainingConfig: {config}")
        print(f"Final model config: {config.model}")
        print(f"Final gridsize: {config.model.gridsize}")
        
        # Check if gridsize is 1 or 2
        if config.model.gridsize == 1:
            print("❌ PROBLEM FOUND: gridsize is 1 instead of 2!")
        elif config.model.gridsize == 2:
            print("✅ SUCCESS: gridsize is correctly set to 2!")
        else:
            print(f"❓ UNEXPECTED: gridsize is {config.model.gridsize}")
            
        return config
        
    except Exception as e:
        print(f"ERROR in setup_configuration: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_legacy_params():
    """Test 6: Check what's in the legacy params after configuration."""
    print("\n=== Test 6: Legacy Params System ===")
    
    try:
        import ptycho.params as params
        
        print("Legacy params.cfg keys and values:")
        for key, value in params.cfg.items():
            if 'grid' in key.lower() or 'size' in key.lower():
                print(f"  {key}: {value}")
        
        # Check specifically for gridsize
        gridsize_key = None
        for key in params.cfg:
            if key == 'gridsize' or 'gridsize' in key.lower():
                gridsize_key = key
                break
                
        if gridsize_key:
            print(f"\nFound gridsize in legacy params: {gridsize_key} = {params.cfg[gridsize_key]}")
        else:
            print("\nNo gridsize found in legacy params - this might be the issue!")
            
        # Check if there's a model.gridsize or similar
        for key in params.cfg:
            if 'model' in key and 'grid' in key:
                print(f"Model-related grid param: {key} = {params.cfg[key]}")
                
        return params.cfg
        
    except Exception as e:
        print(f"ERROR accessing legacy params: {e}")
        return None

def test_step_by_step_merge():
    """Test 5: Step-by-step configuration merge to find the issue."""
    print("\n=== Test 5: Step-by-Step Configuration Merge ===")
    
    # Load YAML
    yaml_path = "configs/gridsize2_minimal.yaml"
    yaml_config = load_yaml_config(yaml_path)
    print(f"1. YAML config: {yaml_config}")
    
    # Simulate args
    args = test_argument_parsing()
    args_config = vars(args)
    print(f"2. Args config (only non-None values): {{k: v for k, v in args_config.items() if v is not None}}")
    
    # Start merge process (copying from setup_configuration)
    merged_config = yaml_config.copy() if yaml_config else {}
    print(f"3. Initial merged_config (from YAML): {merged_config}")
    
    # Override with CLI arguments
    print("4. Checking CLI overrides...")
    for key, value in args_config.items():
        if value is not None:  # Only override if CLI arg was explicitly provided
            print(f"   CLI override: {key} = {value}")
            merged_config[key] = value
        else:
            print(f"   No CLI override for {key} (value is None)")
    
    print(f"5. Merged config after CLI overrides: {merged_config}")
    
    # Handle nested 'model' config from YAML
    model_config_dict = {}
    if 'model' in merged_config and isinstance(merged_config['model'], dict):
        model_config_dict = merged_config['model']
        print(f"6. Model config dict from YAML: {model_config_dict}")
    
    # Create ModelConfig
    from dataclasses import fields
    model_fields = {f.name for f in fields(ModelConfig)}
    print(f"7. Available model fields: {model_fields}")
    
    # Get model args from top-level merged_config
    model_args = {k: v for k, v in merged_config.items() if k in model_fields}
    print(f"8. Model args from top-level config: {model_args}")
    
    # Override with nested model config
    nested_overrides = {k: v for k, v in model_config_dict.items() if k in model_fields}
    print(f"9. Model args from nested config: {nested_overrides}")
    model_args.update(nested_overrides)
    
    print(f"10. Final model args: {model_args}")
    
    # Create ModelConfig
    model_config = ModelConfig(**model_args)
    print(f"11. Created ModelConfig: {model_config}")
    print(f"12. Final gridsize: {model_config.gridsize}")

def main():
    """Run all tests to diagnose the gridsize issue."""
    print("🔍 Diagnosing gridsize configuration issue...\n")
    
    # Test 1: YAML loading
    yaml_config = test_yaml_loading()
    if not yaml_config:
        print("❌ YAML loading failed, cannot continue")
        return
        
    # Test 2: Default values
    default_model, custom_model = test_default_modelconfig()
    
    # Test 3: Argument parsing
    args = test_argument_parsing()
    
    # Test 4: Full configuration setup
    final_config = test_setup_configuration()
    
    # Test 5: Step-by-step debug
    test_step_by_step_merge()
    
    # Test 6: Check legacy params
    legacy_cfg = test_legacy_params()
    
    print("\n" + "="*60)
    print("🏁 DIAGNOSIS SUMMARY:")
    print("="*60)
    
    if yaml_config and 'model' in yaml_config and 'gridsize' in yaml_config['model']:
        yaml_gridsize = yaml_config['model']['gridsize']
        print(f"✅ YAML file specifies gridsize: {yaml_gridsize}")
    else:
        print("❌ YAML file doesn't specify gridsize correctly")
    
    print(f"✅ ModelConfig default gridsize: {ModelConfig().gridsize}")
    
    if final_config:
        final_gridsize = final_config.model.gridsize
        print(f"{'✅' if final_gridsize == 2 else '❌'} Final config gridsize: {final_gridsize}")
        
        if final_gridsize == 2:
            print("\n💡 CONFIGURATION SYSTEM IS WORKING CORRECTLY!")
            print("   The issue must be elsewhere - possibly:")
            print("   1. CLI arguments overriding the YAML config")
            print("   2. Some code loading data with gridsize=1 hardcoded")
            print("   3. The legacy params system not being updated correctly")
            print("   4. Some other module overriding the configuration later")
        else:
            print("\n💡 LIKELY CAUSE:")
            print("   The configuration merge process is not correctly")
            print("   handling the nested 'model' section from the YAML file.")
            print("   Check the setup_configuration() function's handling")
            print("   of the model_config_dict update logic.")
    else:
        print("❌ Failed to create final configuration")
    
    # Test what happens if CLI argument overrides YAML
    print("\n🔍 TESTING POTENTIAL CLI OVERRIDE:")
    test_args = test_argument_parsing()
    test_args.gridsize = 1  # Simulate --gridsize 1 being passed
    print(f"Simulating CLI override: --gridsize 1")
    
    try:
        override_config = setup_configuration(test_args, "configs/gridsize2_minimal.yaml")
        print(f"Result with CLI override: gridsize = {override_config.model.gridsize}")
        if override_config.model.gridsize == 1:
            print("❌ CLI OVERRIDE DETECTED! This is likely the cause.")
            print("   Check the actual command being run for --gridsize arguments.")
        else:
            print("✅ CLI override test passed - not the issue.")
    except Exception as e:
        print(f"Error testing CLI override: {e}")
    
    # Additional debugging suggestions
    print("\n🔍 NEXT STEPS TO FIND THE REAL ISSUE:")
    print("1. Check if the actual command being run has CLI arguments that override gridsize")
    print("2. Check if the data loading code hardcodes gridsize=1")
    print("3. Check the legacy params system to see if gridsize is correctly translated")
    print("4. Add debug prints to the actual training script to see where gridsize changes")
    print("5. Look for any place in the code that calls ModelConfig() without arguments")
    print("   (which would default to gridsize=1)")

if __name__ == "__main__":
    main()