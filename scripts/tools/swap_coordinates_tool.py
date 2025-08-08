#!/usr/bin/env python3
"""
Tool to swap xcoords and ycoords in an NPZ file.

This utility swaps the xcoords and ycoords arrays in a ptychography dataset,
which can be useful for debugging coordinate system issues or testing
coordinate transformations.
"""

import argparse
import numpy as np
from pathlib import Path


def swap_coordinates(input_file: str, output_file: str) -> None:
    """
    Swap xcoords and ycoords in an NPZ file.
    
    Args:
        input_file: Path to input NPZ file
        output_file: Path to output NPZ file
    """
    print(f"Loading data from: {input_file}")
    data = np.load(input_file)
    
    # Convert to dictionary for modification
    data_dict = {key: data[key] for key in data.keys()}
    
    # Check if coordinate arrays exist
    if 'xcoords' not in data_dict or 'ycoords' not in data_dict:
        raise ValueError("Input file must contain 'xcoords' and 'ycoords' arrays")
    
    print(f"Original xcoords range: [{data_dict['xcoords'].min():.2f}, {data_dict['xcoords'].max():.2f}]")
    print(f"Original ycoords range: [{data_dict['ycoords'].min():.2f}, {data_dict['ycoords'].max():.2f}]")
    
    # Swap the coordinates
    temp = data_dict['xcoords'].copy()
    data_dict['xcoords'] = data_dict['ycoords'].copy()
    data_dict['ycoords'] = temp
    
    print(f"Swapped xcoords range: [{data_dict['xcoords'].min():.2f}, {data_dict['xcoords'].max():.2f}]")
    print(f"Swapped ycoords range: [{data_dict['ycoords'].min():.2f}, {data_dict['ycoords'].max():.2f}]")
    
    # Save the modified data
    print(f"Saving swapped data to: {output_file}")
    np.savez_compressed(output_file, **data_dict)
    
    print("Coordinate swap completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Swap xcoords and ycoords in an NPZ file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Swap coordinates and add '_swapped' suffix
    python scripts/tools/swap_coordinates_tool.py input.npz output_swapped.npz
    
    # Process multiple files
    python scripts/tools/swap_coordinates_tool.py train_data.npz train_data_swapped.npz
    python scripts/tools/swap_coordinates_tool.py test_data.npz test_data_swapped.npz
        """
    )
    
    parser.add_argument(
        "input_file", 
        help="Input NPZ file path"
    )
    parser.add_argument(
        "output_file", 
        help="Output NPZ file path"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    # Create output directory if needed
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Perform the coordinate swap
    swap_coordinates(args.input_file, args.output_file)


if __name__ == "__main__":
    main()