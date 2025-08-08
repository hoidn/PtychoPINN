#!/usr/bin/env python3
"""
Tool to generate all 8 coordinate variants of an NPZ file.

This utility creates all combinations of:
- Swap coordinates or not (2 options)
- Flip X sign or not (2 options) 
- Flip Y sign or not (2 options)

Total: 2^3 = 8 variants
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Tuple


def generate_coordinate_variants(input_file: str, output_prefix: str) -> None:
    """
    Generate all 8 coordinate variants of an NPZ file.
    
    Args:
        input_file: Path to input NPZ file
        output_prefix: Prefix for output files (without .npz extension)
    """
    print(f"Loading data from: {input_file}")
    data = np.load(input_file)
    
    # Convert to dictionary for modification
    base_data = {key: data[key] for key in data.keys()}
    
    # Check if coordinate arrays exist
    if 'xcoords' not in base_data or 'ycoords' not in base_data:
        raise ValueError("Input file must contain 'xcoords' and 'ycoords' arrays")
    
    original_x = base_data['xcoords'].copy()
    original_y = base_data['ycoords'].copy()
    
    print(f"Original xcoords range: [{original_x.min():.2f}, {original_x.max():.2f}]")
    print(f"Original ycoords range: [{original_y.min():.2f}, {original_y.max():.2f}]")
    print()
    
    # Generate all 8 combinations
    variants = []
    for swap in [False, True]:
        for flip_x in [False, True]:
            for flip_y in [False, True]:
                # Build suffix
                suffix_parts = []
                if swap:
                    suffix_parts.append("swapped")
                if flip_x:
                    suffix_parts.append("flipx")
                if flip_y:
                    suffix_parts.append("flipy")
                
                if not suffix_parts:
                    suffix = "original"
                else:
                    suffix = "_".join(suffix_parts)
                
                variants.append((swap, flip_x, flip_y, suffix))
    
    # Process each variant
    for swap, flip_x, flip_y, suffix in variants:
        # Start with original coordinates
        x_coords = original_x.copy()
        y_coords = original_y.copy()
        
        # Apply transformations
        if flip_x:
            x_coords = -x_coords
        if flip_y:
            y_coords = -y_coords
        if swap:
            x_coords, y_coords = y_coords, x_coords
        
        # Create modified data dictionary
        variant_data = base_data.copy()
        variant_data['xcoords'] = x_coords
        variant_data['ycoords'] = y_coords
        
        # Generate output filename
        output_file = f"{output_prefix}_{suffix}.npz"
        
        # Save the variant
        print(f"Creating variant: {suffix}")
        print(f"  xcoords range: [{x_coords.min():.2f}, {x_coords.max():.2f}]")
        print(f"  ycoords range: [{y_coords.min():.2f}, {y_coords.max():.2f}]")
        print(f"  Output: {output_file}")
        
        np.savez_compressed(output_file, **variant_data)
        print()
    
    print(f"Generated {len(variants)} coordinate variants successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate all 8 coordinate variants of an NPZ file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Generated variants:
    - original: No transformations
    - swapped: Swap x and y coordinates  
    - flipx: Flip x coordinates sign
    - flipy: Flip y coordinates sign
    - swapped_flipx: Swap coordinates and flip x sign
    - swapped_flipy: Swap coordinates and flip y sign
    - flipx_flipy: Flip both x and y signs
    - swapped_flipx_flipy: Swap coordinates and flip both signs

Examples:
    # Generate all variants with prefix 'train_data_variants'
    python scripts/tools/coordinate_variants_tool.py train_data.npz train_data_variants
    
    # This creates:
    # - train_data_variants_original.npz
    # - train_data_variants_swapped.npz
    # - train_data_variants_flipx.npz
    # - train_data_variants_flipy.npz
    # - train_data_variants_swapped_flipx.npz
    # - train_data_variants_swapped_flipy.npz
    # - train_data_variants_flipx_flipy.npz
    # - train_data_variants_swapped_flipx_flipy.npz
        """
    )
    
    parser.add_argument(
        "input_file", 
        help="Input NPZ file path"
    )
    parser.add_argument(
        "output_prefix", 
        help="Output file prefix (without .npz extension)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    # Create output directory if needed
    output_path = Path(args.output_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate all coordinate variants
    generate_coordinate_variants(args.input_file, args.output_prefix)


if __name__ == "__main__":
    main()