# generate_coord_variants.py

import numpy as np
import os

def generate_variants():
    """
    Loads a base ptychography dataset and generates 8 variants with manipulated
    coordinate systems.
    """
    source_path = 'datasets/fly64/fly001_64_train_converted.npz'
    output_dir = 'datasets/fly64_coord_variants'

    print(f"Loading base dataset from: {source_path}")
    if not os.path.exists(source_path):
        print(f"ERROR: Source file not found at {source_path}")
        return
        
    base_data = np.load(source_path)
    
    # It's crucial to transform both coordinate sets if they exist
    x, y = base_data['xcoords'], base_data['ycoords']
    x_start, y_start = base_data.get('xcoords_start', x), base_data.get('ycoords_start', y)

    # Define the 8 coordinate transformations
    transformations = {
        'identity':      lambda x, y: (x, y),
        'flip_x':        lambda x, y: (-x, y),
        'flip_y':        lambda x, y: (x, -y),
        'flip_xy':       lambda x, y: (-x, -y),
        'swap_xy':       lambda x, y: (y, x),
        'swap_flip_x':   lambda x, y: (y, -x), # Swap then flip new Y
        'swap_flip_y':   lambda x, y: (-y, x), # Swap then flip new X
        'swap_flip_xy':  lambda x, y: (-y, -x),
    }

    print(f"Generating {len(transformations)} dataset variants in: {output_dir}")

    for name, func in transformations.items():
        print(f"  - Applying transformation: {name}")
        
        # Create a mutable copy of the original data
        variant_data = dict(base_data)
        
        # Apply transformation to both coordinate sets
        new_x, new_y = func(x, y)
        new_x_start, new_y_start = func(x_start, y_start)
        
        # Update the dictionary
        variant_data['xcoords'] = new_x
        variant_data['ycoords'] = new_y
        variant_data['xcoords_start'] = new_x_start
        variant_data['ycoords_start'] = new_y_start

        # Save the new .npz file
        output_filename = f"fly001_64_train_converted_{name}.npz"
        output_path = os.path.join(output_dir, output_filename)
        np.savez(output_path, **variant_data)

    print("Generation complete.")

if __name__ == '__main__':
    generate_variants()