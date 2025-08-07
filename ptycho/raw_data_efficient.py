"""Efficient sampling-based neighbor finding for gridsize > 1.

This module provides an optimized implementation that samples points first,
then finds neighbors only for sampled points, avoiding O(N²) operations.
"""

import numpy as np
from scipy.spatial import cKDTree
import logging
from typing import Optional, Tuple, Dict, Any

def generate_grouped_data_efficient(raw_data_instance, N: int, K: int = 4, nsamples: int = 1, 
                                  gridsize: int = 1) -> Dict[str, Any]:
    """
    Efficient implementation of grouped data generation that samples first, then finds neighbors.
    
    This avoids the O(N²) operation of finding all groups first. Instead:
    1. Sample nsamples random points from the dataset
    2. For each sampled point, find its K nearest neighbors
    3. Form groups from these neighbors
    
    Args:
        raw_data_instance: RawData instance containing coordinates and diffraction data
        N: Size of the solution region
        K: Number of nearest neighbors
        nsamples: Number of groups to sample
        gridsize: Grid size (C = gridsize²)
        
    Returns:
        Dict containing grouped data
    """
    if gridsize == 1:
        # Use existing implementation for backward compatibility
        from ptycho.raw_data import get_neighbor_diffraction_and_positions
        return get_neighbor_diffraction_and_positions(raw_data_instance, N, K=K, nsamples=nsamples)
    
    C = gridsize ** 2
    n_points = len(raw_data_instance.xcoords)
    
    logging.info(f"Using efficient sampling for gridsize={gridsize}, requesting {nsamples} groups")
    
    # Validate inputs
    if n_points < C:
        raise ValueError(f"Dataset has only {n_points} points but need at least {C} for gridsize={gridsize}")
    
    if C > K + 1:
        raise ValueError(f"Requested {C} coordinates per group but only {K+1} neighbors available (including self)")
    
    # Build KDTree once
    points = np.column_stack((raw_data_instance.xcoords, raw_data_instance.ycoords))
    tree = cKDTree(points)
    
    # Sample starting points
    n_samples_actual = min(nsamples, n_points)
    if n_samples_actual < nsamples:
        logging.warning(f"Requested {nsamples} groups but only {n_points} points available. Using {n_samples_actual}.")
    
    # Random sampling without replacement
    sampled_indices = np.random.choice(n_points, size=n_samples_actual, replace=False)
    
    # For each sampled point, find neighbors and form a group
    selected_groups = []
    valid_groups = 0
    
    for idx in sampled_indices:
        # Find K nearest neighbors for this point
        distances, nn_indices = tree.query(points[idx], k=K+1)
        
        # Form a group by selecting C points from the neighbors
        if len(nn_indices) >= C:
            # Take the C closest neighbors (including the point itself)
            group = nn_indices[:C]
            selected_groups.append(group)
            valid_groups += 1
    
    if valid_groups == 0:
        raise ValueError("No valid groups could be formed")
    
    selected_groups = np.array(selected_groups)
    logging.info(f"Efficiently sampled {valid_groups} groups without O(N²) computation")
    
    # Now process the selected groups using the existing method
    return raw_data_instance._generate_dataset_from_groups(selected_groups, N, K)


def patch_raw_data_class():
    """Monkey-patch the RawData class to use efficient implementation."""
    import ptycho.raw_data as raw_data_module
    
    # Save the original method
    original_generate_grouped_data = raw_data_module.RawData.generate_grouped_data
    
    def generate_grouped_data_patched(self, N, K=4, nsamples=1, dataset_path: Optional[str] = None):
        """Patched version that uses efficient sampling for gridsize > 1."""
        from ptycho import params
        gridsize = params.get('gridsize', 1)
        
        if gridsize == 1:
            # Use original implementation for gridsize=1
            return original_generate_grouped_data(self, N, K, nsamples, dataset_path)
        else:
            # Use efficient implementation
            return generate_grouped_data_efficient(self, N, K, nsamples, gridsize)
    
    # Apply the patch
    raw_data_module.RawData.generate_grouped_data = generate_grouped_data_patched
    logging.info("Patched RawData.generate_grouped_data with efficient implementation")