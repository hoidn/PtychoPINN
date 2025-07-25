"""Export utilities for saving ptychographic reconstruction results and visualizations.

Provides functions to save reconstructed objects as images and pickled data files.
"""
import dill
import matplotlib.pyplot as plt
import numpy as np
from ptycho.misc import get_path_prefix
from ptycho.params import get

def save_recons(model_type, stitched_obj, ground_truth_obj=None):
    """Save reconstruction results and ground truth images.
    
    Args:
        model_type: Type of model ('supervised', 'pinn', etc.)
        stitched_obj: Reconstructed object array
        ground_truth_obj: Ground truth object array (optional)
    """
    try:
        out_prefix = get('output_prefix')
        if ground_truth_obj is not None:
            plt.imsave(out_prefix + 'amp_orig.png',
                       np.absolute(ground_truth_obj[:, :, 0]),
                       cmap='jet')
            plt.imsave(out_prefix + 'phi_orig.png',
                       np.angle(ground_truth_obj[:, :, 0]),
                       cmap='jet')
        if stitched_obj is not None:
            plt.imsave(out_prefix + 'amp_recon.png', np.absolute(stitched_obj[0][:, :, 0]), cmap='jet')
            plt.imsave(out_prefix + 'phi_recon.png', np.angle(stitched_obj[0][:, :, 0]), cmap='jet')

        with open(out_prefix + '/recon.dill', 'wb') as f:
            dump_data = {'stitched_obj_amp': np.absolute(stitched_obj[0][:, :, 0] if stitched_obj is not None else np.array([])),
                         'stitched_obj_phase': np.angle(stitched_obj[0][:, :, 0]) if stitched_obj is not None else np.array([])}
            if ground_truth_obj is not None:
                dump_data.update({'YY_ground_truth_amp': np.absolute(ground_truth_obj[:, :, 0]),
                                  'YY_ground_truth_phi': np.angle(ground_truth_obj[:, :, 0])})
            dill.dump(dump_data, f)
        
        # No longer calculate metrics here - that's done separately by the caller
        
    except ImportError as e:
        print('object stitching failed. No images will be saved.')
