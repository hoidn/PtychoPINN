import dill
import matplotlib.pyplot as plt
import numpy as np
from ptycho.misc import get_path_prefix

def save_recons(model_type, stitched_obj):
    from ptycho.generate_data import YY_ground_truth
    try:
        out_prefix = get_path_prefix()
        plt.imsave(out_prefix + 'amp_orig.png',
                   np.absolute(YY_ground_truth[:, :, 0]),
                   cmap='jet')
        plt.imsave(out_prefix + 'phi_orig.png',
                   np.angle(YY_ground_truth[:, :, 0]),
                   cmap='jet')
        plt.imsave(out_prefix + 'amp_recon.png', np.absolute(stitched_obj[0][:, :, 0]), cmap='jet')
        plt.imsave(out_prefix + 'phi_recon.png', np.angle(stitched_obj[0][:, :, 0]), cmap='jet')

        with open(out_prefix + '/recon.dill', 'wb') as f:
            dill.dump(
                {'stitched_obj_amp': np.absolute(stitched_obj[0][:, :, 0]),
                 'stitched_obj_phase': np.angle(stitched_obj[0][:, :, 0]),
                 'YY_ground_truth_amp': np.absolute(YY_ground_truth[:, :, 0]),
                 'YY_ground_truth_phi': np.angle(YY_ground_truth[:, :, 0])},
                f)

        d = save_metrics(stitched_obj, YY_ground_truth, label = label)
    except ImportError as e:
        print('object stitching failed. No images will be saved.')
