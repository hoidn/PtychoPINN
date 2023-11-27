import os
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import dill
import argparse
from ptycho import params
from ptycho import misc

plt.rcParams["figure.figsize"] = (10, 10)
matplotlib.rcParams['font.size'] = 12

save_model = True
save_data = False

parser = argparse.ArgumentParser(description='Script to set attributes for ptycho program')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog = 'PtychoPINN',
                        description = 'Generate / load data and train the model',
                        )
    parser.add_argument('--model_type', type=str, default='pinn', help='model type (pinn or supervised)')
    parser.add_argument('--output_prefix', type=str, default='lines2', help='output directory prefix')
    parser.add_argument('--data_source', type=str, default='lines', help='Dataset specification')
    parser.add_argument('--set_phi', action='store_true', default=False, help='If true, simulated objects are given non-zero phase')
    parser.add_argument('--nepochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--offset', type=int, default=4, help='Scan point spacing for simulated (grid-sampled) data')
    parser.add_argument('--gridsize', type=int, default=2, help='Solution region grid size (e.g. 2 -> 2x2, etc.)')
    parser.add_argument('--object_big', type=bool, default=True, help='If true, reconstruct the entire solution region for each set of patterns, instead of just the central N x N region.')
    parser.add_argument('--nll_weight', type=float, default=1., help='Diffraction reconstruction NLL loss weight')
    parser.add_argument('--mae_weight', type=float, default=0., help='Diffraction reconstruction MAE loss weight')

    parser.add_argument('--nimgs_train', type=int, default=params.cfg['nimgs_train'], help='Number of generated training images')
    parser.add_argument('--nimgs_test', type=int, default=params.cfg['nimgs_test'], help='Number of generated testing images')

    parser.add_argument('--outer_offset_train', type=int, default=None, help='Scan point grid offset for (generated) training datasets')
    parser.add_argument('--outer_offset_test', type=int, default=None, help='Scan point grid offset for (generated) testing datasets')

    parser.add_argument('--n_filters_scale', type=int, default=2, help='Number of filters scale')
    parser.add_argument('--max_position_jitter', type=int, default=10, help='Solution region is expanded around the edges by this amount')
    parser.add_argument('--intensity_scale_trainable', type=bool, default=True, help='If true, sets the model-internal normalization of diffraction amplitudes to trainable')

    parser.add_argument('--positions_provided', type=bool, default=False, help='[deprecated] Whether nominal or true (nominal + jitter) positions are provided in simulation runs')
    parser.add_argument('--label', type=str, default='', help='[deprecated] Name of this run')
    args = parser.parse_args()

    # offset between neighboring scan points, in pixels
    model_type = params.cfg['model_type'] = args.model_type
    label = params.cfg['label'] = args.label
    params.cfg['positions.provided'] = args.positions_provided
    params.cfg['data_source'] = args.data_source
    params.cfg['set_phi'] = args.set_phi
    params.cfg['nepochs'] = args.nepochs
    offset = params.cfg['offset'] = args.offset
    params.cfg['max_position_jitter'] = args.max_position_jitter
    params.cfg['output_prefix'] = args.output_prefix
    params.cfg['gridsize'] = args.gridsize
    params.cfg['n_filters_scale'] = args.n_filters_scale
    params.cfg['object.big'] = args.object_big
    params.cfg['intensity_scale.trainable'] = args.intensity_scale_trainable
    params.cfg['nll_weight'] = args.nll_weight
    params.cfg['mae_weight'] = args.mae_weight
    params.cfg['nimgs_train'] = args.nimgs_train
    params.cfg['nimgs_test'] = args.nimgs_test

    params.cfg['outer_offset_train'] = args.outer_offset_train
    params.cfg['outer_offset_test'] = args.outer_offset_test
else:
    model_type = params.cfg['model_type']
    label = params.cfg['label']
    offset = params.cfg['offset']

out_prefix = misc.get_path_prefix()
os.makedirs(out_prefix, exist_ok=True)

from ptycho.generate_data import *
from ptycho import model
from ptycho.evaluation import save_metrics

#if model_type == 'pinn':
#    from ptycho.train_pinn import history, reconstructed_obj, pred_amp
#elif model_type == 'supervised':
#    from ptycho.train_supervised import history, reconstructed_obj
if model_type == 'pinn':
    from ptycho import train_pinn
    history = train_pinn.history
    reconstructed_obj = train_pinn.reconstructed_obj
    pred_amp = train_pinn.pred_amp
elif model_type == 'supervised':
    from ptycho import train_supervised
    history = train_supervised.history
    reconstructed_obj = train_supervised.reconstructed_obj
else:
    raise ValueError

try:
    if model_type == 'pinn':
        from ptycho.train_pinn import stitched_obj
    elif model_type == 'supervised':
        from ptycho.train_supervised import stitched_obj
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


with open(out_prefix + '/history.dill', 'wb') as file_pi:
    dill.dump(history.history, file_pi)

if save_model:
    model.autoencoder.save('{}.h5'.format(out_prefix + 'wts'), save_format="tf")

if save_data:
    with open(out_prefix + '/test_data.dill', 'wb') as f:
        dill.dump(
            {'YY_test_full': YY_test_full,
             'Y_I_test': Y_I_test,
             'Y_phi_test': Y_phi_test,
             'X_test': X_test}, f)

