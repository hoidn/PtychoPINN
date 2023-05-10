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
    parser.add_argument('--model_type', type=str, default='pinn', help='model type')
    parser.add_argument('--label', type=str, default='', help='Name of this run')
    parser.add_argument('--positions_provided', type=bool, default=False, help='Whether positions are provided or not')
    parser.add_argument('--data_source', type=str, default='lines', help='Data source')
    parser.add_argument('--set_phi', type=bool, default=True, help='Whether to set phi or not')
    parser.add_argument('--nepochs', type=int, default=60, help='Number of epochs')

    parser.add_argument('--offset', type=int, default=4, help='Offset')
    parser.add_argument('--max_position_jitter', type=int, default=10, help='Maximum position jitter')
    parser.add_argument('--output_prefix', type=str, default='lines2', help='Output prefix')

    parser.add_argument('--gridsize', type=int, default=2, help='Grid size')
    parser.add_argument('--n_filters_scale', type=int, default=2, help='Number of filters scale')
    parser.add_argument('--object_big', type=bool, default=True, help='If true, reconstruct the entire solution region for each set of patterns, instead of just the central N x N region.')
    parser.add_argument('--intensity_scale_trainable', type=bool, default=True, help='Whether intensity scale is trainable or not')
    parser.add_argument('--nll_weight', type=float, default=1., help='NLL loss weight')
    parser.add_argument('--mae_weight', type=float, default=0., help='MAE loss weight')

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
else:
    model_type = params.cfg['model_type']
    label = params.cfg['label']
    offset = params.cfg['offset']

out_prefix = misc.get_path_prefix()
os.makedirs(out_prefix, exist_ok=True)

from ptycho.generate_data import *
from ptycho import model
from ptycho.evaluation import save_metrics

# Create a TensorBoard callback
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                 histogram_freq=1,
                                                 profile_batch='500,520')

if model_type == 'pinn':
    from ptycho.train_pinn import history, reconstructed_obj, stitched_obj
elif model_type == 'supervised':
    from ptycho.train_supervised import history, stitched_obj

#def show_groundtruth():
#    plt.imshow(np.absolute(YY_test_full[0, clipleft: -clipright, clipleft: -clipright]),
#               interpolation='none', cmap='jet')

plt.imsave(out_prefix + 'amp_recon.png', np.absolute(stitched_obj[0][:, :, 0]), cmap='jet')
plt.imsave(out_prefix + 'phi_recon.png', np.angle(stitched_obj[0][:, :, 0]), cmap='jet')

plt.imsave(out_prefix + 'amp_orig.png',
           np.absolute(YY_ground_truth[:, :, 0]),
           cmap='jet')
plt.imsave(out_prefix + 'phi_orig.png',
           np.angle(YY_ground_truth[:, :, 0]),
           cmap='jet')

with open(out_prefix + '/recon.dill', 'wb') as f:
    dill.dump(
        {'stitched_obj_amp': np.absolute(stitched_obj[0][:, :, 0]),
         'stitched_obj_phase': np.angle(stitched_obj[0][:, :, 0]),
         'YY_ground_truth_amp': np.absolute(YY_ground_truth[:, :, 0]),
         'YY_ground_truth_phi': np.angle(YY_ground_truth[:, :, 0])},
        f)

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

d = save_metrics(stitched_obj, YY_ground_truth, label = label)
