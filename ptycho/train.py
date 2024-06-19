import os
from ptycho.model_manager import ModelManager
from ptycho.export import save_recons
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import dill
import argparse
from ptycho import params
from ptycho import misc
import numpy as np
import h5py

def main(cfg, probeGuess = None):
    if probeGuess is None:
        try:
            probeGuess = cfg['probe']
            print('USING CONFIG PROBE')
        except:
            probeGuess = params.get('probe')
            print('USING DEFAULT SIMULATION PROBE')
    matplotlib.rcParams["figure.figsize"] = (10, 10)
    matplotlib.rcParams['font.size'] = 12

    save_model = True
    save_data = False

    model_type = cfg['model_type']
    label = cfg['label']
    offset = cfg['offset']
    params.cfg['output_prefix'] = misc.get_path_prefix()

    out_prefix = params.get('output_prefix')
    os.makedirs(out_prefix, exist_ok=True)

    from ptycho.data_preprocessing import generate_data
    X_train, Y_I_train, Y_phi_train, X_test, Y_I_test, Y_phi_test, YY_ground_truth, ptycho_dataset, YY_test_full, norm_Y_I_test = generate_data(cfg, probeGuess)

    model_instance_d = dict()
    if model_type == 'pinn':
        from ptycho import train_pinn
        train_output = train_pinn.train_eval(ptycho_dataset, cfg, probeGuess = probeGuess)
        pred_amp = train_output['pred_amp']
        history = train_output['history']
        reconstructed_obj = train_output['reconstructed_obj']
        stitched_obj = train_output['stitched_obj']
        autoencoder = train_output['autoencoder']
        model_instance_d = train_output

    elif model_type == 'supervised':
        from ptycho.train_supervised import stitched_obj
        from ptycho import train_supervised
        history = train_supervised.history
        reconstructed_obj = train_supervised.reconstructed_obj
    else:
        raise ValueError

    d = save_recons(model_type, stitched_obj)

    with open(out_prefix + '/history.dill', 'wb') as file_pi:
        dill.dump(history.history, file_pi)

    if save_model:
        from ptycho.model import ProbeIllumination, negloglik
        from ptycho.tf_helper import Translation
        from ptycho.tf_helper import realspace_loss as hh_realspace_loss
        hh = {'realspace_loss': hh_realspace_loss}

        model_path = '{}/{}'.format(out_prefix, params.get('h5_path'))
        custom_objects = {
            'ProbeIllumination': ProbeIllumination,
            'Translation': Translation,
            'negloglik': negloglik,
            'realspace_loss': hh_realspace_loss
        }
        try:
            ModelManager.save_model(autoencoder, model_path, custom_objects, params.get('intensity_scale'))
        except Exception as e:
            print("model saving failed")
        with h5py.File(model_path, 'a') as f:
            f.attrs['intensity_scale'] = params.get('intensity_scale')

    if save_data:
        with open(out_prefix + '/test_data.dill', 'wb') as f:
            dill.dump(
                {'YY_test_full': YY_test_full,
                 'Y_I_test': Y_I_test,
                 'Y_phi_test': Y_phi_test,
                 'X_test': X_test}, f)

    return ptycho_dataset, YY_ground_truth, YY_test_full, Y_I_test, Y_phi_test, X_test, norm_Y_I_test, history, reconstructed_obj, stitched_obj, d, model_instance_d, cfg

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

    # Create the cfg dictionary from the parsed arguments
    cfg = {
        'model_type': args.model_type,
        'label': args.label,
        'positions.provided': args.positions_provided,
        'data_source': args.data_source,
        'set_phi': args.set_phi,
        'nepochs': args.nepochs,
        'offset': args.offset,
        'max_position_jitter': args.max_position_jitter,
        'output_prefix': args.output_prefix,
        'gridsize': args.gridsize,
        'n_filters_scale': args.n_filters_scale,
        'object.big': args.object_big,
        'intensity_scale.trainable': args.intensity_scale_trainable,
        'nll_weight': args.nll_weight,
        'mae_weight': args.mae_weight,
        'nimgs_train': args.nimgs_train,
        'nimgs_test': args.nimgs_test,
        'outer_offset_train': args.outer_offset_train,
        'outer_offset_test': args.outer_offset_test
    }

    ptycho_dataset, YY_ground_truth, YY_test_full, Y_I_test, Y_phi_test, X_test, norm_Y_I_test, history, reconstructed_obj, stitched_obj, d, cfg = main(cfg)
