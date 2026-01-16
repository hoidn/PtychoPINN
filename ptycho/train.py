"""Legacy training script for PtychoPINN models.

.. deprecated:: 
   This module uses the legacy ``ptycho.params.cfg`` global dictionary pattern
   and direct ``argparse`` parsing. It has been superseded by the modern 
   ``ptycho_train`` command-line tool which uses dataclass-based configuration
   and improved workflow management. **This module should not be used for new 
   development.**

.. warning::
   **DO NOT USE THIS SCRIPT.** Use ``ptycho_train`` instead, which provides:
   
   - Modern dataclass-based configuration system
   - Better error handling and logging
   - Standardized output directory structure
   - YAML configuration file support
   - Integration with current project workflows

This module demonstrates the old architectural pattern where configuration was
managed through a global ``ptycho.params.cfg`` dictionary and command-line 
arguments were parsed directly in the training script. The modern approach
separates concerns through dedicated configuration classes and workflow
orchestration functions.

**Migration Examples:**

Legacy command::

    python -m ptycho.train --model_type pinn --nepochs 60 --gridsize 2

Modern equivalent::

    ptycho_train --train_data_file datasets/my_data.npz \\
                 --model_type pinn \\
                 --nepochs 60 \\
                 --gridsize 2 \\
                 --output_dir my_training_run

Or using YAML configuration::

    ptycho_train --config configs/my_experiment.yaml --output_dir my_run

For historical context, this script orchestrated model training for both PINN
and supervised approaches by: generating/loading synthetic data, configuring
the model through global state, running training loops, and saving results
including weights, training history, and reconstruction visualizations.
"""
# train.py

import os
from ptycho.model_manager import ModelManager
from ptycho import model_manager
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
params.cfg['output_prefix'] = misc.get_path_prefix()

out_prefix = params.get('output_prefix')
os.makedirs(out_prefix, exist_ok=True)

from ptycho import generate_data
ptycho_dataset = generate_data.ptycho_dataset
YY_ground_truth = generate_data.YY_ground_truth
YY_test_full = generate_data.YY_test_full
Y_I_test = generate_data.Y_I_test
Y_phi_test = generate_data.Y_phi_test
X_test = generate_data.X_test
norm_Y_I_test = generate_data.norm_Y_I_test
from ptycho import model
from ptycho.evaluation import save_metrics

if model_type == 'pinn':
    from ptycho import train_pinn
    print("DEBUG: generate_data diff norm {}".format(np.mean(np.abs(ptycho_dataset.train_data.X))))
    train_output = train_pinn.train_eval(ptycho_dataset)
    pred_amp = train_output['pred_amp']
    history = train_output['history']
    reconstructed_obj = train_output['reconstructed_obj']
    stitched_obj = train_output['stitched_obj']

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
    model_manager.save(out_prefix)

if save_data:
    with open(out_prefix + '/test_data.dill', 'wb') as f:
        dill.dump(
            {'YY_test_full': YY_test_full,
             'Y_I_test': Y_I_test,
             'Y_phi_test': Y_phi_test,
             'X_test': X_test}, f)
