import argparse

def init(nphotons):
    from ptycho.params import cfg
    from ptycho.params import cfg
    cfg['positions.provided'] = False
    cfg['data_source'] = 'lines'
    cfg['set_phi'] = False
    cfg['nepochs'] = 60

    cfg['offset'] = 4
    cfg['max_position_jitter'] = 3
    cfg['output_prefix'] = 'lines3'

    cfg['gridsize'] = 2
    cfg['n_filters_scale'] = 2
    cfg['object.big'] = True
    cfg['intensity_scale.trainable'] = True
    cfg['probe.trainable'] = False

    cfg['outer_offset_train'] = 8
    cfg['outer_offset_test'] = 20
    cfg['nimgs_train'] = 2
    cfg['nimgs_test'] = 2

    cfg['nphotons'] = nphotons

def plot_results(stitched_obj, YY_ground_truth, d):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    # reconstructed amplitude images
    img1 = axs.imshow(np.absolute(stitched_obj)[0], cmap='jet', interpolation='none')
    axs.set_title(f'Reconstructed amplitude - FRC50: {d["frc50"][0]:.2f}')

    fig.colorbar(img1, ax=axs)

def execute(nphotons, reload_modules=False):
    from ptycho.evaluation import save_metrics, trim
    from ptycho.tf_helper import pad
    from ptycho.evaluation import save_metrics, trim
    from ptycho.tf_helper import pad
    from ptycho.params import cfg
    cfg['nphotons'] = nphotons

    cfg['data_source'] = 'lines'
    cfg['offset'] = 4
    cfg['max_position_jitter'] = 10
    cfg['output_prefix'] = 'lines2'

    cfg['gridsize'] = 2
    cfg['n_filters_scale'] = 2
    cfg['object.big'] = True
    cfg['intensity_scale.trainable'] = True

    from ptycho import train
    if reload_modules:
        reload(train.generate_data)
        reload(train.train_pinn.model)
        reload(train.train_pinn)
        reload(train)

    stitched_obj, YY_ground_truth = train.stitched_obj, train.YY_ground_truth

    from ptycho.train_pinn import train as train_pinn, eval as eval_pinn

    d = save_metrics(stitched_obj, YY_ground_truth, label='PINN,NLL,overlaps')
    #d0 = d

    plot_results(stitched_obj, YY_ground_truth, d)
    # Corrected the indentation and scope of the return statement
    return d, YY_ground_truth, stitched_obj

def parse_arguments():
    parser = argparse.ArgumentParser(description='Ptychographic reconstruction script.')
    parser.add_argument('nphotons', type=float, help='Number of photons')
    args = parser.parse_args()
    return args.nphotons

if __name__ == '__main__':
    nphotons = parse_arguments()
    init(nphotons)

    d, YY_ground_truth, stitched_obj = execute(nphotons)

from importlib import reload
def run_experiment_with_photons(photons_list):
    print("DEBUG: Starting run_experiment_with_photons")
    results = {}
    first_iteration = True
    for nphotons in photons_list:
        init(nphotons)
        print("DEBUG: nphotons set to", nphotons, "in run_experiment_with_photons")
        if  first_iteration:
            d, YY_ground_truth, stitched_obj = execute(nphotons, reload_modules=False)
        else:
            d, YY_ground_truth, stitched_obj = execute(nphotons, reload_modules=True)
        first_iteration = False
        results[nphotons] = {'d': d, 'YY_ground_truth': YY_ground_truth, 'stitched_obj': stitched_obj}
    return results
import os
import dill
import pandas as pd

def load_recent_experiment_data(directory, N):
    subdirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    recent_subdirs = subdirs[:N]

    data = {}
    for subdir in recent_subdirs:
        params_path = os.path.join(subdir, 'params.dill')
        metrics_path = os.path.join(subdir, 'metrics.csv')

        with open(params_path, 'rb') as f:
            params = dill.load(f)
        metrics = pd.read_csv(metrics_path)

        nphotons = params['nphotons']
        if nphotons not in data or os.path.getmtime(params_path) > os.path.getmtime(os.path.join(data[nphotons]['dir'], 'params.dill')):
            data[nphotons] = {'params': params, 'metrics': metrics, 'dir': subdir}

    return {k: {'params': v['params'], 'metrics': v['metrics']} for k, v in data.items()}
def is_valid_run(subdir):
    return os.path.exists(os.path.join(subdir, 'params.dill'))
