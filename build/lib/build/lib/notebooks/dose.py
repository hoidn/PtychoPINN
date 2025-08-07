import argparse

def init(nphotons, loss_fn='nll'):
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

    if loss_fn == 'mae':
        cfg['mae_weight'] = 1.
        cfg['nll_weight'] = 0.
    elif loss_fn == 'nll':
        pass  # Keep the current behavior
    else:
        raise ValueError(f"Invalid loss_fn: {loss_fn}. Must be 'mae' or 'nll'.")

def plot_results(stitched_obj, YY_ground_truth, d):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    # reconstructed amplitude images
    img1 = axs.imshow(np.absolute(stitched_obj)[0], cmap='jet', interpolation='none')
    axs.set_title(f'Reconstructed amplitude - FRC50: {d["frc50"][0]:.2f}')

    fig.colorbar(img1, ax=axs)

def execute(nphotons, reload_modules=False):
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
    from ptycho import misc

    plot_results(stitched_obj, YY_ground_truth, train.d)
    # Corrected the indentation and scope of the return statement
    return train.d, YY_ground_truth, stitched_obj, train.train_output

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
def run_experiment_with_photons(photons_list, loss_fn='nll'):
    print("DEBUG: Starting run_experiment_with_photons")
    results = {}
    first_iteration = True
    for nphotons in photons_list:
        init(nphotons, loss_fn=loss_fn)
        print("DEBUG: nphotons set to", nphotons, "in run_experiment_with_photons")
        if  first_iteration:
            d, YY_ground_truth, stitched_obj, train_output = execute(nphotons, reload_modules=False)
        else:
            d, YY_ground_truth, stitched_obj, train_output = execute(nphotons, reload_modules=True)
        first_iteration = False
        results[nphotons] = {'d': d, 'YY_ground_truth': YY_ground_truth, 'stitched_obj': stitched_obj, 'train_output': train_output}
    return results
import os
import dill
import pandas as pd
import numpy as np
from matplotlib.image import imread

def has_amp_recon(subdir):
    return os.path.exists(os.path.join(subdir, 'amp_recon.png'))

def load_recent_experiment_data(directory, N):
    subdirs = [os.path.join(directory, d) for d in os.listdir(directory) if is_valid_run(os.path.join(directory, d)) and has_amp_recon(os.path.join(directory, d))]
    print(subdirs)
    recent_subdirs = subdirs[:N]
    subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    data = {}
    for subdir in recent_subdirs:
        params_path = os.path.join(subdir, 'params.dill')
        metrics_path = os.path.join(subdir, 'metrics.csv')

        with open(params_path, 'rb') as f:
            params = dill.load(f)
        metrics = pd.read_csv(metrics_path)

        nphotons = (np.log10(params['nphotons']))
        print('NPOHOT {}'.format(nphotons))
        #if nphotons not in data or os.path.getmtime(params_path) > os.path.getmtime(os.path.join(data[nphotons]['dir'], 'params.dill')):
        amp_recon_path = os.path.join(subdir, 'amp_recon.png')
        amp_recon = imread(amp_recon_path)
        data[nphotons] = {'params': params, 'metrics': metrics, 'amp_recon': amp_recon, 'dir': subdir}

    return {k: {'params': v['params'], 'metrics': v['metrics']} for k, v in data.items()}
def is_valid_run(subdir):
    return os.path.exists(os.path.join(subdir, 'params.dill'))
import matplotlib.pyplot as plt

def generate_and_save_heatmap(experiment_entry, ax=None, photon_dose=None):
    if ax is None:
        fig, ax = plt.subplots()
    stitched_obj = experiment_entry['stitched_obj'][0, :, :, 0]
    metrics = experiment_entry['d']
    frc50 = metrics.get('frc50', [None])[0]
    psnr = metrics.get('psnr', [None])[0]

    ax.imshow(np.abs(stitched_obj), cmap='jet', interpolation='nearest')
    title = f'FRC50: {frc50:.2f}, PSNR: {psnr:.2f}'
    if photon_dose is not None:
        title = f'Photons: {photon_dose:.0e}, ' + title
    ax.set_title(title)
    ax.axis('off')

def generate_2x2_heatmap_plots(res, layout=(1, 4), filename='heatmap_plots.png', axs=None,
                               fig = None):
#    fig, axs = plt.subplots(layout[0], layout[1], figsize=(12, 4*layout[0]))
#    axs = axs.flatten()
    for i, (photon_dose, experiment_entry) in enumerate(res.items()):
        generate_and_save_heatmap(experiment_entry, axs[i], photon_dose)
    plt.tight_layout()
    plt.savefig(filename)
    if axs is None:
        plt.tight_layout()
        plt.savefig(filename)
        #plt.close(fig)

def plot_heatmap_from_experiment(res, nphot, index):
    import matplotlib.pyplot as plt
    c = res[nphot]['train_output']['dataset']
    plt.imshow(np.log10(c.X[index][:, :, 0]), cmap='viridis', interpolation='nearest')
    #plt.imshow(np.log10(.5 + c.X[index][:, :, 0]), cmap='viridis', interpolation='nearest')
    plt.title(f'{nphot:.0e} photons', fontsize = 10)
    plt.savefig(f'heatmap_photon_dose_{nphot:.0e}_index_{index}.png')
    #plt.show()
def plot_heatmaps_for_all_photons(res, index):
    for nphot in res.keys():
        plot_heatmap_from_experiment(res, nphot, index)
    fig, axs = plt.subplots(layout[0], layout[1], figsize=(12, 3*layout[0]))

def generate_2x2_heatmap_plots_using_function(res, index, layout=(1, 4), filename='heatmap_plots_2x2.png', border_color='black', border_width=2, axs=None):
    a, b = layout
    #fig, axs = plt.subplots(1, b, figsize=(24, 3))
    #fig, axs = plt.subplots(layout[0], layout[1], figsize=(12, 3*layout[0])) if axs is None else (None, axs)
    axs = axs.flatten()
    photon_doses = list(res.keys())[: b]  # Select the first 4 photon doses for the 2x2 grid
    for i, nphot in enumerate(photon_doses):
        ax = axs[i]
        c = res[nphot]['train_output']['dataset']
        heatmap = ax.imshow(np.log10(c.X[index][:, :, 0]), cmap='viridis', interpolation='nearest')
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_width)
        #ax.imshow(np.log10(.5 + c.X[index][:, :, 0]), cmap='viridis', interpolation='nearest')
        #ax.set_title(f'{nphot:.0e} photons', fontsize=16)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    if axs is None:
        plt.tight_layout()
        plt.savefig(filename)
        #plt.show()

def stack_and_display_horizontal_plots(res, index, layout=(1, 4), figsize=(24, 8), crop_size=None):
    from matplotlib import pyplot as plt
    import numpy as np

    a, b = layout
    fig, axs = plt.subplots(2, b, figsize=figsize)

    if crop_size is not None:
        def crop_center(img, cropx, cropy):
            y, x = img.shape
            startx = x // 2 - (cropx // 2)
            starty = y // 2 - (cropy // 2)
            return img[starty:starty + cropy, startx:startx + cropx]

        cropped_res = {}
        for dose, entry in res.items():
            stitched_obj = entry['stitched_obj'][0, :, :, 0]
            cropped_obj = crop_center(stitched_obj, crop_size, crop_size)
            padded_obj = np.pad(cropped_obj, ((0, crop_size - cropped_obj.shape[0]), (0, crop_size - cropped_obj.shape[1])), mode='constant')
            cropped_res[dose] = {'stitched_obj': np.expand_dims(np.expand_dims(padded_obj, axis=0), axis=-1), **{k: v for k, v in entry.items() if k != 'stitched_obj'}}

        generate_2x2_heatmap_plots(cropped_res, layout=layout, axs=axs[0])
    else:
        generate_2x2_heatmap_plots(res, layout=layout, axs=axs[0])

    generate_2x2_heatmap_plots_using_function(res, index, layout=layout, axs=axs[1], border_color='black', border_width=2)
    plt.tight_layout()
    fig.savefig(f'stacked_dose_progression_index_{index}.png')
    plt.show()
