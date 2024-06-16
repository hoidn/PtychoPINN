import argparse
from collections import defaultdict
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
from ptycho.params import cfg
from ptycho import misc

def get_variable_keys(callbacks):
    """
    Identifies keys in configuration dictionaries returned by callbacks that have multiple distinct values.

    Parameters:
        callbacks (list of functions): List of callback functions that return configuration dictionaries.

    Returns:
        set: Set of keys that have variable values across configurations.
    """
    value_tracker = defaultdict(set)
    for callback in callbacks:
        config = callback()
        for key, value in config.items():
            value_tracker[key].add(str(value)[:10])
    return {key for key, values in value_tracker.items() if len(values) > 1}

def generate_config_key(config, variable_keys):
    """
    Generates a tuple of key-value pairs for variable keys.

    Parameters:
        config (dict): Configuration dictionary.
        variable_keys (set): Set of keys that are variable.

    Returns:
        tuple: Tuple of (key, value) for variable keys.
    """
    return tuple((key, config[key]) for key in sorted(variable_keys) if key in config)

def plot_results(stitched_obj, YY_ground_truth, d):
    """
    Plots the results of the experiment.
    """
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    img1 = axs.imshow(np.absolute(stitched_obj)[0], cmap='jet', interpolation='none')
    axs.set_title(f'Reconstructed amplitude - FRC50: {d["frc50"][0]:.2f}')
    fig.colorbar(img1, ax=axs)

def execute(train, config, reload_modules=False):
    """
    Executes the training and plotting functions, optionally reloading modules.
    """
    for key, value in config.items():
        cfg[key] = value

    if reload_modules:
        reload(train.generate_data)
        reload(train.model)
        reload(train.train_pinn)
        reload(train)

    stitched_obj, YY_ground_truth = train.stitched_obj, train.YY_ground_truth
    plot_results(stitched_obj, YY_ground_truth, train.d)
    return train.d, YY_ground_truth, stitched_obj, train.train_output

def run_experiment_with_photons(callbacks):
    """
    Runs experiments based on a list of callbacks that generate configurations.
    """
    print("DEBUG: Starting run_experiment_with_photons")
    results = {}
    first_iteration = True
    variable_keys = get_variable_keys(callbacks)

    for get_config in callbacks:
        config = get_config()
        print("DEBUG: Config set for experiment:", config)

        if first_iteration:
            from ptycho import train
            d, YY_ground_truth, stitched_obj, train_output = execute(train, config, reload_modules=False)
            first_iteration = False
        else:
            d, YY_ground_truth, stitched_obj, train_output = execute(train, config, reload_modules=True)

        config_key = generate_config_key(config, variable_keys)
        results[config_key] = {'d': d, 'YY_ground_truth': YY_ground_truth, 'stitched_obj': stitched_obj, 'train_output': train_output}

    return results

#if __name__ == '__main__':
#    nphotons = parse_arguments()
#    # Assume callbacks are predefined or fetched here
