#!/usr/bin/env python3
# ptycho_simulate_cli.py

import argparse
import os
import matplotlib.pyplot as plt
from ptycho.workflows.components import (
    setup_configuration,
    run_cdi_example,
    update_params,

)

def save_plot_to_file(fig, filename):
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_html_report(output_dir, image_files):
    html_content = "<html><body>"
    for image_file in image_files:
        html_content += f'<img src="{image_file}" style="max-width:100%"><br><br>'
    html_content += "</body></html>"
    
    with open(os.path.join(output_dir, 'report.html'), 'w') as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description="Simulate ptychography data and generate visualizations.")
    parser.add_argument("input_file", help="Path to the input .npz file containing probe and object guesses.")
    parser.add_argument("output_dir", help="Directory to save output visualizations.")
    parser.add_argument("--nimages", type=int, default=2000, help="Number of images to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--nepochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--output_prefix", default="tmp", help="Prefix for output files.")
    parser.add_argument("--intensity_scale_trainable", action="store_true", help="Make intensity scale trainable.")
    parser.add_argument("--positions_provided", action="store_true", help="Positions are provided.")
    parser.add_argument("--probe_big", action="store_true", help="Use big probe.")
    parser.add_argument("--probe_mask", action="store_true", help="Use probe mask.")
    parser.add_argument("--data_source", default="generic", help="Data source type.")
    parser.add_argument("--gridsize", type=int, default=1, help="Grid size.")
    parser.add_argument("--train_data_file_path", help="Path to train data file.")
    parser.add_argument("--test_data_file_path", help="Path to test data file.")
    parser.add_argument("--N", type=int, default=128, help="Size of the simulation grid.")
    parser.add_argument("--probe_scale", type=int, default=4, help="Probe scale factor.")
    parser.add_argument("--nphotons", type=float, default=1e9, help="Number of photons.")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    params = {
        "nepochs": args.nepochs,
        "output_prefix": args.output_prefix,
        "intensity_scale.trainable": args.intensity_scale_trainable,
        "positions.provided": args.positions_provided,
        "probe.big": args.probe_big,
        "probe.mask": args.probe_mask,
        "data_source": args.data_source,
        "gridsize": args.gridsize,
        "train_data_file_path": args.train_data_file_path,
        "test_data_file_path": args.test_data_file_path,
        "N": args.N,
        "probe_scale": args.probe_scale,
        "nphotons": args.nphotons,
    }
    

    update_params(params)
    config = setup_configuration(args, args.config)

    from ptycho import probe
    from ptycho.nongrid_simulation import (
        simulate_from_npz,
        visualize_simulated_data,
        plot_random_groups,
        compare_reconstructions,
    )
    from ptycho import tf_helper as hh
    from ptycho import baselines as bl
    from ptycho.workflows.components import create_ptycho_data_container

    # Simulate data
    simulated_data, ground_truth_patches = simulate_from_npz(
        args.input_file, args.nimages, random_seed=args.seed
    )

    # Set the probe
    probe.set_probe_guess(None, simulated_data.probeGuess)

    # Prepare data for visualization
    data_for_vis = {
        'diffraction_patterns': simulated_data.diff3d,
        'ground_truth_patches': ground_truth_patches,
        'probe_guess': simulated_data.probeGuess,
        'object': simulated_data.objectGuess,
        'x_coordinates': simulated_data.xcoords,
        'y_coordinates': simulated_data.ycoords,
    }

    # Generate and save visualizations
    image_files = []

    # Visualize simulated data
    plt.figure(figsize=(20, 20))
    visualize_simulated_data(data_for_vis)
    filename = os.path.join(args.output_dir, 'simulated_data_summary.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    image_files.append(filename)
    plt.close()

    # Plot random groups
    for i in range(3):  # Generate 3 sets of random groups
        plt.figure(figsize=(15, 15))
        plot_random_groups(simulated_data, K=5, seed=args.seed)
        filename = os.path.join(args.output_dir, f'random_groups_{i+1}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        image_files.append(filename)
        plt.close()

    # Run CDI example and compare reconstructions
    config = setup_configuration(args, None)
    train_data = create_ptycho_data_container(simulated_data, config)
    recon_amp, recon_phase, results = run_cdi_example(train_data, train_data, config)

    # Train baseline model
    baseline_model = bl.train(train_data.X[:, :, :, :1], train_data.Y_I[:, :, :, :1], train_data.Y_phi[:, :, :, :1])
    baseline_pred_I, baseline_pred_phi = baseline_model[0].predict([train_data.X[:, :, :, 0]])

    # Compare reconstructions
    plt.figure(figsize=(20, 20))
    compare_reconstructions(
        results['obj_tensor_full'],
        results['global_offsets'],
        simulated_data.objectGuess,
        hh.combine_complex(baseline_pred_I, baseline_pred_phi)
    )
    filename = os.path.join(args.output_dir, 'reconstruction_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    image_files.append(filename)
    plt.close()

    # Generate HTML report
    generate_html_report(args.output_dir, image_files)

    print(f"Simulation and visualization complete. Results saved in {args.output_dir}")
    print(f"Open {os.path.join(args.output_dir, 'report.html')} to view the visualizations.")

if __name__ == "__main__":
    main()
