#!/usr/bin/env python3
# ptycho_simulate_cli.py

import argparse
import os
import sys
import matplotlib.pyplot as plt
from ptycho.workflows.components import (
    setup_configuration,
    run_cdi_example,
)

def save_plot_to_file(fig, filename):
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_html_report(output_dir, image_files, args, params):
    import base64

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Ptychography Simulation Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
            }
            h1, h2 {
                color: #2c3e50;
                text-align: center;
            }
            .image-container {
                margin-bottom: 30px;
            }
            img {
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
            }
            .image-title {
                font-weight: bold;
                margin-top: 10px;
                text-align: center;
            }
            .command, .parameters {
                background-color: #f4f4f4;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
                margin-bottom: 20px;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .parameter-name {
                font-weight: bold;
            }
            .parameter-description {
                margin-left: 20px;
                margin-bottom: 10px;
            }
        </style>
    </head>
    <body>
        <h1>Ptychography Simulation Report</h1>
        
        <h2>Launch Command</h2>
        <div class="command">
        {' '.join(sys.argv)}
        </div>
        
        <h2>Model Parameters</h2>
        <div class="parameters">
    """

    for key, value in params.items():
        html_content += f'<p><span class="parameter-name">{key}:</span> {value}</p>\n'
        if key == "N":
            html_content += '<p class="parameter-description">Size of the simulation grid.</p>\n'
        elif key == "probe_scale":
            html_content += '<p class="parameter-description">Probe scale factor.</p>\n'
        elif key == "nphotons":
            html_content += '<p class="parameter-description">Number of photons.</p>\n'
        elif key == "mae_weight":
            html_content += '<p class="parameter-description">Weight for MAE loss.</p>\n'
        elif key == "nll_weight":
            html_content += '<p class="parameter-description">Weight for NLL loss.</p>\n'
        elif key == "nepochs":
            html_content += '<p class="parameter-description">Number of epochs for training.</p>\n'
        elif key == "intensity_scale.trainable":
            html_content += '<p class="parameter-description">Whether intensity scale is trainable.</p>\n'
        elif key == "positions.provided":
            html_content += '<p class="parameter-description">Whether positions are provided.</p>\n'
        elif key == "probe.big":
            html_content += '<p class="parameter-description">Whether to use a big probe.</p>\n'
        elif key == "probe.mask":
            html_content += '<p class="parameter-description">Whether to use a probe mask.</p>\n'
        elif key == "data_source":
            html_content += '<p class="parameter-description">Type of data source.</p>\n'
        elif key == "gridsize":
            html_content += '<p class="parameter-description">Grid size for simulation.</p>\n'

    html_content += """
        </div>
        
        <h2>Visualizations</h2>
    """

    for image_file in image_files:
        image_name = os.path.basename(image_file)
        image_title = image_name.replace('_', ' ').replace('.png', '').title()
        
        # Read the image file and encode it in base64
        with open(image_file, 'rb') as img_f:
            image_data = img_f.read()
            encoded_image = base64.b64encode(image_data).decode('utf-8')
        
        # Determine the image's MIME type
        mime_type = 'image/png'  # Adjust if using other image formats
        
        # Embed the image in the HTML using a data URI
        html_content += f"""
        <div class="image-container">
            <img src="data:{mime_type};base64,{encoded_image}" alt="{image_title}">
            <p class="image-title">{image_title}</p>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

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
    parser.add_argument("--intensity_scale_trainable", action="store_true", default=False, help="Make intensity scale trainable.")
    parser.add_argument("--positions_provided", action="store_true", default=True, help="Positions are provided.")
    parser.add_argument("--probe_big", action="store_true", default=True, help="Use big probe.")
    parser.add_argument("--probe_mask", action="store_true", default=False, help="Use probe mask.")
    parser.add_argument("--data_source", default="generic", help="Data source type.")
    parser.add_argument("--gridsize", type=int, default=1, help="Grid size.")
    parser.add_argument("--train_data_file_path", default=None, help="Path to train data file.")
    parser.add_argument("--test_data_file_path", default=None, help="Path to test data file.")
    parser.add_argument("--N", type=int, default=128, help="Size of the simulation grid.")
    parser.add_argument("--probe_scale", type=int, default=4, help="Probe scale factor.")
    parser.add_argument("--nphotons", type=float, default=1e9, help="Number of photons.")
    parser.add_argument("--mae_weight", type=float, default=1, help="Weight for MAE loss.")
    parser.add_argument("--nll_weight", type=float, default=0, help="Weight for NLL loss.")
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
        "mae_weight": args.mae_weight,
        "nll_weight": args.nll_weight,
    }
    
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
    #plt.figure(figsize=(20, 20))
    visualize_simulated_data(data_for_vis, args.output_dir)
    filename = os.path.join(args.output_dir, "simulated_data_visualization.png")
    #plt.savefig(filename, dpi=300, bbox_inches='tight')
    image_files.append(filename)
    #plt.close()

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

    # Generate HTML report with embedded images, launch command, and model parameters
    generate_html_report(args.output_dir, image_files, args, params)

    print(f"Simulation and visualization complete. Results saved in {args.output_dir}")
    print(f"Open {os.path.join(args.output_dir, 'report.html')} to view the visualizations.")

if __name__ == "__main__":
    main()

