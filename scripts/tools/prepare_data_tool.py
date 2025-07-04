#!/usr/bin/env python3
"""
Ptychography Data Preparation Tool

A unified tool to prepare ptychography datasets by applying one of several
pre-processing techniques. This tool is designed to be chained for multiple
operations (e.g., interpolate first, then smooth).

Operations:
  1. --apodize: Smoothly tapers the probe edges to zero.
  2. --smooth: Applies a Gaussian filter to the specified target ('probe' or 'object').
  3. --interpolate: Upsamples the real-space probe and object via spline
                   interpolation.

Usage Examples:
  # Interpolate, then smooth the probe's amp and phase in two steps:
  # 1. Interpolate
  python prepare_data_tool.py in.npz temp_interp.npz --interpolate --zoom-factor 2.0
  # 2. Smooth the probe in the result of step 1
  python prepare_data_tool.py temp_interp.npz out_final.npz --smooth --target probe --sigma 1.5
"""

import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from scipy.signal.windows import tukey
from skimage.restoration import unwrap_phase
from typing import Tuple, Dict, Any

def load_and_identify_arrays(npz_path: str) -> Tuple[Dict[str, np.ndarray], str, str]:
    """Loads NPZ and identifies probe and object keys."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Input file not found: {npz_path}")

    print(f"Loading data from: {npz_path}")
    with np.load(npz_path) as f:
        data_dict = {key: f[key] for key in f.files}

    if 'probeGuess' not in data_dict or 'objectGuess' not in data_dict:
        raise KeyError("NPZ file must contain both 'probeGuess' and 'objectGuess'.")

    probe_shape = data_dict['probeGuess'].shape
    object_shape = data_dict['objectGuess'].shape

    if np.prod(probe_shape) < np.prod(object_shape):
        probe_key, object_key = 'probeGuess', 'objectGuess'
    else:
        probe_key, object_key = 'objectGuess', 'probeGuess'
        print(f"Warning: 'probeGuess' is larger than 'objectGuess'. Swapping roles.")
    
    print(f"Identified Probe: '{probe_key}' (shape: {data_dict[probe_key].shape})")
    print(f"Identified Object: '{object_key}' (shape: {data_dict[object_key].shape})")
    return data_dict, probe_key, object_key

# --- Transformation Functions ---

def apodize_probe(probe: np.ndarray, alpha: float, apodize_phase: bool) -> np.ndarray:
    """Applies a 2D Tukey window to the probe's amplitude and optionally its phase."""
    print(f"Applying Tukey apodization to probe (alpha={alpha})...")
    if probe.ndim != 2: raise ValueError("Probe must be 2D.")
    
    amplitude = np.abs(probe)
    h, w = amplitude.shape
    window_2d = np.outer(tukey(h, alpha=alpha), tukey(w, alpha=alpha))
    apodized_amplitude = amplitude * window_2d
    print("- Amplitude apodized.")

    if apodize_phase:
        print("- Unwrapping and apodizing phase...")
        unwrapped_phase = unwrap_phase(np.angle(probe))
        final_phase = unwrapped_phase * window_2d
        print("- Phase unwrapped and apodized.")
    else:
        print("- Phase preserved (not apodized).")
        final_phase = np.angle(probe)

    return (apodized_amplitude * np.exp(1j * final_phase)).astype(probe.dtype)

def smooth_complex_array(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Applies a Gaussian filter to a complex array by filtering its real and imaginary parts."""
    print(f"Applying Gaussian filter to complex array of shape {arr.shape} (sigma={sigma})...")
    if not np.iscomplexobj(arr):
        raise ValueError("Input array for smoothing must be complex.")
    
    # To correctly smooth the phase, we must operate on the unwrapped phase
    amplitude = np.abs(arr)
    unwrapped_phase = unwrap_phase(np.angle(arr))

    # Smooth the amplitude and the unwrapped phase separately
    smoothed_amplitude = gaussian_filter(amplitude, sigma=sigma)
    smoothed_unwrapped_phase = gaussian_filter(unwrapped_phase, sigma=sigma)
    
    print("- Amplitude and unwrapped phase smoothed.")

    # Recombine the smoothed components
    return (smoothed_amplitude * np.exp(1j * smoothed_unwrapped_phase)).astype(arr.dtype)

def interpolate_array(arr: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Upsamples a complex 2D array using cubic spline interpolation."""
    if arr.ndim != 2: raise ValueError("Interpolation only supports 2D arrays.")
    print(f"Interpolating array from {arr.shape} with zoom factor {zoom_factor}...")
    
    if np.iscomplexobj(arr):
        real_part = zoom(arr.real, zoom_factor, order=3)
        imag_part = zoom(arr.imag, zoom_factor, order=3)
        interpolated_arr = real_part + 1j * imag_part
    else:
        interpolated_arr = zoom(arr, zoom_factor, order=3)
        
    print(f"  New shape: {interpolated_arr.shape}")
    return interpolated_arr.astype(arr.dtype)

# --- Main Logic and Plotting ---

def save_and_plot(data_dict: Dict[str, Any], output_path: str, title: str):
    """Saves the NPZ and a verification plot."""
    # ... (This function remains the same as the previous version) ...
    print(f"Saving new NPZ file to: {output_path}")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.savez_compressed(output_path, **data_dict)
    print(f"Success! Created {output_path}")

    probe_key = next(k for k in data_dict if 'probe' in k.lower())
    object_key = next(k for k in data_dict if 'object' in k.lower())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"Verification: {title}", fontsize=16)
    
    probe = data_dict[probe_key]
    im_pa = axes[0, 0].imshow(np.abs(probe), cmap='viridis')
    axes[0, 0].set_title(f"Probe Amplitude (Shape: {probe.shape})")
    fig.colorbar(im_pa, ax=axes[0, 0])
    
    im_pp = axes[0, 1].imshow(np.angle(probe), cmap='twilight')
    axes[0, 1].set_title(f"Probe Phase (Shape: {probe.shape})")
    fig.colorbar(im_pp, ax=axes[0, 1])

    obj = data_dict[object_key]
    im_oa = axes[1, 0].imshow(np.abs(obj), cmap='viridis')
    axes[1, 0].set_title(f"Object Amplitude (Shape: {obj.shape})")
    fig.colorbar(im_oa, ax=axes[1, 0])

    im_op = axes[1, 1].imshow(np.angle(obj), cmap='twilight')
    axes[1, 1].set_title(f"Object Phase (Shape: {obj.shape})")
    fig.colorbar(im_op, ax=axes[1, 1])

    for ax in axes.flat: ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = os.path.splitext(output_path)[0] + '_verification.png'
    plt.savefig(plot_filename)
    print(f"Saved verification plot to: {plot_filename}")
    plt.close()


def main():
    """Main function to parse arguments and run the selected operation."""
    parser = argparse.ArgumentParser(
        description="A unified tool for preparing ptychography data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_npz", help="Path to the original .npz file.")
    parser.add_argument("output_npz", help="Path to save the new, processed .npz file.")

    op_group = parser.add_mutually_exclusive_group(required=True)
    op_group.add_argument("--apodize", action="store_true", help="Apply Tukey apodization to the probe.")
    op_group.add_argument("--smooth", action="store_true", help="Apply Gaussian filter to the specified target.")
    op_group.add_argument("--interpolate", action="store_true", help="Upsample probe and object via interpolation.")

    parser.add_argument("--target", choices=['probe', 'object'], default='probe', help="Target for the --smooth operation.")
    parser.add_argument("--alpha", type=float, default=0.25, help="Alpha for Tukey window (for --apodize).")
    parser.add_argument("--sigma", type=float, default=1.0, help="Sigma for Gaussian filter (for --smooth).")
    parser.add_argument("--zoom-factor", type=float, default=2.0, help="Zoom factor for interpolation (for --interpolate).")
    parser.add_argument("--apodize-phase", action="store_true", help="Also unwrap and apodize phase (for --apodize).")

    args = parser.parse_args()

    data_dict, probe_key, object_key = load_and_identify_arrays(args.input_npz)
    
    if args.apodize:
        title = f"Probe Apodized (alpha={args.alpha}, phase={args.apodize_phase})"
        data_dict[probe_key] = apodize_probe(data_dict[probe_key], args.alpha, args.apodize_phase)
    
    elif args.smooth:
        if args.target == 'probe':
            target_key = probe_key
        else:
            target_key = object_key
        title = f"{args.target.capitalize()} Smoothed (sigma={args.sigma})"
        data_dict[target_key] = smooth_complex_array(data_dict[target_key], args.sigma)
        
    elif args.interpolate:
        title = f"Interpolated (Zoom Factor={args.zoom_factor})"
        data_dict[probe_key] = interpolate_array(data_dict[probe_key], args.zoom_factor)
        data_dict[object_key] = interpolate_array(data_dict[object_key], args.zoom_factor)

    save_and_plot(data_dict, args.output_npz, title)

if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError, argparse.ArgumentError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
