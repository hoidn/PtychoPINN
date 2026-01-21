import numpy as np
import matplotlib.pyplot as plt
from ptycho.loader import PtychoDataContainer, compute_dataset_intensity_stats


def load_ptycho_data(file_path: str) -> PtychoDataContainer:
    """
    Load the npz-serialized ptycho data with dataset_intensity_stats preservation.

    This function loads PtychoDataContainer data from an NPZ file and ensures
    dataset_intensity_stats are preserved for proper intensity_scale calculation.
    Per Phase D4f.2: when loading NPZ files, the function first checks for stored
    stats keys (dataset_intensity_stats_batch_mean, dataset_intensity_stats_n_samples)
    and falls back to recomputing from the X array if not present.

    IMPORTANT: Uses compute_dataset_intensity_stats to ensure proper dataset-derived
    intensity_scale can be computed without touching _tensor_cache (PINN-CHUNKED-001).

    Args:
        file_path (str): Path to the npz file.

    Returns:
        PtychoDataContainer: Loaded ptycho data with dataset_intensity_stats attached.

    See: specs/spec-ptycho-core.md §Normalization Invariants
    See: docs/findings.md PINN-CHUNKED-001
    """
    data = np.load(file_path, allow_pickle=True)

    # Reconstruct dataset_intensity_stats from saved keys if present
    # Otherwise fall back to computing from X data
    dataset_intensity_stats = None
    if 'dataset_intensity_stats_batch_mean' in data and 'dataset_intensity_stats_n_samples' in data:
        # Use stored raw diffraction stats (preferred - these are from pre-normalization)
        batch_mean = float(data['dataset_intensity_stats_batch_mean'])
        n_samples = int(data['dataset_intensity_stats_n_samples'])
        dataset_intensity_stats = {
            'batch_mean_sum_intensity': batch_mean,
            'n_samples': n_samples,
        }
        print(f"inspect_ptycho_data: loaded stored dataset_intensity_stats: "
              f"batch_mean={batch_mean:.6f}, n_samples={n_samples}")
    else:
        # Fall back: compute stats from X array
        # NOTE: This uses normalized data so the stats will approximate the (N/2)²
        # target and calculate_intensity_scale will likely use the closed-form fallback.
        # For proper dataset-derived scale, resave the NPZ with raw stats.
        X = data['X']
        dataset_intensity_stats = compute_dataset_intensity_stats(X, is_normalized=False)
        print(f"inspect_ptycho_data: computed dataset_intensity_stats from X: "
              f"batch_mean={dataset_intensity_stats['batch_mean_sum_intensity']:.6f}, "
              f"n_samples={dataset_intensity_stats['n_samples']}")

    return PtychoDataContainer(
        X=data['X'],
        Y_I=data['Y_I'],
        Y_phi=data['Y_phi'],
        norm_Y_I=data['norm_Y_I'],
        YY_full=data['YY_full'],
        coords_nominal=data['coords_nominal'],
        coords_true=data['coords_true'],
        nn_indices=data['nn_indices'],
        global_offsets=data['global_offsets'],
        local_offsets=data['local_offsets'],
        probeGuess=data['probe'],
        dataset_intensity_stats=dataset_intensity_stats
    )

def inspect_ptycho_frames(data: PtychoDataContainer, num_frames: int = 2):
    """
    Visually inspect a couple of frames from X, Y_I, and Y_phi.

    Args:
        data (PtychoDataContainer): Loaded ptycho data.
        num_frames (int): Number of frames to display. Defaults to 2.
    """
    fig, axes = plt.subplots(3, num_frames, figsize=(5*num_frames, 15))
    
    for i in range(num_frames):
        axes[0, i].imshow(data.X[i, ..., 0], cmap='viridis')
        axes[0, i].set_title(f'X - Frame {i}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(data.Y_I[i, ..., 0], cmap='viridis')
        axes[1, i].set_title(f'Y_I - Frame {i}')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(data.Y_phi[i, ..., 0], cmap='viridis')
        axes[2, i].set_title(f'Y_phi - Frame {i}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inspect_ptycho_data.py <path_to_npz_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    
    try:
        # Load the data
        ptycho_data = load_ptycho_data(file_path)
        
        # Inspect the frames
        inspect_ptycho_frames(ptycho_data)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
