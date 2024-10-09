import numpy as np
import matplotlib.pyplot as plt
from ptycho.loader import PtychoDataContainer

def load_ptycho_data(file_path: str) -> PtychoDataContainer:
    """
    Load the npz-serialized ptycho data.

    Args:
        file_path (str): Path to the npz file.

    Returns:
        PtychoDataContainer: Loaded ptycho data.
    """
    data = np.load(file_path)
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
        probe=data['probe']
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
    # Replace with the actual path to your npz file
    file_path = "path/to/your/ptycho_data.npz"
    
    # Load the data
    ptycho_data = load_ptycho_data(file_path)
    
    # Inspect the frames
    inspect_ptycho_frames(ptycho_data)
