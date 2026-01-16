"""
Interactive Jupyter Notebook Utilities for PtychoPINN Development

This module provides specialized visualization and development utilities for interactive analysis 
in Jupyter notebooks, focusing on ptychographic reconstruction results comparison, debugging, 
and exploratory data analysis. It bridges the core reconstruction pipeline with interactive 
development workflows.

Primary Functions:
  - compare(): Automated side-by-side comparison of PtychoPINN vs ground truth reconstructions
  - mk_comparison(): Flexible visualization of 2-3 reconstruction methods with phase/amplitude plots
  - reconstruct_image(): Single-function interface for model inference on test data
  - probeshow(): Interactive probe visualization with scan position overlay
  - print_shapes(): Debug utility for data container inspection

Architecture Integration:
  This module operates at the high-level workflow tier, consuming processed data from the 
  loader.py pipeline and providing visualization interfaces for model outputs. It serves as 
  the primary interface between the core PtychoPINN system and interactive research workflows.

Data Flow Integration:
  ```python
  # Typical interactive workflow
  test_data = loader.PtychoDataContainer(...)  # From ptycho.loader
  obj_tensor, offsets = reconstruct_image(test_data)  # This module
  compare(obj_tensor, offsets, ground_truth)  # Visualization
  ```

Legacy System Dependencies:
  - Depends on ptycho.params global configuration for intensity scaling in reconstruct_image()
  - Uses legacy loader.reassemble_position() for grid-based patch stitching
  - Compatible with both modern coordinate-based and legacy grid-based data flows

Development Context:
  Designed for exploratory analysis during model development, hyperparameter tuning, and 
  result validation. Functions prioritize immediate visual feedback over production efficiency.
  The cropping utilities help focus analysis on scientifically relevant regions by removing 
  background padding from reconstructed images.

Typical Interactive Usage:
  ```python
  # Load and process data
  test_data = loader.create_test_data(npz_file)
  
  # Generate reconstruction
  obj_tensor, global_offsets = reconstruct_image(test_data)
  
  # Compare with ground truth
  compare(obj_tensor, global_offsets, test_data.objectGuess)
  
  # Inspect probe properties
  probeshow(test_data.probeGuess, test_data)
  
  # Debug data shapes
  print_shapes(test_data.__dict__)
  ```

Notes:
  - All visualization functions use matplotlib with notebook-optimized layouts
  - Cropping functions assume scientific images with uniform background regions
  - Comparison functions automatically handle complex number visualization (phase/amplitude)
  - Functions are designed for single-call convenience rather than programmatic efficiency
"""

import matplotlib.pyplot as plt
import numpy as np

def crop_to_non_uniform_region_with_buffer(img_array, buffer=0):
    """
    Crop the image to the non-uniform region with an additional buffer in each direction.

    Parameters:
    - img_array: The numpy array of the image.
    - buffer: The number of pixels to expand the cropped region in each direction.

    Returns:
    - cropped_img_array: The numpy array of the cropped image.
    """

    # Convert to grayscale if it is not already
    if len(img_array.shape) == 3:
        gray_img_array = img_array[:, :, 0]
    else:
        gray_img_array = img_array

    # Find the background pixel value, assuming it is the mode of the corner pixels
    corner_pixels = [gray_img_array[0, 0], gray_img_array[0, -1], gray_img_array[-1, 0], gray_img_array[-1, -1]]
    background_pixel = max(set(corner_pixels), key=corner_pixels.count)

    # Detect the non-uniform region
    rows, cols = np.where(gray_img_array != background_pixel)
    if rows.size > 0 and cols.size > 0:
        row_min, row_max, col_min, col_max = rows.min(), rows.max(), cols.min(), cols.max()
        # Apply the buffer, ensuring we don't go out of the image bounds
        row_min = max(row_min - buffer, 0)
        row_max = min(row_max + buffer, gray_img_array.shape[0] - 1)
        col_min = max(col_min - buffer, 0)
        col_max = min(col_max + buffer, gray_img_array.shape[1] - 1)
    else:
        raise ValueError("No non-uniform region found")

    # Crop the image to the non-uniform region with the buffer
    cropped_img_array = gray_img_array[row_min:row_max+1, col_min:col_max+1]

    return cropped_img_array

import matplotlib.pyplot as plt

def mk_comparison(method1, method2, method1_name='PtychoNN', method2_name='ground truth', method0=None, method0_name='ePIE', phase_vmin=None, phase_vmax=None):
    """
    Create a comparison plot of phase and amplitude images for 2 or 3 methods.

    Parameters:
    - method1: Complex 2D array of method1 data
    - method2: Complex 2D array of method2 data
    - method1_name: Name of the first method (default: 'PtychoNN')
    - method2_name: Name of the second method (default: 'ground truth')
    - method0: Complex 2D array of method0 data (optional)
    - method0_name: Name of the optional third method (default: 'ePIE')
    - phase_vmin: Minimum data value for phase plots (optional)
    - phase_vmax: Maximum data value for phase plots (optional)
    """
    num_methods = 3 if method0 is not None else 2
    fig, axs = plt.subplots(2, num_methods, figsize=(5*num_methods, 10))

    methods = [method0, method1, method2] if num_methods == 3 else [method1, method2]
    method_names = [method0_name, method1_name, method2_name] if num_methods == 3 else [method1_name, method2_name]

    for i, (method, name) in enumerate(zip(methods, method_names)):
        # Phase plot
        phase_img = axs[0, i].imshow(np.angle(method), cmap='gray', vmin=phase_vmin, vmax=phase_vmax)
        axs[0, i].set_title(f'{name} Phase')
        axs[0, i].axis('off')
        fig.colorbar(phase_img, ax=axs[0, i], orientation='vertical')

        # Amplitude plot
        amp_img = axs[1, i].imshow(np.abs(method), cmap='viridis')
        axs[1, i].set_title(f'{name} Amplitude')
        axs[1, i].axis('off')
        fig.colorbar(amp_img, ax=axs[1, i], orientation='vertical')

    # Adjust layout to prevent overlap
    plt.tight_layout(pad=3.0)
    plt.show()

def compare(obj_tensor_full, global_offsets, objectGuess, ptychonn_tensor=None):
    from ptycho import loader

    # Process PtychoPINN data
    ptychopinn_image = loader.reassemble_position(obj_tensor_full, global_offsets[:, :, :, :], M=20)
    ptychopinn_phase = crop_to_non_uniform_region_with_buffer(np.angle(ptychopinn_image[..., 0]), buffer=-20)
    ptychopinn_amplitude = crop_to_non_uniform_region_with_buffer(np.abs(ptychopinn_image[..., 0]), buffer=-20)

    # Process ground truth data
    gt_phase = crop_to_non_uniform_region_with_buffer(np.angle(objectGuess), buffer=-20)
    gt_amplitude = crop_to_non_uniform_region_with_buffer(np.abs(objectGuess), buffer=-20)

    # Process PtychoNN data if provided
    if ptychonn_tensor is not None:
        ptychonn_image = loader.reassemble_position(ptychonn_tensor, global_offsets[:, :, :, :], M=20)
        ptychonn_phase = crop_to_non_uniform_region_with_buffer(np.angle(ptychonn_image[..., 0]), buffer=-20)
        ptychonn_amplitude = crop_to_non_uniform_region_with_buffer(np.abs(ptychonn_image[..., 0]), buffer=-20)
        
        # Create comparison with all three methods
        mk_comparison(ptychopinn_phase + 1j * ptychopinn_amplitude, 
                      gt_phase + 1j * gt_amplitude, 
                      method1_name='PtychoPINN', 
                      method2_name='ground truth',
                      method0=ptychonn_phase + 1j * ptychonn_amplitude, 
                      method0_name='PtychoNN')
    else:
        # Create comparison with only PtychoPINN and ground truth
        mk_comparison(ptychopinn_phase + 1j * ptychopinn_amplitude, 
                      gt_phase + 1j * gt_amplitude, 
                      method1_name='PtychoPINN', 
                      method2_name='ground truth')

# TODO type annotation
def reconstruct_image(test_data, diffraction_to_obj = None):
    from ptycho import model, params # Import delayed to avoid early model graph construction
    global_offsets = test_data.global_offsets
    local_offsets = test_data.local_offsets

    if diffraction_to_obj is None:
        diffraction_to_obj = model.diffraction_to_obj
    obj_tensor_full = diffraction_to_obj.predict(
                    [test_data.X * params.params()['intensity_scale'],
                    local_offsets])
    return obj_tensor_full, global_offsets

def print_shapes(test_data):
    for key, value in test_data.items():
        if value is not None:
            if isinstance(value, tuple):
                print(f"{key}\t")
                for i, array in enumerate(value):
                    print(f"  Array {i+1}{array.shape}, \t {array.dtype}")
            else:
                print(f"{key}\t{value.shape}, {value.dtype}")

def probeshow(probeGuess, test_data):
    # Creating a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plotting the magnitude of the complex array
    img1 = ax1.imshow(np.abs(probeGuess), cmap='viridis')
    ax1.set_title('probe amplitude')
    fig.colorbar(img1, ax=ax1, orientation='vertical')

    # Plotting the phase of the complex array
    img2 = ax2.imshow(np.angle(probeGuess), cmap='jet')
    ax2.set_title('probe phase')
    fig.colorbar(img2, ax=ax2, orientation='vertical')

    # Plotting the scan point positions
    ax3.scatter(*(test_data.global_offsets.squeeze().T))
    ax3.set_title('scan point positions')

    # Improving layout
    plt.tight_layout()
    plt.show()


def track_dict_changes(input_dict, callback):
    # Copy the original dictionary to track changes
    original_dict = input_dict.copy()
    # Execute the callback function
    callback(input_dict)
    # Determine which keys have changed or added
    changed_or_added_keys = [key for key in input_dict if input_dict.get(key) != original_dict.get(key)]
    return changed_or_added_keys

def mk_epie_comparison2x2(ptycho_pinn_phase, epie_phase, ptycho_pinn_amplitude,epie_amplitude):
    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # PtychoPINN phase with color bar
    ptycho_pinn_phase_img = axs[0, 0].imshow(ptycho_pinn_phase, cmap='gray')
    axs[0, 0].set_title('PtychoPINN Phase')
    axs[0, 0].axis('off')
    fig.colorbar(ptycho_pinn_phase_img, ax=axs[0, 0], orientation='vertical')

    # ePIE phase with color bar
    epie_phase_img = axs[0, 1].imshow(epie_phase, cmap='gray')
    axs[0, 1].set_title('ePIE Phase')
    axs[0, 1].axis('off')
    fig.colorbar(epie_phase_img, ax=axs[0, 1], orientation='vertical')

    # PtychoPINN amplitude with color bar
    ptycho_pinn_amplitude_img = axs[1, 0].imshow(ptycho_pinn_amplitude, cmap='gray')#,
                                               # vmin = .2
    axs[1, 0].set_title('PtychoPINN Amplitude')
    axs[1, 0].axis('off')
    fig.colorbar(ptycho_pinn_amplitude_img, ax=axs[1, 0], orientation='vertical')

    # ePIE amplitude with color bar
    epie_amplitude_img = axs[1, 1].imshow(epie_amplitude, cmap='gray')
    axs[1, 1].set_title('ePIE Amplitude')
    axs[1, 1].axis('off')
    fig.colorbar(epie_amplitude_img, ax=axs[1, 1], orientation='vertical')

    # Adjust layout to prevent overlap
    plt.tight_layout(pad=3.0)

    plt.show()

# object heatmaps
## Creating a figure and two subplots
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#
## Plotting the amplitude of the complex object
#ax1.imshow(np.absolute(objectGuess), cmap='viridis')
#ax1.set_title('Amplitude')
#
## Plotting the phase of the complex object
#ax2.imshow(np.angle(objectGuess), cmap='viridis')
#ax2.set_title('Phase')
#
## Adjust layout
#plt.tight_layout()
#plt.show()
