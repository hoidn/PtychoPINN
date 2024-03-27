import matplotlib.pyplot as plt
from ptycho import model
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

def mk_epie_comparison2x2(ptycho_pinn_phase, epie_phase, ptycho_pinn_amplitude, epie_amplitude):
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

# TODO type annotation
def reconstruct_image(test_data):
    global_offsets = test_data.global_offsets
    local_offsets = test_data.local_offsets

#    obj_tensor_full, _, _ = model.autoencoder.predict(
#                    [test_data['X'] * model.params()['intensity_scale'],
#                                    local_offsets])
    obj_tensor_full = model.diffraction_to_obj.predict(
                    [test_data.X * model.params()['intensity_scale'],
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
