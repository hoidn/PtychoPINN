import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

from scipy.ndimage import zoom
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.ndimage import gaussian_filter as gf

def letter_to_array(letter, font_path, font_size, image_size):
    # Create a blank image
    img = Image.new('L', image_size, 255)  # 'L' stands for 8-bit pixels, black and white

    # Get drawing context
    d = ImageDraw.Draw(img)

    # Define font
    font = ImageFont.truetype(font_path, font_size)

    # Get text width and height
    text_width, text_height = d.textsize(letter, font=font)

    # Calculate X, Y position of the text
    x = (image_size[0] - text_width) / 2
    y = (image_size[1] - text_height) / 2

    # Draw the text onto the image
    d.text((x, y), letter, font=font, fill=(0))

    # Convert the image data to a numpy array
    data = np.array(img)

    # Convert to binary (0 and 1)
    binary_data = np.where(data < 128, 1, 0)

    return binary_data

# Use a font available on your system (this path is for demonstration; adjust accordingly)
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
sprite = letter_to_array('V', font_path, font_size=50, image_size=(60, 60))

def create_canvas(size):
    return np.zeros((size, size))

def create_sprite():
    return sprite

def add_sprite_to_canvas(canvas, sprite, repetitions):
    for _ in range(repetitions):
        scale = 0.05 + .4 * np.random.rand()
        scaled_sprite = scipy.ndimage.zoom(sprite, scale)

        tx = np.random.randint(0, canvas.shape[0] - scaled_sprite.shape[0])
        ty = np.random.randint(0, canvas.shape[1] - scaled_sprite.shape[1])

        x_end = min(tx + scaled_sprite.shape[0], canvas.shape[0])
        y_end = min(ty + scaled_sprite.shape[1], canvas.shape[1])

        canvas[tx:x_end, ty:y_end] += scaled_sprite[:x_end-tx, :y_end-ty]

#def visualize_canvas(canvas):
#    plt.imshow(canvas, cmap='gray')
#    plt.show()

def mk_vs(N, nfeats = 1000):
    from . import fourier as f
    assert not N % 2
    canvas = create_canvas(N)
    sprite = create_sprite()
    add_sprite_to_canvas(canvas, sprite, nfeats)
    res = canvas[..., None]
    res = f.gf(res, 1) + 2 * f.gf(res, 5) + 5 * f.gf(res, 10)
    return res / res.max()
#    res = np.zeros((N, N, 1))
#    res[:, :, :] = generate_map(indexlaw, sigma, threshold, boxsize)[..., None]
#    return res
