import torch
import numpy as np
from .tf_helper import *

def test_get_mask():
    input_tensor = torch.tensor([[1.0, 0.5, 0.8], [0.3, 0.9, 0.2]])
    support_threshold = 0.6
    expected_output = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    assert torch.all(torch.eq(get_mask(input_tensor, support_threshold), expected_output))

def test_combine_complex():
    amp = torch.tensor([[1.0, 0.5], [0.8, 0.3]])
    phi = torch.tensor([[0.0, np.pi/2], [np.pi/4, np.pi]])
    expected_output = torch.view_as_complex(torch.tensor([[[1.0, 0.0], [0.0, 0.5]], [[0.5657, 0.5657], [-0.3, 0.0]]]))
    assert torch.allclose(combine_complex(amp, phi), expected_output)

def test_pad_obj():
    input_tensor = torch.ones((1, 4, 4, 1))
    h, w = 8, 8
    expected_output = torch.ones((1, 8, 8, 1))
    assert torch.all(torch.eq(pad_obj(input_tensor, h, w), expected_output))

def test__fromgrid():
    params()['N'] = 2
    img = torch.ones((1, 2, 2, 2, 2, 1))
    expected_output = torch.ones((4, 2, 2, 1))
    assert torch.all(torch.eq(_fromgrid(img), expected_output))

def test__togrid():
    params()['N'] = 2
    params()['gridsize'] = 2
    img = torch.ones((4, 2, 2, 1))
    expected_output = torch.ones((1, 2, 2, 2, 2, 1))
    assert torch.all(torch.eq(_togrid(img), expected_output))

def test_togrid():
    params()['N'] = 2
    params()['gridsize'] = 2
    img1 = torch.ones((4, 2, 2, 1))
    img2 = torch.ones((4, 2, 2, 1))
    expected_output = (torch.ones((1, 2, 2, 2, 2, 1)), torch.ones((1, 2, 2, 2, 2, 1)))
    assert all(torch.all(torch.eq(x, y)) for x, y in zip(togrid(img1, img2), expected_output))

def test__grid_to_channel():
    params()['gridsize'] = 2
    grid = torch.ones((1, 2, 2, 3, 3, 1))
    expected_output = torch.ones((1, 3, 3, 4))
    assert torch.all(torch.eq(_grid_to_channel(grid), expected_output))

def test_grid_to_channel():
    params()['gridsize'] = 2
    grid1 = torch.ones((1, 2, 2, 3, 3, 1))
    grid2 = torch.ones((1, 2, 2, 3, 3, 1))
    expected_output = (torch.ones((1, 3, 3, 4)), torch.ones((1, 3, 3, 4)))
    assert all(torch.all(torch.eq(x, y)) for x, y in zip(grid_to_channel(grid1, grid2), expected_output))

def test__flat_to_channel():
    params()['gridsize'] = 2
    params()['N'] = 3
    img = torch.ones((4, 3, 3, 1))
    expected_output = torch.ones((1, 3, 3, 4))
    assert torch.all(torch.eq(_flat_to_channel(img), expected_output))

def test__flat_to_channel_2():
    params()['gridsize'] = 2
    img = torch.ones((1, 3, 4, 1))
    expected_output = torch.ones((1, 3, 4, 4))
    assert torch.all(torch.eq(_flat_to_channel_2(img), expected_output))

def test__channel_to_flat():
    img = torch.ones((1, 3, 3, 4))
    expected_output = torch.ones((4, 3, 3, 1))
    assert torch.all(torch.eq(_channel_to_flat(img), expected_output))

def test__channel_to_patches():
    params()['gridsize'] = 2
    params()['N'] = 3
    channel = torch.ones((1, 3, 3, 4))
    expected_output = torch.ones((1, 2, 2, 9))
    assert torch.all(torch.eq(_channel_to_patches(channel), expected_output))

def test_pad_patches():
    imgs = torch.ones((1, 4, 4, 1))
    padded_size = 8
    expected_output = torch.ones((1, 8, 8, 1))
    assert torch.all(torch.eq(pad_patches(imgs, padded_size), expected_output))

def test_pad():
    imgs = torch.ones((1, 4, 4, 1))
    size = 2
    expected_output = torch.ones((1, 8, 8, 1))
    assert torch.all(torch.eq(pad(imgs, size), expected_output))

def test_trim_reconstruction():
    x = torch.ones((1, 8, 8, 1))
    N = 4
    expected_output = torch.ones((1, 4, 4, 1))
    assert torch.all(torch.eq(trim_reconstruction(x, N), expected_output))

def test_flatten_offsets():
    channels = torch.ones((1, 3, 3, 4))
    expected_output = torch.ones((4, 3))
    assert torch.all(torch.eq(flatten_offsets(channels), expected_output))

def test_pad_reconstruction():
    channels = torch.ones((1, 3, 3, 4))
    expected_output = torch.ones((4, 7, 7, 1))
    assert torch.all(torch.eq(pad_reconstruction(channels), expected_output))

def test_pad_and_diffract():
    input_tensor = torch.ones((1, 4, 4, 1))
    h, w = 8, 8
    padded_expected = torch.ones((1, 8, 8, 1))
    input_expected = torch.ones((1, 8, 8, 1))
    padded, input = pad_and_diffract(input_tensor, h, w)
    assert torch.all(torch.eq(padded, padded_expected))
    assert torch.allclose(input, input_expected, atol=1e-6)
