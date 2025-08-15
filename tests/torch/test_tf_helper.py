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

import torch
from typing import Callable

def test_mk_centermask():
    inputs = torch.ones((2, 8, 8, 3))
    N = 4
    c = 3

    # Test case 1: Check if the function returns the correct center mask when kind='center'
    expected_center_mask = torch.zeros((2, 8, 8, 3))
    expected_center_mask[:, 2:6, 2:6, :] = 1
    center_mask = mk_centermask(inputs, N, c, kind='center')
    assert torch.allclose(center_mask, expected_center_mask)

    # Test case 2: Check if the function returns the correct border mask when kind='border'
    expected_border_mask = torch.ones((2, 8, 8, 3))
    expected_border_mask[:, 2:6, 2:6, :] = 0
    border_mask = mk_centermask(inputs, N, c, kind='border')
    assert torch.allclose(border_mask, expected_border_mask)

    # Test case 3: Check if the function raises a ValueError when kind is not 'center' or 'border'
    try:
        mk_centermask(inputs, N, c, kind='invalid')
        assert False, "Expected ValueError was not raised"
    except ValueError:
        pass

def test_mk_norm():
    channels = torch.ones((2, 8, 8, 4))

    # Mock fn_reassemble_real function
    def mock_fn_reassemble_real(tensor, average=False):
        return tensor * 2

    # Test case 1: Check if the function returns the correct norm values
    expected_norm = torch.ones((2, 8, 8, 4)) * 2 + 0.001
    norm = mk_norm(channels, mock_fn_reassemble_real)
    assert torch.allclose(norm, expected_norm)

    # Test case 2: Check if the function handles different input shapes correctly
    channels_2 = torch.ones((4, 16, 16, 8))
    expected_norm_2 = torch.ones((4, 16, 16, 8)) * 2 + 0.001
    norm_2 = mk_norm(channels_2, mock_fn_reassemble_real)
    assert torch.allclose(norm_2, expected_norm_2)

def test_reassemble_patches():
    channels = torch.ones((2, 8, 8, 4)) + 1j * torch.ones((2, 8, 8, 4))

    # Mock fn_reassemble_real function
    def mock_fn_reassemble_real(tensor, average=False):
        return tensor * 2

    # Test case 1: Check if the function correctly reassembles patches when average=False
    expected_output = torch.ones((2, 8, 8, 4)) * 2 + 1j * torch.ones((2, 8, 8, 4)) * 2
    output = reassemble_patches(channels, mock_fn_reassemble_real, average=False)
    assert torch.allclose(output, expected_output)

    # Test case 2: Check if the function correctly reassembles patches when average=True
    expected_output_avg = torch.ones((2, 8, 8, 4)) * 2 + 1j * torch.ones((2, 8, 8, 4)) * 2
    output_avg = reassemble_patches(channels, mock_fn_reassemble_real, average=True)
    assert torch.allclose(output_avg, expected_output_avg)

    # Test case 3: Check if the function handles complex input channels correctly
    channels_real = torch.ones((2, 8, 8, 4))
    channels_imag = torch.ones((2, 8, 8, 4)) * 2
    channels_complex = torch.complex(channels_real, channels_imag)
    expected_output_complex = torch.ones((2, 8, 8, 4)) * 2 + 1j * torch.ones((2, 8, 8, 4)) * 4
    output_complex = reassemble_patches(channels_complex, mock_fn_reassemble_real, average=False)
    assert torch.allclose(output_complex, expected_output_complex)

def test__reassemble_patches_position_real():
    imgs = torch.ones((2, 8, 8, 4))
    offsets_xy = torch.tensor([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
    padded_size = 16

    # Mock Translation class
    class MockTranslation:
        def __call__(self, inputs):
            return inputs[0] + inputs[1].unsqueeze(-1).unsqueeze(-1)

    # Mock helper functions
    def mock_flatten_offsets(offsets_xy):
        return offsets_xy.view(-1, 2)

    def mock__channel_to_flat(imgs):
        return imgs.view(-1, 8, 8, 1)

    def mock_pad_patches(imgs_flat, padded_size):
        return torch.ones((8, padded_size, padded_size, 1))

    def mock__flat_to_channel(imgs_flat_bigN_translated, N):
        return imgs_flat_bigN_translated.view(2, N, N, 4)

    # Test case 1: Check if the function correctly reassembles patches when agg=True
    expected_output_agg = torch.ones((2, padded_size, padded_size, 1)) * 4
    output_agg = _reassemble_patches_position_real(imgs, offsets_xy, agg=True, padded_size=padded_size,
                                                   flatten_offsets=mock_flatten_offsets,
                                                   _channel_to_flat=mock__channel_to_flat,
                                                   pad_patches=mock_pad_patches,
                                                   Translation=MockTranslation(),
                                                   _flat_to_channel=mock__flat_to_channel)
    assert torch.allclose(output_agg, expected_output_agg)

    # Test case 2: Check if the function correctly reassembles patches when agg=False
    expected_output_no_agg = torch.ones((2, padded_size, padded_size, 4))
    output_no_agg = _reassemble_patches_position_real(imgs, offsets_xy, agg=False, padded_size=padded_size,
                                                      flatten_offsets=mock_flatten_offsets,
                                                      _channel_to_flat=mock__channel_to_flat,
                                                      pad_patches=mock_pad_patches,
                                                      Translation=MockTranslation(),
                                                      _flat_to_channel=mock__flat_to_channel)
    assert torch.allclose(output_no_agg, expected_output_no_agg)

    # Test case 3: Check if the function handles different input shapes and offsets correctly
    imgs_2 = torch.ones((4, 16, 16, 8))
    offsets_xy_2 = torch.tensor([[[1, 1], [2, 2], [3, 3], [4, 4]], [[5, 5], [6, 6], [7, 7], [8, 8]]])
    padded_size_2 = 32
    expected_output_agg_2 = torch.ones((4, padded_size_2, padded_size_2, 1)) * 8
    output_agg_2 = _reassemble_patches_position_real(imgs_2, offsets_xy_2, agg=True, padded_size=padded_size_2,
                                                     flatten_offsets=mock_flatten_offsets,
                                                     _channel_to_flat=mock__channel_to_flat,
                                                     pad_patches=mock_pad_patches,
                                                     Translation=MockTranslation(),
                                                     _flat_to_channel=mock__flat_to_channel)
    assert torch.allclose(output_agg_2, expected_output_agg_2)

def test_mk_reassemble_position_real():
    input_positions = torch.tensor([[[1, 1], [2, 2]], [[3, 3], [4, 4]]])

    # Mock _reassemble_patches_position_real function
    def mock__reassemble_patches_position_real(imgs, offsets_xy, **kwargs):
        return imgs + offsets_xy.sum()

    # Test case 1: Check if the function returns a callable that correctly reassembles patches
    imgs = torch.ones((2, 8, 8, 4))
    expected_output = imgs + 10
    reassemble_fn = mk_reassemble_position_real(input_positions)
    output = reassemble_fn(imgs)
    assert torch.allclose(output, expected_output)

    # Test case 2: Check if the returned callable handles different input shapes and keyword arguments correctly
    imgs_2 = torch.ones((4, 16, 16, 8))
    expected_output_2 = imgs_2 + 10
    output_2 = reassemble_fn(imgs_2)
    assert torch.allclose(output_2, expected_output_2)

    import torch
import numpy as np

def test_translate():
    # Test case 1: Single input tensor
    tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    translation = torch.tensor([1.0, -1.0], dtype=torch.float32)
    expected_output = torch.tensor([[5.0, 6.0, 0.0], [2.0, 3.0, 0.0]], dtype=torch.float32)
    output = translate(tensor.unsqueeze(0), translation.unsqueeze(0)).squeeze(0)
    assert torch.allclose(output, expected_output, atol=1e-6)

    # Test case 2: Batched input tensors
    batch_size = 2
    channels = 3
    height = 4
    width = 5
    imgs = torch.randn(batch_size, channels, height, width)
    offsets = torch.tensor([[1.0, -1.0], [-2.0, 2.0]], dtype=torch.float32)
    expected_output = torch.tensor([
        [
            [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.1132, 0.1562, 0.1697],
             [0.0000, 0.7470, 0.8155, 0.1878, 0.4034],
             [0.0000, 1.3378, 0.9931, 0.2400, 0.2372]],
            [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.6398, 0.8829, 0.9593],
             [0.0000, 1.0086, 1.1025, 0.2537, 0.5450],
             [0.0000, 0.8617, 0.6401, 0.1547, 0.1529]],
            [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.2566, 0.3541, 0.3848],
             [0.0000, 0.5297, 0.5783, 0.1331, 0.2861],
             [0.0000, 0.6287, 0.4674, 0.1129, 0.1117]]
        ],
        [
            [[0.0000, 0.9102, 0.8985, 0.0256, 0.1092],
             [0.0000, 0.0082, 0.1560, 0.1651, 0.1176],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
            [[0.0000, 0.6713, 0.6630, 0.0189, 0.0806],
             [0.0000, 0.0118, 0.2246, 0.2374, 0.1691],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]],
            [[0.0000, 0.5032, 0.4970, 0.0142, 0.0604],
             [0.0000, 0.0117, 0.2213, 0.2342, 0.1667],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]
        ]
    ], dtype=torch.float32)
    output = translate(imgs, offsets)
    assert torch.allclose(output, expected_output, atol=1e-4)

    # Test case 3: Complex input tensor
    real_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    imag_tensor = torch.tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], dtype=torch.float32)
    complex_tensor = torch.complex(real_tensor, imag_tensor)
    translation = torch.tensor([1.0, -1.0], dtype=torch.float32)
    expected_real_output = torch.tensor([[5.0, 6.0, 0.0], [2.0, 3.0, 0.0]], dtype=torch.float32)
    expected_imag_output = torch.tensor([[2.5, 3.0, 0.0], [1.0, 1.5, 0.0]], dtype=torch.float32)
    expected_output = torch.complex(expected_real_output, expected_imag_output)
    output = translate(complex_tensor.unsqueeze(0), translation.unsqueeze(0)).squeeze(0)
    assert torch.allclose(output, expected_output, atol=1e-6)

