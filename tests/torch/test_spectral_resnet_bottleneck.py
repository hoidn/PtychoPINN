import torch


def test_factorized_spectral_conv_preserves_shape():
    from ptycho_torch.generators.spectral_resnet_bottleneck import FactorizedSpectralConv2d

    layer = FactorizedSpectralConv2d(channels=32, modes=12)
    x = torch.randn(2, 32, 32, 32)

    y = layer(x)

    assert tuple(y.shape) == (2, 32, 32, 32)


def test_shared_spectral_resnet_bottleneck_reuses_one_shared_spectral_operator():
    from ptycho_torch.generators.spectral_resnet_bottleneck import SharedSpectralResnetBottleneck

    bottleneck = SharedSpectralResnetBottleneck(
        channels=32,
        n_blocks=3,
        modes=12,
        share_spectral_weights=True,
    )

    spectral_ids = {id(block.shared_spectral) for block in bottleneck.blocks}

    assert len(spectral_ids) == 1
    assert id(bottleneck.shared_spectral) in spectral_ids


def test_spectral_resnet_block_uses_raw_local_body_not_nested_resnet_block():
    from ptycho_torch.generators.resnet_components import ResnetBlock
    from ptycho_torch.generators.spectral_resnet_bottleneck import (
        FactorizedSpectralConv2d,
        SpectralResnetBlock,
    )

    shared_spectral = FactorizedSpectralConv2d(channels=32, modes=12)
    block = SpectralResnetBlock(channels=32, shared_spectral=shared_spectral)

    assert not isinstance(block.local_conv_body, ResnetBlock)
    assert not hasattr(block.local_conv_body, "layerscale")
    assert hasattr(block, "local_scale")


def test_spectral_resnet_bottleneck_generator_module_forward():
    from ptycho_torch.generators.spectral_resnet_bottleneck import (
        SpectralResnetBottleneckGeneratorModule,
    )

    model = SpectralResnetBottleneckGeneratorModule(
        in_channels=1,
        out_channels=2,
        hidden_channels=32,
        n_blocks=4,
        modes=12,
        C=1,
        hybrid_downsample_steps=2,
        resnet_blocks=6,
        spectral_bottleneck_blocks=6,
        spectral_bottleneck_modes=12,
    )
    x = torch.randn(2, 1, 128, 128)

    y = model(x)

    assert tuple(y.shape) == (2, 128, 128, 1, 2)
