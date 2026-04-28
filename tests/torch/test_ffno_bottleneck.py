import torch


def test_factorized_ffno_block_preserves_shape():
    from ptycho_torch.generators.ffno_bottleneck import FactorizedFfnoBlock
    from ptycho_torch.generators.spectral_resnet_bottleneck import FactorizedSpectralConv2d

    shared = FactorizedSpectralConv2d(channels=128, modes=12)
    block = FactorizedFfnoBlock(channels=128, shared_spectral=shared, mlp_ratio=2.0, norm="instance")
    x = torch.randn(2, 128, 32, 32)

    y = block(x)

    assert tuple(y.shape) == (2, 128, 32, 32)


def test_shared_factorized_ffno_bottleneck_reuses_one_shared_spectral_operator():
    from ptycho_torch.generators.ffno_bottleneck import SharedFactorizedFfnoBottleneck

    bottleneck = SharedFactorizedFfnoBottleneck(
        channels=64,
        n_blocks=3,
        modes=12,
        share_spectral_weights=True,
        mlp_ratio=2.0,
        gate_init=0.1,
        norm="instance",
    )

    spectral_ids = {id(block.shared_spectral) for block in bottleneck.blocks}

    assert len(spectral_ids) == 1
    assert id(bottleneck.shared_spectral) in spectral_ids


def test_factorized_ffno_block_has_expand_act_project_feedforward_path():
    from ptycho_torch.generators.ffno_bottleneck import FactorizedFfnoBlock
    from ptycho_torch.generators.spectral_resnet_bottleneck import FactorizedSpectralConv2d

    shared = FactorizedSpectralConv2d(channels=32, modes=8)
    block = FactorizedFfnoBlock(channels=32, shared_spectral=shared, mlp_ratio=2.0, norm="instance")

    assert isinstance(block.expand, torch.nn.Conv2d)
    assert isinstance(block.act, torch.nn.GELU)
    assert isinstance(block.project, torch.nn.Conv2d)


def test_ffno_bottleneck_generator_module_forward_preserves_shape():
    from ptycho_torch.generators.ffno_bottleneck import FfnoBottleneckGeneratorModule

    model = FfnoBottleneckGeneratorModule(
        channels=128,
        n_blocks=4,
        modes=12,
        share_spectral_weights=True,
        mlp_ratio=2.0,
        gate_init=0.1,
        norm="instance",
    )
    x = torch.randn(2, 128, 32, 32)

    y = model(x)

    assert tuple(y.shape) == (2, 128, 32, 32)


def test_factorized_ffno_block_localconv_preserves_shape_and_exposes_explicit_branch():
    from ptycho_torch.generators.ffno_bottleneck import FactorizedFfnoBlock
    from ptycho_torch.generators.spectral_resnet_bottleneck import FactorizedSpectralConv2d

    shared = FactorizedSpectralConv2d(channels=64, modes=12)
    block = FactorizedFfnoBlock(
        channels=64,
        shared_spectral=shared,
        mlp_ratio=2.0,
        norm="instance",
        local_conv_kernel_size=3,
    )
    x = torch.randn(2, 64, 32, 32)

    y = block(x)

    assert tuple(y.shape) == (2, 64, 32, 32)
    assert isinstance(block.local_conv, torch.nn.Conv2d)
    assert block.local_conv.kernel_size == (3, 3)
    assert block.local_conv.padding == (1, 1)
