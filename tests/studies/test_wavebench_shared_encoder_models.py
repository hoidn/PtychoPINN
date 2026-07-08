import torch

from ptycho_torch.generators.fno import _FallbackSpectralConv2d

from scripts.studies.wavebench_shared_encoder import models as wb_models


def test_supported_rows_build_for_c32_and_c64_and_emit_single_channel_predictions():
    rows = (
        "cnn",
        "hybrid_resnet",
        "spectral_resnet_bottleneck_net",
        "fno",
        "ffno",
    )

    for latent_channels in (32, 64):
        x = torch.randn(2, 1, 128, 128)
        for row in rows:
            model = wb_models.build_shared_encoder_row(row=row, latent_channels=latent_channels)
            y = model(x)
            profile = wb_models.profile_model(model)

            assert tuple(y.shape) == (2, 1, 128, 128), (row, latent_channels, tuple(y.shape))
            assert profile["encoder_parameters"] > 0
            assert profile["body_parameters"] > 0
            assert (
                profile["total_parameters"]
                == profile["encoder_parameters"] + profile["body_parameters"]
            )


def test_fno_body_uses_real_spectral_convolution():
    model = wb_models.build_shared_encoder_row(row="fno", latent_channels=32)
    spectral_modules = [m for m in model.body.modules() if isinstance(m, _FallbackSpectralConv2d)]

    assert spectral_modules, "fno row must use a real FFT-based spectral conv (none found)"
    for module in spectral_modules:
        assert module.weights.dtype == torch.cfloat, (
            f"spectral conv weights must be complex; got {module.weights.dtype}"
        )
        assert module.modes == 12, f"fno spectral modes must be 12; got {module.modes}"
        assert module.weights.shape[-2:] == (12, 12), (
            f"spectral weight shape must reflect modes=12; got {tuple(module.weights.shape)}"
        )

    profile = wb_models.profile_model(model)
    assert profile["body_parameters"] >= 100_000, (
        "fno body parameter count is too small for a real spectral FNO with modes=12, "
        f"width=32, blocks=4; got {profile['body_parameters']}"
    )


def test_unknown_row_is_rejected():
    try:
        wb_models.build_shared_encoder_row(row="does_not_exist", latent_channels=32)
    except ValueError as exc:
        assert "Unknown shared-encoder row" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown shared-encoder row")
