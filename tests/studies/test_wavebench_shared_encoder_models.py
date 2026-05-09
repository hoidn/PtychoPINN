import torch

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


def test_unknown_row_is_rejected():
    try:
        wb_models.build_shared_encoder_row(row="does_not_exist", latent_channels=32)
    except ValueError as exc:
        assert "Unknown shared-encoder row" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown shared-encoder row")
