import torch


def test_image128_model_profiles_build_and_record_parameter_counts():
    from scripts.studies.pdebench_image128.models import build_model_from_profile, describe_model
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    for profile_id in [
        "hybrid_resnet_base",
        "hybrid_resnet_cns",
        "hybrid_resnet_cns_transpose",
        "hybrid_resnet_base_down1",
        "hybrid_resnet_skip_add",
        "hybrid_resnet_modes24",
        "spectral_resnet_bottleneck_base",
        "spectral_resnet_bottleneck_noshare",
        "ffno_bottleneck_base",
        "fno_base",
        "unet_strong",
        "unet_tiny_smoke",
    ]:
        profile = get_model_profile(profile_id)
        model = build_model_from_profile(
            profile,
            in_channels=1,
            out_channels=1,
            spatial_shape=(128, 128),
        )
        y = model(torch.zeros(1, 1, 128, 128))
        description = describe_model(model, profile=profile)

        assert tuple(y.shape) == (1, 1, 128, 128)
        assert description["profile_id"] == profile_id
        assert description["parameter_count"] > 0


def test_hybrid_resnet_image_model_uses_stride_conv_downsampling():
    from ptycho_torch.generators.hybrid_resnet import StrideConvDownsample
    from scripts.studies.pdebench_image128.models import HybridResnetImageModel

    model = HybridResnetImageModel(
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        fno_modes=12,
        fno_blocks=4,
        resnet_blocks=6,
        downsample_steps=2,
    )

    assert len(model.downsample_layers) == 2
    assert all(isinstance(layer, StrideConvDownsample) for layer in model.downsample_layers)


def test_hybrid_resnet_modes24_profile_only_changes_fno_modes():
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    base = get_model_profile("hybrid_resnet_base").to_model_config()
    modes24 = get_model_profile("hybrid_resnet_modes24").to_model_config()

    assert base["fno_modes"] == 12
    assert modes24["fno_modes"] == 24
    assert {key: value for key, value in modes24.items() if key not in {"profile_id", "fno_modes"}} == {
        key: value for key, value in base.items() if key not in {"profile_id", "fno_modes"}
    }


def test_hybrid_resnet_base_down1_profile_only_changes_downsample_depth():
    from scripts.studies.pdebench_image128.models import PadCropWrapper, build_model_from_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    base = get_model_profile("hybrid_resnet_base").to_model_config()
    down1 = get_model_profile("hybrid_resnet_base_down1").to_model_config()

    assert down1["hybrid_downsample_steps"] == 1
    assert {key: value for key, value in down1.items() if key not in {"profile_id", "hybrid_downsample_steps", "evidence_scope"}} == {
        key: value for key, value in base.items() if key not in {"profile_id", "evidence_scope", "hybrid_downsample_steps"}
    }

    model = build_model_from_profile(
        get_model_profile("hybrid_resnet_base_down1"),
        in_channels=8,
        out_channels=4,
        spatial_shape=(128, 128),
    )
    assert isinstance(model, PadCropWrapper)
    assert len(model.module.downsample_layers) == 1
    assert len(model.module.upsample_layers) == 1
    y = model(torch.zeros(1, 8, 128, 128))
    assert tuple(y.shape) == (1, 4, 128, 128)


def test_ffno_bottleneck_profile_records_parameter_count_before_first_forward():
    from scripts.studies.pdebench_image128.models import build_model_from_profile, describe_model
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    profile = get_model_profile("ffno_bottleneck_base")
    model = build_model_from_profile(
        profile,
        in_channels=8,
        out_channels=4,
        spatial_shape=(128, 128),
    )

    description = describe_model(model, profile=profile)

    assert description["profile_id"] == "ffno_bottleneck_base"
    assert description["parameter_count"] > 0

    y = model(torch.zeros(1, 8, 128, 128))

    assert tuple(y.shape) == (1, 4, 128, 128)


def test_hybrid_resnet_skip_add_profile_only_changes_skip_config():
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    base = get_model_profile("hybrid_resnet_base").to_model_config()
    skip = get_model_profile("hybrid_resnet_skip_add").to_model_config()

    assert skip["hybrid_skip_connections"] is True
    assert skip["hybrid_skip_style"] == "add"
    assert {
        key: value
        for key, value in skip.items()
        if key not in {"profile_id", "hybrid_skip_connections", "hybrid_skip_style", "evidence_scope"}
    } == {
        key: value for key, value in base.items() if key not in {"profile_id", "evidence_scope"}
    }


def test_hybrid_resnet_cns_profile_promotes_skip_add_to_canonical_cns_variant():
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    skip = get_model_profile("hybrid_resnet_skip_add").to_model_config()
    cns = get_model_profile("hybrid_resnet_cns").to_model_config()

    assert cns["hybrid_skip_connections"] is True
    assert cns["hybrid_skip_style"] == "add"
    assert cns["hybrid_upsampler"] == "pixelshuffle"
    assert cns["evidence_scope"] == "benchmark_candidate"
    assert {
        key: value for key, value in cns.items() if key not in {"profile_id", "evidence_scope", "hybrid_upsampler"}
    } == {
        key: value for key, value in skip.items() if key not in {"profile_id", "evidence_scope"}
    }


def test_hybrid_resnet_skip_add_builds_two_decoder_skip_fusions_for_cns_shape():
    from scripts.studies.pdebench_image128.models import PadCropWrapper, build_model_from_profile, describe_model
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    model = build_model_from_profile(
        get_model_profile("hybrid_resnet_skip_add"),
        in_channels=8,
        out_channels=4,
        spatial_shape=(128, 128),
    )

    assert isinstance(model, PadCropWrapper)
    assert model.module.skip_connections is True
    assert model.module.hybrid_skip_style == "add"
    assert [plan["key"] for plan in model.module.skip_fusion_plan] == ["d2", "d1"]
    assert set(model.module.skip_fusion_projections.keys()) == {"d1", "d2"}
    description = describe_model(model, profile=get_model_profile("hybrid_resnet_skip_add"))
    assert description["parameter_count"] > 0

    y = model(torch.zeros(1, 8, 128, 128))

    assert tuple(y.shape) == (1, 4, 128, 128)


def test_hybrid_resnet_cns_preserves_canonical_two_stage_skip_add_shell():
    from scripts.studies.pdebench_image128.models import PadCropWrapper, build_model_from_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    model = build_model_from_profile(
        get_model_profile("hybrid_resnet_cns"),
        in_channels=8,
        out_channels=4,
        spatial_shape=(128, 128),
    )

    assert isinstance(model, PadCropWrapper)
    assert len(model.module.downsample_layers) == 2
    assert len(model.module.upsample_layers) == 2
    assert model.module.skip_connections is True
    assert model.module.hybrid_skip_style == "add"
    assert [plan["key"] for plan in model.module.skip_fusion_plan] == ["d2", "d1"]
    y = model(torch.zeros(1, 8, 128, 128))
    assert tuple(y.shape) == (1, 4, 128, 128)


def test_spectral_resnet_bottleneck_profile_builds_under_canonical_cns_skip_add_shell():
    from scripts.studies.pdebench_image128.models import PadCropWrapper, build_model_from_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    model = build_model_from_profile(
        get_model_profile("spectral_resnet_bottleneck_base"),
        in_channels=8,
        out_channels=4,
        spatial_shape=(128, 128),
    )

    assert isinstance(model, PadCropWrapper)
    assert model.module.skip_connections is True
    assert model.module.hybrid_skip_style == "add"
    assert [plan["key"] for plan in model.module.skip_fusion_plan] == ["d2", "d1"]
    assert set(model.module.skip_fusion_projections.keys()) == {"d1", "d2"}
    y = model(torch.zeros(1, 8, 128, 128))
    assert tuple(y.shape) == (1, 4, 128, 128)


def test_ffno_bottleneck_profile_builds_under_canonical_cns_skip_add_shell():
    from scripts.studies.pdebench_image128.models import PadCropWrapper, build_model_from_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    model = build_model_from_profile(
        get_model_profile("ffno_bottleneck_base"),
        in_channels=8,
        out_channels=4,
        spatial_shape=(128, 128),
    )

    assert isinstance(model, PadCropWrapper)
    assert model.module.skip_connections is True
    assert model.module.hybrid_skip_style == "add"
    assert [plan["key"] for plan in model.module.skip_fusion_plan] == ["d2", "d1"]
    assert set(model.module.skip_fusion_projections.keys()) == {"d1", "d2"}
    y = model(torch.zeros(1, 8, 128, 128))
    assert tuple(y.shape) == (1, 4, 128, 128)


def test_hybrid_resnet_accepts_cfd_cns_history_window_channels():
    from scripts.studies.pdebench_image128.models import build_model_from_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    model = build_model_from_profile(
        get_model_profile("hybrid_resnet_base"),
        in_channels=8,
        out_channels=4,
        spatial_shape=(128, 128),
    )

    y = model(torch.zeros(1, 8, 128, 128))

    assert tuple(y.shape) == (1, 4, 128, 128)


def test_hybrid_upsampler_study_profiles_build_and_preserve_shape():
    from scripts.studies.pdebench_image128.models import build_model_from_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    for profile_id in [
        "hybrid_resnet_base",
        "hybrid_resnet_interp_bilinear_conv",
        "hybrid_resnet_pixelshuffle",
    ]:
        model = build_model_from_profile(
            get_model_profile(profile_id),
            in_channels=8,
            out_channels=4,
            spatial_shape=(128, 128),
        )
        y = model(torch.zeros(1, 8, 128, 128))
        assert tuple(y.shape) == (1, 4, 128, 128)


def test_hybrid_upsampler_study_profiles_only_change_decoder_choice():
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    base = get_model_profile("hybrid_resnet_base").to_model_config()
    interp = get_model_profile("hybrid_resnet_interp_bilinear_conv").to_model_config()
    pixel = get_model_profile("hybrid_resnet_pixelshuffle").to_model_config()

    assert interp["base_model"] == "hybrid_resnet"
    assert pixel["base_model"] == "hybrid_resnet"
    assert interp["hybrid_upsampler"] == "interp_bilinear_conv"
    assert pixel["hybrid_upsampler"] == "pixelshuffle"
    assert {key: value for key, value in interp.items() if key not in {"profile_id", "hybrid_upsampler", "evidence_scope"}} == {
        key: value for key, value in base.items() if key not in {"profile_id", "evidence_scope"}
    }
    assert {key: value for key, value in pixel.items() if key not in {"profile_id", "hybrid_upsampler", "evidence_scope"}} == {
        key: value for key, value in base.items() if key not in {"profile_id", "evidence_scope"}
    }


def test_hybrid_cns_upsampler_study_profiles_build_and_preserve_shape():
    from scripts.studies.pdebench_image128.models import build_model_from_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    for profile_id in [
        "hybrid_resnet_cns_transpose",
        "hybrid_resnet_cns_interp_bilinear_conv",
        "hybrid_resnet_cns_pixelshuffle",
        "hybrid_resnet_cns",
    ]:
        model = build_model_from_profile(
            get_model_profile(profile_id),
            in_channels=8,
            out_channels=4,
            spatial_shape=(128, 128),
        )
        y = model(torch.zeros(1, 8, 128, 128))
        assert tuple(y.shape) == (1, 4, 128, 128)
        assert model.module.skip_connections is True
        assert model.module.hybrid_skip_style == "add"


def test_hybrid_cns_upsampler_study_profiles_only_change_decoder_choice():
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    base = get_model_profile("hybrid_resnet_cns_transpose").to_model_config()
    interp = get_model_profile("hybrid_resnet_cns_interp_bilinear_conv").to_model_config()
    pixel = get_model_profile("hybrid_resnet_cns_pixelshuffle").to_model_config()

    assert interp["base_model"] == "hybrid_resnet"
    assert pixel["base_model"] == "hybrid_resnet"
    assert interp["hybrid_skip_connections"] is True
    assert pixel["hybrid_skip_connections"] is True
    assert interp["hybrid_skip_style"] == "add"
    assert pixel["hybrid_skip_style"] == "add"
    assert interp["hybrid_upsampler"] == "interp_bilinear_conv"
    assert pixel["hybrid_upsampler"] == "pixelshuffle"
    assert {
        key: value
        for key, value in interp.items()
        if key not in {"profile_id", "hybrid_upsampler", "evidence_scope"}
    } == {
        key: value
        for key, value in base.items()
        if key not in {"profile_id", "hybrid_upsampler", "evidence_scope"}
    }
    assert {
        key: value
        for key, value in pixel.items()
        if key not in {"profile_id", "hybrid_upsampler", "evidence_scope"}
    } == {
        key: value
        for key, value in base.items()
        if key not in {"profile_id", "hybrid_upsampler", "evidence_scope"}
    }


def test_hybrid_resnet_cns_transpose_alias_is_manual_only_repro_of_prepromotion_decoder():
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    canonical = get_model_profile("hybrid_resnet_cns").to_model_config()
    transpose = get_model_profile("hybrid_resnet_cns_transpose").to_model_config()

    assert canonical["hybrid_upsampler"] == "pixelshuffle"
    assert transpose["hybrid_upsampler"] == "cyclegan_transpose"
    assert transpose["evidence_scope"] == "readiness-only"
    assert {
        key: value
        for key, value in transpose.items()
        if key not in {"profile_id", "hybrid_upsampler", "evidence_scope"}
    } == {
        key: value
        for key, value in canonical.items()
        if key not in {"profile_id", "hybrid_upsampler", "evidence_scope"}
    }


def test_spectral_resnet_bottleneck_profile_is_manual_only_and_not_in_primary_sets():
    from scripts.studies.pdebench_image128.run_config import (
        PRIMARY_CFD_CNS_PROFILE_IDS,
        PRIMARY_DARCY_PROFILE_IDS,
        READINESS_CFD_CNS_PROFILE_IDS,
        get_model_profile,
    )

    profile = get_model_profile("spectral_resnet_bottleneck_base")

    assert profile.base_model == "spectral_resnet_bottleneck_net"
    assert "spectral_resnet_bottleneck_base" not in PRIMARY_DARCY_PROFILE_IDS
    assert "spectral_resnet_bottleneck_base" not in PRIMARY_CFD_CNS_PROFILE_IDS
    assert "spectral_resnet_bottleneck_base" not in READINESS_CFD_CNS_PROFILE_IDS


def test_spectral_resnet_bottleneck_noshare_profile_is_manual_only_and_not_in_primary_sets():
    from scripts.studies.pdebench_image128.run_config import (
        PRIMARY_CFD_CNS_PROFILE_IDS,
        PRIMARY_DARCY_PROFILE_IDS,
        READINESS_CFD_CNS_PROFILE_IDS,
        get_model_profile,
    )

    profile = get_model_profile("spectral_resnet_bottleneck_noshare")

    assert profile.base_model == "spectral_resnet_bottleneck_net"
    assert "spectral_resnet_bottleneck_noshare" not in PRIMARY_DARCY_PROFILE_IDS
    assert "spectral_resnet_bottleneck_noshare" not in PRIMARY_CFD_CNS_PROFILE_IDS
    assert "spectral_resnet_bottleneck_noshare" not in READINESS_CFD_CNS_PROFILE_IDS


def test_ffno_bottleneck_profile_is_manual_only_and_not_in_primary_sets():
    from scripts.studies.pdebench_image128.run_config import (
        PRIMARY_CFD_CNS_PROFILE_IDS,
        PRIMARY_DARCY_PROFILE_IDS,
        READINESS_CFD_CNS_PROFILE_IDS,
        get_model_profile,
    )

    profile = get_model_profile("ffno_bottleneck_base")

    assert profile.base_model == "ffno_bottleneck_net"
    assert "ffno_bottleneck_base" not in PRIMARY_DARCY_PROFILE_IDS
    assert "ffno_bottleneck_base" not in PRIMARY_CFD_CNS_PROFILE_IDS
    assert "ffno_bottleneck_base" not in READINESS_CFD_CNS_PROFILE_IDS


def test_author_ffno_profile_is_manual_only_and_not_in_primary_sets():
    from scripts.studies.pdebench_image128.run_config import (
        PRIMARY_CFD_CNS_PROFILE_IDS,
        PRIMARY_DARCY_PROFILE_IDS,
        READINESS_CFD_CNS_PROFILE_IDS,
        get_model_profile,
    )

    profile = get_model_profile("author_ffno_cns_base")

    assert profile.base_model == "author_ffno_cns_net"
    assert "author_ffno_cns_base" not in PRIMARY_DARCY_PROFILE_IDS
    assert "author_ffno_cns_base" not in PRIMARY_CFD_CNS_PROFILE_IDS
    assert "author_ffno_cns_base" not in READINESS_CFD_CNS_PROFILE_IDS


def test_cfd_cns_default_profile_sets_use_canonical_skip_hybrid():
    from scripts.studies.pdebench_image128.run_config import (
        PRIMARY_CFD_CNS_PROFILE_IDS,
        PRIMARY_DARCY_PROFILE_IDS,
        READINESS_CFD_CNS_PROFILE_IDS,
    )

    assert PRIMARY_CFD_CNS_PROFILE_IDS == ["hybrid_resnet_cns", "fno_base", "unet_strong"]
    assert READINESS_CFD_CNS_PROFILE_IDS == ["hybrid_resnet_cns", "fno_base", "unet_tiny_smoke"]
    assert PRIMARY_DARCY_PROFILE_IDS == ["hybrid_resnet_base", "fno_base", "unet_strong"]


def test_hybrid_resnet_cns_builds_canonical_two_level_skip_shell():
    from scripts.studies.pdebench_image128.models import PadCropWrapper, build_model_from_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    model = build_model_from_profile(
        get_model_profile("hybrid_resnet_cns"),
        in_channels=8,
        out_channels=4,
        spatial_shape=(128, 128),
    )

    assert isinstance(model, PadCropWrapper)
    assert len(model.module.downsample_layers) == 2
    assert len(model.module.upsample_layers) == 2
    assert model.module.skip_connections is True
    assert model.module.hybrid_skip_style == "add"
    assert [plan["key"] for plan in model.module.skip_fusion_plan] == ["d2", "d1"]

    y = model(torch.zeros(1, 8, 128, 128))

    assert tuple(y.shape) == (1, 4, 128, 128)


def test_spectral_resnet_bottleneck_profile_uses_same_cns_skip_shell():
    from scripts.studies.pdebench_image128.models import PadCropWrapper, build_model_from_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    model = build_model_from_profile(
        get_model_profile("spectral_resnet_bottleneck_base"),
        in_channels=8,
        out_channels=4,
        spatial_shape=(128, 128),
    )

    assert isinstance(model, PadCropWrapper)
    assert len(model.module.downsample_layers) == 2
    assert len(model.module.upsample_layers) == 2
    assert model.module.skip_connections is True
    assert model.module.hybrid_skip_style == "add"
    assert [plan["key"] for plan in model.module.skip_fusion_plan] == ["d2", "d1"]

    y = model(torch.zeros(1, 8, 128, 128))

    assert tuple(y.shape) == (1, 4, 128, 128)


def test_spectral_noshare_profile_only_flips_weight_sharing():
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    shared = get_model_profile("spectral_resnet_bottleneck_base").to_model_config()
    noshare = get_model_profile("spectral_resnet_bottleneck_noshare").to_model_config()

    assert noshare["base_model"] == "spectral_resnet_bottleneck_net"
    assert shared["spectral_bottleneck_share_weights"] is True
    assert noshare["spectral_bottleneck_share_weights"] is False
    assert {
        key: value
        for key, value in noshare.items()
        if key not in {"profile_id", "spectral_bottleneck_share_weights", "evidence_scope"}
    } == {
        key: value
        for key, value in shared.items()
        if key not in {"profile_id", "spectral_bottleneck_share_weights", "evidence_scope"}
    }


def test_spectral_noshare_profile_builds_under_canonical_cns_shell():
    from scripts.studies.pdebench_image128.models import build_model_from_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    model = build_model_from_profile(
        get_model_profile("spectral_resnet_bottleneck_noshare"),
        in_channels=8,
        out_channels=4,
        spatial_shape=(128, 128),
    )

    y = model(torch.zeros(1, 8, 128, 128))

    assert tuple(y.shape) == (1, 4, 128, 128)


def test_cns_hybrid_spectral_ablation_profiles_pin_fixed_shell_and_vary_only_registered_axes():
    from scripts.studies.pdebench_image128.run_config import (
        PRIMARY_CFD_CNS_PROFILE_IDS,
        READINESS_CFD_CNS_PROFILE_IDS,
        get_model_profile,
    )

    profile_ids = [
        "spectral_resnet_bottleneck_base",
        "spectral_resnet_bottleneck_shared_blocks8",
        "spectral_resnet_bottleneck_shared_blocks10",
        "spectral_resnet_bottleneck_noshare",
        "spectral_resnet_bottleneck_noshare_blocks8",
        "spectral_resnet_bottleneck_noshare_blocks10",
    ]
    fixed_shell = {
        "base_model": "spectral_resnet_bottleneck_net",
        "hidden_channels": 32,
        "fno_modes": 12,
        "fno_blocks": 4,
        "hybrid_downsample_steps": 2,
        "hybrid_resnet_blocks": 6,
        "hybrid_skip_connections": True,
        "hybrid_skip_style": "add",
        "hybrid_upsampler": "pixelshuffle",
        "spectral_bottleneck_modes": 12,
        "spectral_bottleneck_gate_init": 0.1,
        "spectral_bottleneck_gate_mode": "shared",
    }

    configs = {profile_id: get_model_profile(profile_id).to_model_config() for profile_id in profile_ids}

    assert list(configs) == profile_ids
    for profile_id, config in configs.items():
        for key, value in fixed_shell.items():
            assert config[key] == value, (profile_id, key, config.get(key))
        assert profile_id not in PRIMARY_CFD_CNS_PROFILE_IDS
        assert profile_id not in READINESS_CFD_CNS_PROFILE_IDS

    for config in configs.values():
        varying = {
            "spectral_bottleneck_share_weights": config["spectral_bottleneck_share_weights"],
            "spectral_bottleneck_blocks": config["spectral_bottleneck_blocks"],
        }
        fixed = {
            key: value
            for key, value in config.items()
            if key not in {"profile_id", "evidence_scope", "spectral_bottleneck_share_weights", "spectral_bottleneck_blocks"}
        }
        assert varying["spectral_bottleneck_share_weights"] in {True, False}
        assert varying["spectral_bottleneck_blocks"] in {6, 8, 10}
        assert fixed == {
            key: value
            for key, value in configs["spectral_resnet_bottleneck_base"].items()
            if key not in {"profile_id", "evidence_scope", "spectral_bottleneck_share_weights", "spectral_bottleneck_blocks"}
        }


def test_ffno_bottleneck_profile_uses_same_cns_skip_shell():
    from scripts.studies.pdebench_image128.models import PadCropWrapper, build_model_from_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    model = build_model_from_profile(
        get_model_profile("ffno_bottleneck_base"),
        in_channels=8,
        out_channels=4,
        spatial_shape=(128, 128),
    )

    assert isinstance(model, PadCropWrapper)
    assert len(model.module.downsample_layers) == 2
    assert len(model.module.upsample_layers) == 2
    assert model.module.skip_connections is True
    assert model.module.hybrid_skip_style == "add"
    assert [plan["key"] for plan in model.module.skip_fusion_plan] == ["d2", "d1"]

    y = model(torch.zeros(1, 8, 128, 128))

    assert tuple(y.shape) == (1, 4, 128, 128)


def test_optional_bottleneck_profiles_stay_out_of_default_bundle_sets():
    from scripts.studies.pdebench_image128.run_config import (
        PRIMARY_CFD_CNS_PROFILE_IDS,
        PRIMARY_DARCY_PROFILE_IDS,
        READINESS_CFD_CNS_PROFILE_IDS,
    )

    for profile_id in [
        "spectral_resnet_bottleneck_base",
        "spectral_resnet_bottleneck_noshare",
        "ffno_bottleneck_base",
    ]:
        assert profile_id not in PRIMARY_DARCY_PROFILE_IDS
        assert profile_id not in PRIMARY_CFD_CNS_PROFILE_IDS
        assert profile_id not in READINESS_CFD_CNS_PROFILE_IDS


def test_unet_tiny_smoke_is_not_a_strong_baseline():
    from scripts.studies.pdebench_image128.models import assert_strong_baseline_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    tiny = get_model_profile("unet_tiny_smoke")
    strong = get_model_profile("unet_strong")

    assert_strong_baseline_profile(strong)
    try:
        assert_strong_baseline_profile(tiny)
    except ValueError as exc:
        assert "readiness-only" in str(exc)
    else:
        raise AssertionError("unet_tiny_smoke must not satisfy the strong-baseline gate")


def test_missing_fno_dependency_is_reported_as_blocker(monkeypatch):
    from scripts.studies.pdebench_image128 import models
    from scripts.studies.pdebench_image128.models import ModelBuildBlocker
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    def fail_import():
        raise ImportError("missing neuralop")

    monkeypatch.setattr(models, "_import_neuralop_fno", fail_import)
    profile = get_model_profile("fno_base")

    try:
        models.build_model_from_profile(profile, in_channels=1, out_channels=1, spatial_shape=(128, 128))
    except ModelBuildBlocker as exc:
        payload = exc.to_payload()
        assert payload["model"] == "fno_base"
        assert payload["reason"] == "model_dependency_unavailable"
        assert "neuralop" in payload["message"]
    else:
        raise AssertionError("missing neuralop should block FNO explicitly")


def test_gnot_builder_blocks_cleanly_when_source_root_missing(monkeypatch):
    from scripts.studies.pdebench_image128.models import ModelBuildBlocker, build_model_from_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    monkeypatch.setenv("GNOT_ROOT", "/definitely/missing/gnot")
    profile = get_model_profile("gnot_cns_base")

    try:
        build_model_from_profile(
            profile,
            in_channels=8,
            out_channels=4,
            spatial_shape=(128, 128),
            task_metadata={
                "task_id": "2d_cfd_cns",
                "dx": 1.0 / 128.0,
                "dy": 1.0 / 128.0,
                "dt": 0.05,
                "time_steps": 21,
                "history_len": 2,
                "field_order": ["density", "Vx", "Vy", "pressure"],
            },
        )
    except ModelBuildBlocker as exc:
        payload = exc.to_payload()
        assert payload["model"] == "gnot_cns_base"
        assert payload["reason"] == "model_dependency_unavailable"
        assert "GNOT" in payload["message"]
    else:
        raise AssertionError("missing GNOT source root should block the GNOT profile explicitly")


def test_author_ffno_builder_blocks_cleanly_when_source_root_missing(monkeypatch):
    from scripts.studies.pdebench_image128.models import ModelBuildBlocker, build_model_from_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    monkeypatch.setenv("AUTHOR_FFNO_ROOT", "/definitely/missing/fourierflow")
    profile = get_model_profile("author_ffno_cns_base")

    try:
        build_model_from_profile(
            profile,
            in_channels=8,
            out_channels=4,
            spatial_shape=(128, 128),
            task_metadata={
                "task_id": "2d_cfd_cns",
                "dx": 1.0 / 128.0,
                "dy": 1.0 / 128.0,
                "dt": 0.05,
                "time_steps": 21,
                "history_len": 2,
                "field_order": ["density", "Vx", "Vy", "pressure"],
            },
        )
    except ModelBuildBlocker as exc:
        payload = exc.to_payload()
        assert payload["model"] == "author_ffno_cns_base"
        assert payload["reason"] == "model_dependency_unavailable"
        assert "FFNO" in payload["message"]
    else:
        raise AssertionError("missing author FFNO source root should block the author FFNO profile explicitly")


def test_author_ffno_builder_blocks_cleanly_when_required_dependency_missing(monkeypatch):
    from scripts.studies.pdebench_image128 import author_ffno_adapter
    from scripts.studies.pdebench_image128.models import ModelBuildBlocker, build_model_from_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    def fail_load():
        raise author_ffno_adapter.AuthorFfnoAdapterBuildError(
            "model_dependency_unavailable",
            "manual author FFNO module load failed: missing einops",
        )

    monkeypatch.setattr(author_ffno_adapter, "load_author_ffno_dependencies", fail_load)
    profile = get_model_profile("author_ffno_cns_base")

    try:
        build_model_from_profile(
            profile,
            in_channels=8,
            out_channels=4,
            spatial_shape=(128, 128),
            task_metadata={
                "task_id": "2d_cfd_cns",
                "dx": 1.0 / 128.0,
                "dy": 1.0 / 128.0,
                "dt": 0.05,
                "time_steps": 21,
                "history_len": 2,
                "field_order": ["density", "Vx", "Vy", "pressure"],
            },
        )
    except ModelBuildBlocker as exc:
        payload = exc.to_payload()
        assert payload["model"] == "author_ffno_cns_base"
        assert payload["reason"] == "model_dependency_unavailable"
        assert "einops" in payload["message"]
    else:
        raise AssertionError("missing author FFNO dependencies should block the profile explicitly")


def test_author_ffno_profile_description_records_external_source_provenance(monkeypatch):
    from scripts.studies.pdebench_image128 import author_ffno_adapter
    from scripts.studies.pdebench_image128.models import build_model_from_profile, describe_model
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    class FakeWnLinear(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int, wnorm: bool = False):
            super().__init__()
            self.linear = torch.nn.Linear(in_features, out_features)

        def forward(self, x):
            return self.linear(x)

    class FakeFfno(torch.nn.Module):
        def __init__(
            self,
            modes: int,
            width: int,
            input_dim: int = 12,
            dropout: float = 0.0,
            in_dropout: float = 0.0,
            n_layers: int = 4,
            share_weight: bool = False,
            share_fork: bool = False,
            factor: int = 2,
            ff_weight_norm: bool = False,
            n_ff_layers: int = 2,
            gain: float = 1.0,
            layer_norm: bool = False,
            use_fork: bool = False,
            mode: str = "full",
        ):
            super().__init__()
            self.in_proj = torch.nn.Linear(input_dim, width)
            self.out = FakeWnLinear(width, 1)

        def forward(self, x, **kwargs):
            hidden = torch.relu(self.in_proj(x))
            return {"forecast": self.out(hidden)}

    monkeypatch.setattr(
        author_ffno_adapter,
        "load_author_ffno_dependencies",
        lambda: (
            FakeFfno,
            FakeWnLinear,
            {
                "external_repo": "https://github.com/alasdairtran/fourierflow",
                "external_commit": "deadbeef",
                "host_environment": {"conda_env": "ptycho311"},
            },
        ),
    )

    profile = get_model_profile("author_ffno_cns_base")
    model = build_model_from_profile(
        profile,
        in_channels=8,
        out_channels=4,
        spatial_shape=(128, 128),
        task_metadata={
            "task_id": "2d_cfd_cns",
            "dx": 1.0 / 128.0,
            "dy": 1.0 / 128.0,
            "dt": 0.05,
            "time_steps": 21,
            "history_len": 2,
            "field_order": ["density", "Vx", "Vy", "pressure"],
        },
    )
    description = describe_model(model, profile=profile)

    assert description["profile_id"] == "author_ffno_cns_base"
    assert description["external_source_provenance"]["external_repo"] == "https://github.com/alasdairtran/fourierflow"
    assert description["external_source_provenance"]["external_commit"] == "deadbeef"
    assert description["external_source_provenance"]["host_environment"]["conda_env"] == "ptycho311"
