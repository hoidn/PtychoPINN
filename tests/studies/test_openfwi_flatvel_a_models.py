import torch


def test_hybrid_resnet_smoke_builds_and_preserves_target_shape():
    from scripts.studies.openfwi_flatvel_a.models import build_model

    model = build_model(
        "hybrid_resnet_smoke",
        in_channels=5,
        out_channels=1,
        spatial_shape=(70, 70),
        profile_config={},
    )
    x = torch.randn(2, 5, 70, 70)

    y = model(x)

    assert y.shape == (2, 1, 70, 70)


def test_unet_smoke_runs_backward():
    from scripts.studies.openfwi_flatvel_a.models import build_model

    model = build_model(
        "unet_smoke",
        in_channels=5,
        out_channels=1,
        spatial_shape=(70, 70),
        profile_config={},
    )
    x = torch.randn(2, 5, 70, 70)
    target = torch.randn(2, 1, 70, 70)
    loss = torch.nn.functional.l1_loss(model(x), target)

    loss.backward()

    assert any(param.grad is not None for param in model.parameters())


def test_official_inversionnet_probe_blocks_without_repo(tmp_path):
    from scripts.studies.openfwi_flatvel_a.models import probe_official_inversionnet

    blocker = probe_official_inversionnet(tmp_path / "missing")

    assert blocker["status"] == "blocked"
    assert blocker["reason"] == "official_repo_missing"
