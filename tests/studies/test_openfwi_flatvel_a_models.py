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


def test_official_inversionnet_probe_imports_existing_checkout_and_runs_forward(tmp_path):
    from scripts.studies.openfwi_flatvel_a.models import probe_official_inversionnet

    repo = tmp_path / "OpenFWI"
    repo.mkdir()
    (repo / "LICENSE").write_text("BSD-3-Clause\n", encoding="utf-8")
    (repo / "network.py").write_text(
        """
import torch
import torch.nn as nn

class InversionNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros((x.shape[0], 1, 70, 70), dtype=x.dtype, device=x.device)

model_dict = {"InversionNet": InversionNet}
""",
        encoding="utf-8",
    )

    result = probe_official_inversionnet(repo)

    assert result["status"] == "compatible"
    assert result["repo_path"] == str(repo.resolve())
    assert result["license_path"] == str((repo / "LICENSE").resolve())
    assert result["import_attempt"]["status"] == "imported"
    assert result["forward_pass"]["status"] == "succeeded"
    assert result["forward_pass"]["input_shape"] == [1, 5, 1000, 70]
    assert result["forward_pass"]["output_shape"] == [1, 1, 70, 70]
