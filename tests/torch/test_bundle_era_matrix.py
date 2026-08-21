"""Era x load-path matrix: every supported era loads, unsupported eras fail loudly."""
from __future__ import annotations

import pytest

from tests.torch.era_fixtures import build_bundle

SUPPORTED_ERAS = ["portable_v1", "portable_v2_json", "portable_v3", "portable_v4"]


@pytest.mark.parametrize("era", SUPPORTED_ERAS)
def test_supported_era_loads_through_strict_path(tmp_path, era):
    from ptycho_torch.workflows.bundle_io import load_inference_bundle_torch

    bundle = build_bundle(tmp_path, era)
    models, params = load_inference_bundle_torch(bundle)
    assert set(models) >= {"autoencoder", "diffraction_to_obj"}
    assert params["artifact_schema_version"].startswith("torch-artifact-portable-")


def test_dill_era_fails_loudly_naming_the_migration_script(tmp_path):
    from ptycho_torch.workflows.bundle_io import load_inference_bundle_torch

    bundle = build_bundle(tmp_path, "dill_era")
    with pytest.raises(Exception, match="migrate_legacy_bundle"):
        load_inference_bundle_torch(bundle)


@pytest.mark.parametrize("era", ["portable_v2_json", "portable_v3"])
def test_migration_door_lands_v4_from_historical_era(tmp_path, era):
    from ptycho_torch.artifact_schema import (
        CURRENT_ARTIFACT_SCHEMA_VERSION,
    )
    from ptycho_torch.migrate_bundle import migrate_bundle
    from ptycho_torch.workflows.bundle_io import load_inference_bundle_torch

    source = build_bundle(tmp_path / "src", era)
    out_zip = migrate_bundle(source, tmp_path / "out")

    models, params = load_inference_bundle_torch(out_zip.parent)
    assert set(models) >= {"autoencoder", "diffraction_to_obj"}
    assert params["artifact_schema_version"] == CURRENT_ARTIFACT_SCHEMA_VERSION
