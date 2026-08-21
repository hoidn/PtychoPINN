"""Era x load-path matrix: every supported era loads, unsupported eras fail loudly."""
from __future__ import annotations

import pytest

from tests.torch.era_fixtures import build_bundle

SUPPORTED_ERAS = ["portable_v1", "portable_v2_json", "portable_v3"]


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
