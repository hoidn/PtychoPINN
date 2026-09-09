"""Schema-aware sealed-configuration comparison."""
from __future__ import annotations

import pytest


def test_projection_drops_only_default_valued_later_fields():
    from ptycho_torch.config_identity import (
        find_model_spec_schema,
        project_model_fields,
        project_model_spec,
    )
    from ptycho_torch.config_params import ModelConfig
    from ptycho_torch.model_spec import (
        MODEL_SPEC_V3_MODEL_FIELDS,
        MODEL_SPEC_V3_VERSION,
        MODEL_SPEC_V4_MODEL_FIELDS,
        MODEL_SPEC_V4_VERSION,
    )

    defaults = ModelConfig()
    added = set(MODEL_SPEC_V4_MODEL_FIELDS) - set(MODEL_SPEC_V3_MODEL_FIELDS)
    assert added
    fields = {name: getattr(defaults, name, None) for name in MODEL_SPEC_V4_MODEL_FIELDS}
    spec = {"schema_version": MODEL_SPEC_V4_VERSION, "model_config": fields, "parity_scale_mode": "off"}

    projected = project_model_spec(spec, MODEL_SPEC_V3_VERSION)
    assert projected["schema_version"] == MODEL_SPEC_V3_VERSION
    assert set(projected["model_config"]) == set(MODEL_SPEC_V4_MODEL_FIELDS) - added
    assert project_model_spec(spec, MODEL_SPEC_V4_VERSION) == spec
    with pytest.raises(ValueError, match="not at its default"):
        project_model_spec(
            {**spec, "model_config": {**fields, "vit_depth": 99}},
            MODEL_SPEC_V3_VERSION,
        )

    recorded = {"N": 128, "amp_activation": "sigmoid"}
    expected = {**recorded, "vit_depth": defaults.vit_depth}
    assert project_model_fields(expected, recorded) == recorded
    with pytest.raises(ValueError, match="not at its default"):
        project_model_fields({**expected, "vit_depth": 99}, recorded)
    assert project_model_fields(None, recorded) is None
    assert find_model_spec_schema({"a": {"model_spec": spec}}) == MODEL_SPEC_V4_VERSION
    assert find_model_spec_schema({"a": 1}) is None
