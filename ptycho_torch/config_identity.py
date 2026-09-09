"""Schema-aware comparison of sealed scientific configurations.

Sealed campaign records (row seals, arm identities, launch gates) store the
resolved configuration of the code that ran them. Later backward-compatible
schema additions (a new model field with a default, a new model-spec version
that only adds such fields) must not read as scientific drift, while any
non-default value for such a field must. These helpers project a current
composition onto the sealed schema exactly, and raise on real drift.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ptycho_torch.config_params import ModelConfig
from ptycho_torch.model_spec import (
    MODEL_SPEC_V3_MODEL_FIELDS,
    MODEL_SPEC_V3_VERSION,
    MODEL_SPEC_V4_MODEL_FIELDS,
    MODEL_SPEC_V4_VERSION,
)


def project_model_fields(expected: Any, recorded: Any) -> Any:
    """Drop default-valued model fields that a sealed record predates."""
    if not isinstance(expected, Mapping) or not isinstance(recorded, Mapping):
        return expected
    defaults = ModelConfig()
    projected = dict(expected)
    for name in set(projected) - set(recorded):
        if not hasattr(defaults, name) or projected[name] != getattr(defaults, name):
            raise ValueError(
                f"model field {name!r} is not at its default; the sealed record "
                "cannot represent it"
            )
        projected.pop(name)
    return projected


def project_model_spec(spec: Mapping[str, Any], schema_version: str) -> dict[str, Any]:
    """Project a current model-spec payload onto the schema a seal was written with."""
    projected = {key: (dict(value) if isinstance(value, Mapping) else value) for key, value in spec.items()}
    current = projected.get("schema_version")
    if current == schema_version:
        return projected
    if current != MODEL_SPEC_V4_VERSION or schema_version != MODEL_SPEC_V3_VERSION:
        raise ValueError(
            f"cannot project model spec {current!r} onto sealed schema {schema_version!r}"
        )
    defaults = ModelConfig()
    fields = dict(projected["model_config"])
    for name in set(MODEL_SPEC_V4_MODEL_FIELDS) - set(MODEL_SPEC_V3_MODEL_FIELDS):
        if fields.get(name) != getattr(defaults, name):
            raise ValueError(
                f"model spec field {name!r} is not at its default; sealed "
                f"{schema_version!r} identity cannot represent it"
            )
        fields.pop(name)
    projected["model_config"] = fields
    projected["schema_version"] = schema_version
    return projected


def find_model_spec_schema(value: Any) -> str | None:
    """Return the first ``model_spec.schema_version`` found in a nested record."""
    if isinstance(value, Mapping):
        spec = value.get("model_spec")
        if isinstance(spec, Mapping) and isinstance(spec.get("schema_version"), str):
            return spec["schema_version"]
        for item in value.values():
            found = find_model_spec_schema(item)
            if found is not None:
                return found
    return None


__all__ = ["find_model_spec_schema", "project_model_fields", "project_model_spec"]
