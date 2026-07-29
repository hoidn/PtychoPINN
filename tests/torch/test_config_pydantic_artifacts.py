"""Frozen pre-Pydantic internal Torch configuration wire artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch


FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "config"
PINNED_REVISION = "f762bd27bccca3f9dfe9ecfad500af9589cb7777"
FIXTURE_NAMES = (
    "pydantic_pre_migration_torch_artifact_v1.json",
    "pydantic_pre_migration_torch_artifact_v2.json",
    "pydantic_pre_migration_torch_tensor_mask.json",
)


def _read_payload(name: str):
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _canonical_bytes(payload) -> bytes:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (serialized + "\n").encode("utf-8")


def test_pre_migration_fixture_inventory_is_complete():
    expected = {
        *FIXTURE_NAMES,
        "README.md",
        "generate_pre_migration_fixtures.py",
    }

    assert FIXTURE_ROOT.is_dir()
    assert expected <= {
        path.name for path in FIXTURE_ROOT.iterdir() if path.is_file()
    }


def test_generator_reproduces_each_canonical_fixture_byte_stream():
    generator = FIXTURE_ROOT / "generate_pre_migration_fixtures.py"

    result = subprocess.run(
        [sys.executable, str(generator), "--stdout"],
        cwd=Path(__file__).parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    generated = json.loads(result.stdout)

    assert tuple(sorted(generated)) == tuple(sorted(FIXTURE_NAMES))
    for name in FIXTURE_NAMES:
        expected = _canonical_bytes(generated[name])
        assert (FIXTURE_ROOT / name).read_bytes() == expected


def test_internal_v1_is_an_exact_projection_and_decodes():
    from ptycho_torch.artifact_schema import (
        ARTIFACT_SCHEMA_V1_VERSION,
        ARTIFACT_V1_DATA_FIELDS,
        ARTIFACT_V1_INFERENCE_FIELDS,
        ARTIFACT_V1_TRAINING_FIELDS,
        decode_artifact_identity,
        from_json_payload,
    )
    from ptycho_torch.model_spec import (
        MODEL_SPEC_V1_MODEL_FIELDS,
        MODEL_SPEC_V1_VERSION,
    )

    raw = _read_payload("pydantic_pre_migration_torch_artifact_v1.json")
    model_spec = raw["model_spec"]

    assert raw["backend"] == "pytorch"
    assert ARTIFACT_SCHEMA_V1_VERSION == "torch-artifact-v1"
    assert MODEL_SPEC_V1_VERSION == "torch-model-spec-v1"
    assert raw["schema_version"] == ARTIFACT_SCHEMA_V1_VERSION
    assert model_spec["schema_version"] == MODEL_SPEC_V1_VERSION
    assert tuple(model_spec["model_config"]) == tuple(
        sorted(MODEL_SPEC_V1_MODEL_FIELDS)
    )
    assert tuple(raw["data_config"]) == tuple(sorted(ARTIFACT_V1_DATA_FIELDS))
    assert tuple(raw["training_config"]) == tuple(
        sorted(ARTIFACT_V1_TRAINING_FIELDS)
    )
    assert tuple(raw["inference_config"]) == tuple(
        sorted(ARTIFACT_V1_INFERENCE_FIELDS)
    )

    decoded = decode_artifact_identity(from_json_payload(raw))
    model = decoded.model_spec.to_model_config()

    assert decoded.model_spec.schema_version == "torch-model-spec-v2"
    assert model.mode == "Supervised"
    assert model.architecture == "ffno"
    assert model.object_big is True
    assert model.object_layout == "grouped_patches"
    assert model.training_canvas == "relative_overlap"
    assert model.training_patch_weighting == "probe"
    assert decoded.data_config.neighbor_function == "4_quadrant"
    assert decoded.data_config.scan_pattern == "Rectangular"
    assert decoded.training_config.scheduler == "WarmupCosine"
    assert decoded.training_config.optimizer == "adamw"
    assert decoded.inference_config.patch_weighting == "uniform"
    assert decoded.ci_statistics == {
        "rms_input_scale": [0.125, 0.25],
        "rms_probe_scale": [2.0],
    }


def test_internal_v2_is_current_and_decodes_to_the_same_identity_as_v1():
    from ptycho_torch.artifact_schema import (
        CURRENT_ARTIFACT_SCHEMA_VERSION,
        decode_artifact_identity,
        from_json_payload,
    )
    from ptycho_torch.model_spec import (
        CURRENT_MODEL_SPEC_VERSION,
        MODEL_SPEC_V2_MODEL_FIELDS,
    )

    raw_v1 = _read_payload("pydantic_pre_migration_torch_artifact_v1.json")
    raw_v2 = _read_payload("pydantic_pre_migration_torch_artifact_v2.json")

    assert CURRENT_ARTIFACT_SCHEMA_VERSION == "torch-artifact-v2"
    assert CURRENT_MODEL_SPEC_VERSION == "torch-model-spec-v2"
    assert raw_v2["schema_version"] == CURRENT_ARTIFACT_SCHEMA_VERSION
    assert raw_v2["model_spec"]["schema_version"] == CURRENT_MODEL_SPEC_VERSION
    assert set(raw_v2["model_spec"]["model_config"]) == set(
        MODEL_SPEC_V2_MODEL_FIELDS
    )
    assert "object_big" not in raw_v2["model_spec"]["model_config"]
    assert raw_v2["model_spec"]["model_config"]["object_layout"] == (
        "grouped_patches"
    )

    decoded_v1 = decode_artifact_identity(from_json_payload(raw_v1))
    decoded_v2 = decode_artifact_identity(from_json_payload(raw_v2))

    assert decoded_v2.model_spec.to_model_config() == (
        decoded_v1.model_spec.to_model_config()
    )
    assert decoded_v2.data_config == decoded_v1.data_config
    assert decoded_v2.training_config == decoded_v1.training_config
    assert decoded_v2.inference_config == decoded_v1.inference_config
    assert decoded_v2.ci_statistics == decoded_v1.ci_statistics


def test_complex_tensor_mask_tag_decodes_exactly_and_is_defensively_copied():
    from ptycho_torch.artifact_schema import from_json_payload
    from ptycho_torch.model_spec import MODEL_SPEC_V2_MODEL_FIELDS, ModelSpec

    raw = _read_payload("pydantic_pre_migration_torch_tensor_mask.json")
    tagged = raw["model_spec"]["model_config"]["probe_mask_tensor"]

    assert raw["field_name"] == "probe_mask_tensor"
    assert raw["model_spec_fields"] == sorted(MODEL_SPEC_V2_MODEL_FIELDS)
    assert tagged == {
        "__ptychopinn_torch_tensor__": True,
        "data": {
            "imag": [2.0, 0.5, -4.0, -6.5, 8.125, 0.0],
            "real": [1.0, -3.0, 0.0, 5.25, -7.75, 9.0],
        },
        "dtype": "complex64",
        "shape": [2, 3],
    }

    decoded_payload = from_json_payload(raw)
    decoded_tensor = decoded_payload["model_spec"]["model_config"][
        "probe_mask_tensor"
    ]
    expected = torch.tensor(
        [
            [1.0 + 2.0j, -3.0 + 0.5j, -4.0j],
            [5.25 - 6.5j, -7.75 + 8.125j, 9.0 + 0.0j],
        ],
        dtype=torch.complex64,
    )
    assert decoded_tensor.dtype == torch.complex64
    assert tuple(decoded_tensor.shape) == (2, 3)
    torch.testing.assert_close(decoded_tensor, expected, rtol=0.0, atol=0.0)

    spec = ModelSpec.from_payload(decoded_payload["model_spec"])
    decoded_tensor.zero_()
    first = spec.to_model_config().probe_mask_tensor
    second = spec.to_model_config().probe_mask_tensor

    torch.testing.assert_close(first, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(second, expected, rtol=0.0, atol=0.0)
    assert first.data_ptr() != second.data_ptr()
    first.add_(100.0 + 100.0j)
    torch.testing.assert_close(
        spec.to_model_config().probe_mask_tensor,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_readme_pins_revision_reproduction_command_and_fixture_hashes():
    readme = (FIXTURE_ROOT / "README.md").read_text(encoding="utf-8")

    assert PINNED_REVISION in readme
    assert (
        "python tests/fixtures/config/generate_pre_migration_fixtures.py "
        "--stdout"
    ) in readme
    for name in FIXTURE_NAMES:
        digest = hashlib.sha256((FIXTURE_ROOT / name).read_bytes()).hexdigest()
        assert f"| `{name}` | `{digest}` |" in readme
