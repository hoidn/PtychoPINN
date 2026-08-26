"""Rename-elimination contract: one name per torch config quantity.

Wave A renamed ``DataConfig.K`` -> ``neighbor_count`` and
``TrainingConfig.n_groups`` -> ``training_groups``, and dropped
``DataConfig.groups_per_center`` from the persisted wire + dataclass. The
2026-08-24 centered-nearest convergence (torch-artifact-v5) then retired the
K-choose-C oversampling policy family and the pre-centered planner/API names.

These tests pin the post-rename spelling at the dataclass, resolver, and
wire-encode boundaries so a stale alias cannot silently reappear, and add a
static removal ratchet over production code, current config/examples, and the
normative doc set so the retired grouping policy cannot be re-advertised.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import fields
from pathlib import Path

import pytest

from ptycho_torch.config_params import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.config_resolution import (
    normalize_inference_patch,
    normalize_training_patch,
)
from ptycho_torch.model_spec import derive_model_spec

REPO_ROOT = Path(__file__).resolve().parents[2]


def _field_names(config_type) -> set[str]:
    return {item.name for item in fields(config_type)}


def test_dataclass_fields_use_new_spellings():
    data_names = _field_names(DataConfig)
    assert "neighbor_count" in data_names
    assert "K" not in data_names
    assert "groups_per_center" not in data_names

    training_names = _field_names(TrainingConfig)
    assert "training_groups" in training_names
    assert "n_groups" not in training_names


def test_training_patch_accepts_neighbor_count_rejects_K():
    normalized = normalize_training_patch({"neighbor_count": 7})
    assert normalized.values["neighbor_count"] == 7

    with pytest.raises(ValueError, match="unknown training input field"):
        normalize_training_patch({"K": 7})


def test_training_patch_accepts_training_groups_rejects_n_groups():
    normalized = normalize_training_patch({"training_groups": 4})
    assert normalized.values["training_groups"] == 4

    with pytest.raises(ValueError, match="unknown training input field"):
        normalize_training_patch({"n_groups": 4})


def test_unknown_training_field_suggests_nearest_canonical_name():
    with pytest.raises(ValueError) as exc_info:
        normalize_training_patch(
            {"training_groups": 4, "zzz": 1, "archtecture": "ffno"}
        )

    message = str(exc_info.value)
    assert "unknown training input field(s): archtecture, zzz" in message
    assert "archtecture: architecture" in message


def test_inference_patch_accepts_new_spellings():
    normalized = normalize_inference_patch({"neighbor_count": 3, "inference_groups": 2})
    assert normalized.values["neighbor_count"] == 3
    assert normalized.values["inference_groups"] == 2

    # Legacy spelling is a permanently fenced alias (H2): it must normalize
    # to the canonical key, not survive verbatim.
    legacy = normalize_inference_patch({"training_groups": 2})
    assert legacy.values["inference_groups"] == 2
    assert "training_groups" not in legacy.values

    with pytest.raises(ValueError, match="unknown inference input field"):
        normalize_inference_patch({"K": 3})


def test_encode_artifact_identity_emits_v5_spellings():
    from ptycho_torch.artifact_schema import (
        CURRENT_ARTIFACT_SCHEMA_VERSION,
        encode_artifact_identity,
    )
    from ptycho_torch.config_bridge import to_model_config

    data = DataConfig(N=64, gridsize=1, probe_scale=4.0)
    model = ModelConfig(object_big=False, probe_big=False, probe_mask=False)
    training = TrainingConfig(device="cpu", torch_loss_mode="poisson")
    inference = InferenceConfig()
    spec = derive_model_spec(to_model_config(data, model), model, data)
    payload = encode_artifact_identity(spec, data, training, inference)
    assert payload["schema_version"] == CURRENT_ARTIFACT_SCHEMA_VERSION
    assert "neighbor_count" in payload["data_config"]
    assert "K" not in payload["data_config"]
    assert "groups_per_center" not in payload["data_config"]
    assert "training_groups" in payload["training_config"]
    assert "n_groups" not in payload["training_config"]
    # The centered-nearest era carries the live canvas name and no retired
    # K-choose-C policy fields on the data wire.
    assert "group_padding_step" in payload["data_config"]
    assert "neighbor_pool_size" not in payload["data_config"]
    assert "max_neighbor_distance" not in payload["data_config"]


# ---------------------------------------------------------------------------
# Static removal ratchet (Task 7 of the 2026-08-24 centered-nearest grouping
# convergence). Scans production code, current config/examples, and the
# normative doc set for retired runtime identifiers. Allowed residual
# occurrences are limited to:
#   * frozen artifact-era field declarations and upgrade code
#     (ptycho_torch/artifact_eras.py, scripts/migrate_legacy_bundle.py);
#   * explicit migration diagnostics (ptycho/config/resolution.py,
#     scripts/studies/ablation/configuration.py).
# Historical completed plan/design bodies are not scanned here.
# ---------------------------------------------------------------------------

RETIRED_GROUPING_IDENTIFIERS = (
    # Retired configuration fields (Step-5 grep scope of the task brief).
    "enable_oversampling",
    "neighbor_pool_size",
    "neighbor_function",
    "K_quadrant",
    "min_neighbor_distance",
    "max_neighbor_distance",
    "scan_pattern",
    "require_complete_group_coverage",
    # Retired grouping-policy planner/API names.
    "plan_sample_then_group",
    "plan_scan_centered",
    "group_coords",
    "get_neighbor_indices",
    "center_scan_id_available",
)

# Production files that may legitimately name the retired policy. Each entry
# exists for one of the two allowed reasons: frozen artifact-era declarations
# + upgrade code, or explicit migration diagnostics. The allowlist may only
# shrink as those purposes are retired.
_ALLOWLISTED_PRODUCTION_FILES = frozenset(
    {
        # Explicit migration diagnostics for retired grouping field names.
        "ptycho/config/resolution.py",
        # Frozen v1-v4 artifact-era field declarations + pre-centered upgrade
        # code (C1-only legacy projection).
        "ptycho_torch/artifact_eras.py",
        # Offline bundle migrator: retired-field drop set for v5 re-seals.
        "scripts/migrate_legacy_bundle.py",
        # REJECTED_PATH_REASONS migration diagnostics for the ablation study.
        "scripts/studies/ablation/configuration.py",
    }
)

# Normative docs / current config / example files scanned for the retired
# identifiers. After the centered-nearest convergence these must contain no
# residual identifiers; any deliberate quote of a migration diagnostic must be
# allowlisted here by filename with the exact identifier it may keep.
_ALLOWED_NORMATIVE_FILES: dict[str, frozenset[str]] = {
    # Deliberate migration-diagnostic quotes introduced by the 2026-08-24
    # centered-nearest convergence doc pass. Each names the retired flag only
    # to state that it is removed/retired; drop the pair when the quoted
    # diagnostic itself is retired.
    "docs/COMMANDS_REFERENCE.md": frozenset(
        {"enable_oversampling", "neighbor_pool_size"}
    ),
    "docs/GRIDSIZE_N_GROUPS_GUIDE.md": frozenset(
        {"enable_oversampling", "neighbor_pool_size"}
    ),
    "docs/debugging/TROUBLESHOOTING.md": frozenset(
        {"enable_oversampling", "neighbor_pool_size"}
    ),
    "examples/sampling/README.md": frozenset(
        {"enable_oversampling", "neighbor_pool_size"}
    ),
    "examples/sampling/migration_from_legacy.sh": frozenset(
        {"enable_oversampling", "neighbor_pool_size"}
    ),
}


def _tracked_files(*prefixes: str) -> list[Path]:
    """Existing tracked files under the given repo-relative prefixes."""
    completed = subprocess.run(
        ["git", "ls-files", "--", *prefixes],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(f"git ls-files failed: {completed.stderr.strip()}")
    paths: list[Path] = []
    for name in completed.stdout.splitlines():
        path = REPO_ROOT / name
        if path.is_file():
            paths.append(path)
    return paths


def _production_files() -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    for path in _tracked_files("ptycho", "ptycho_torch", "scripts"):
        if path.suffix != ".py":
            continue
        files.append((path.relative_to(REPO_ROOT).as_posix(), path.read_text()))
    return files


def _normative_files() -> list[tuple[str, str]]:
    files: list[tuple[str, str]] = []
    for path in _tracked_files(
        "docs/specs",
        "specs/ptychodus_api_spec.md",
        "docs/CONFIGURATION.md",
        "docs/COMMANDS_REFERENCE.md",
        "docs/GRIDSIZE_N_GROUPS_GUIDE.md",
        "docs/DATA_GENERATION_GUIDE.md",
        "docs/DATA_MANAGEMENT_GUIDE.md",
        "docs/DEVELOPER_GUIDE.md",
        "docs/architecture.md",
        "docs/architecture_torch.md",
        "docs/debugging",
        "docs/migration",
        "docs/workflows",
        "scripts/simulation/README.md",
        "examples/sampling",
        "studies/object_family_repro/conf/config.yaml",
    ):
        if path.is_dir() or path.suffix not in (".md", ".sh", ".yaml"):
            continue
        files.append((path.relative_to(REPO_ROOT).as_posix(), path.read_text()))
    return files


def test_retired_grouping_identifiers_absent_outside_allowlist():
    failures: list[str] = []
    for rel, text in _production_files():
        if rel in _ALLOWLISTED_PRODUCTION_FILES:
            continue
        for identifier in RETIRED_GROUPING_IDENTIFIERS:
            if identifier in text:
                failures.append(f"{rel}: retired identifier {identifier!r}")
    for rel, text in _normative_files():
        allowed = _ALLOWED_NORMATIVE_FILES.get(rel, frozenset())
        for identifier in RETIRED_GROUPING_IDENTIFIERS:
            if identifier in text and identifier not in allowed:
                failures.append(f"{rel}: retired identifier {identifier!r}")
    assert not failures, (
        "retired grouping identifiers found outside the allowlist:\n"
        + "\n".join(failures)
    )


def test_group_padding_step_is_the_sole_live_canvas_name():
    data_names = _field_names(DataConfig)
    assert "group_padding_step" in data_names
    assert "max_neighbor_distance" not in data_names

    production_hits: list[str] = []
    for rel, text in _production_files():
        if rel in _ALLOWLISTED_PRODUCTION_FILES:
            continue
        if "max_neighbor_distance" in text:
            production_hits.append(rel)
    assert not production_hits, (
        "retired canvas name max_neighbor_distance still used in production: "
        + ", ".join(production_hits)
    )

    doc_hits: list[str] = []
    for rel, text in _normative_files():
        if "max_neighbor_distance" in text:
            doc_hits.append(rel)
    assert not doc_hits, (
        "retired canvas name max_neighbor_distance still advertised in "
        "normative docs/config/examples: " + ", ".join(doc_hits)
    )


def test_one_planner_definition_and_two_materializer_call_sites():
    definitions: list[str] = []
    call_sites: list[str] = []
    for rel, text in _production_files():
        for line in text.splitlines():
            if "plan_nearest_groups" not in line:
                continue
            if re.search(r"^\s*def plan_nearest_groups\s*\(", line):
                definitions.append(rel)
            elif "plan_nearest_groups(" in line:
                call_sites.append(rel)
    assert definitions == ["ptycho/grouping.py"], (
        "exactly one plan_nearest_groups definition expected in "
        f"ptycho/grouping.py, found {definitions}"
    )
    assert call_sites == [
        "ptycho/raw_data.py",
        "ptycho_torch/dataloader.py",
    ], (
        "exactly the two intended materializer call sites expected "
        f"(RawData RAM + PtychoDataset mmap), found {call_sites}"
    )

    grouping_source = (REPO_ROOT / "ptycho/grouping.py").read_text()
    assert 'CENTERED_NEAREST_GROUPING_CONTRACT = "centered-nearest-v1"' in (
        grouping_source
    ), "the centered-nearest-v1 contract marker must stay pinned in ptycho/grouping.py"
