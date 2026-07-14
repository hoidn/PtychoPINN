"""Checked-spec parsing for grid-lines reference-performance qualification.

The spec (``scripts/studies/specs/grid_lines_reference_performance.toml``)
declares the shared procedural dataset recipe plus one ``[[references]]`` arm
per architecture family. Each arm binds the exact dictionary-loader runner
operands to a v4 integration-bridge contract. Fixture-sourced arms lock their
SSIM floors from the integration fixture now; frozen-artifact-sourced arms
stay fail-closed (``unlocked``) until Task 20b freezes the reference artifact
and pins its hash here. Which architecture each arm runs lives entirely in
the TOML — this module carries no architecture-specific branch.

Everything in this module is dry-run safe: no NPZ content is read, no Torch
or TensorFlow import happens, and no accelerator is touched.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .dataset_reference import (
    GridLinesReferenceRecipe,
    parse_grid_lines_reference_recipe,
)
from .datasets import DatasetError
from .runtime_errors import StudyRequestError
from .verdicts import (
    BRIDGE_DIFFERENCE_IDS,
    INTEGRATION_BRIDGE_CONTRACT_SCHEMA_VERSION,
    INTEGRATION_BRIDGE_EVIDENCE_SCHEMA_VERSION,
    IntegrationBridgeRequirement,
    VerdictInputError,
)

REFERENCE_SPEC_KIND = "grid_lines_reference_performance_v1"

#: Bridge command-operand fields that must equal the runner operands verbatim.
BRIDGE_COMMAND_FIELDS = (
    "architecture",
    "generator_output_mode",
    "hybrid_encoder_conv_hidden_scale",
    "training_patch_weighting",
    "physics_forward_mode",
    "amplitude_physics_gain",
    "torch_loss_mode",
    "seed",
    "epochs",
)
#: Bridge geometry fields pinned by the shared dataset recipe.
_BRIDGE_RECIPE_FIELDS = {
    "probe_target_n": "N",
    "raw_probe_archive_sha256": "probe_archive_sha256",
    "raw_probe_array_sha256": "raw_probe_array_sha256",
    "transformed_probe_sha256": "transformed_probe_sha256",
    "probe_smoothing_sigma": "probe_smoothing_sigma",
    "grid_lines_size": "size",
    "grid_lines_n": "N",
    "grid_lines_gridsize": "gridsize",
    "grid_lines_offset": "offset",
    "outer_offset_train": "outer_offset_train",
    "outer_offset_test": "outer_offset_test",
    "nimgs_train": "nimgs_train",
    "nimgs_test": "nimgs_test",
}
_RUNNER_PATH_FIELDS = frozenset(
    {"train_npz", "test_npz", "output_dir", "artifact_root"}
)
_REQUIRED_RUNNER_FIELDS = frozenset(
    {
        "architecture",
        "generator_output_mode",
        "N",
        "gridsize",
        "seed",
        "epochs",
        "torch_loss_mode",
        "training_patch_weighting",
        "physics_forward_mode",
        "amplitude_physics_gain",
        "hybrid_encoder_conv_hidden_scale",
        "position_crop_border",
        "enable_checkpointing",
    }
)
_DIFFERENCE_CLASSIFICATIONS = frozenset(
    {"harmless", "performance_relevant", "comparison_invalidating"}
)


@dataclass(frozen=True)
class ExpectedDifference:
    """Predeclared classification for one anticipated provenance difference."""

    field: str
    classification: str
    justification: str


@dataclass(frozen=True)
class ReferenceArm:
    """One reference-qualification arm bound to a v4 bridge contract."""

    id: str
    runner: Mapping[str, Any]
    bridge: Mapping[str, Any]
    expected_differences: Mapping[str, ExpectedDifference]
    frozen_reference_artifact: Path | None
    frozen_reference_artifact_declared: str | None

    @property
    def floor_source(self) -> str:
        return str(self.bridge["ssim_floor_source"])

    @property
    def floors_locked(self) -> bool:
        """True when the floors are usable now (fixture, or frozen + pinned)."""
        if self.floor_source == "fixture":
            return True
        return (
            self.bridge.get("ssim_floor_reference_artifact_sha256") is not None
            and self.frozen_reference_artifact is not None
            and self.frozen_reference_artifact.is_file()
        )


@dataclass(frozen=True)
class ReferenceQualificationSpec:
    """Parsed reference-qualification spec: shared recipe plus arms."""

    study_id: str
    spec_path: Path
    base_dir: Path
    recipe: GridLinesReferenceRecipe
    arms: tuple[ReferenceArm, ...]

    def arm(self, reference_id: str) -> ReferenceArm:
        for arm in self.arms:
            if arm.id == reference_id:
                return arm
        raise StudyRequestError(f"unknown reference arm {reference_id!r}")


def _closed_table(
    value: Any, *, path: str, allowed: set[str], required: set[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StudyRequestError(f"{path} must be a table")
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise StudyRequestError(f"{path} has unexpected keys {unexpected}")
    missing = sorted(required - set(value))
    if missing:
        raise StudyRequestError(f"{path} is missing required keys {missing}")
    return dict(value)


def _normalized_bridge(table: Mapping[str, Any], path: str) -> dict[str, Any]:
    if not isinstance(table, Mapping):
        raise StudyRequestError(f"{path} must be a table")
    bridge = dict(table)
    bridge.setdefault("probe_mask_diameter", None)
    bridge.setdefault("ssim_floor_reference_artifact_sha256", None)
    expected = set(IntegrationBridgeRequirement.__dataclass_fields__)
    if set(bridge) != expected:
        missing = sorted(expected - set(bridge))
        unexpected = sorted(set(bridge) - expected)
        raise StudyRequestError(
            f"{path} does not match the closed v4 bridge contract "
            f"(missing={missing}, unexpected={unexpected})"
        )
    return bridge


def _validate_runner(table: Mapping[str, Any], path: str) -> dict[str, Any]:
    from scripts.studies.grid_lines_torch_runner import TorchRunnerConfig

    if not isinstance(table, Mapping):
        raise StudyRequestError(f"{path} must be a table")
    runner = dict(table)
    known = set(TorchRunnerConfig.__dataclass_fields__)
    forbidden = sorted(set(runner) & _RUNNER_PATH_FIELDS)
    if forbidden:
        raise StudyRequestError(
            f"{path} must not declare runtime-owned path operands {forbidden}"
        )
    unknown = sorted(set(runner) - known)
    if unknown:
        raise StudyRequestError(
            f"{path} declares unknown TorchRunnerConfig operands {unknown}"
        )
    missing = sorted(_REQUIRED_RUNNER_FIELDS - set(runner))
    if missing:
        raise StudyRequestError(f"{path} is missing required operands {missing}")
    if runner["enable_checkpointing"] is not True:
        raise StudyRequestError(
            f"{path}.enable_checkpointing must be true: checkpoint identity is "
            "required reference evidence"
        )
    gain = runner["amplitude_physics_gain"]
    if (
        isinstance(gain, bool)
        or not isinstance(gain, (int, float))
        or not math.isfinite(float(gain))
        or float(gain) <= 0.0
    ):
        raise StudyRequestError(
            f"{path}.amplitude_physics_gain must be finite and positive"
        )
    return runner


def _validate_arm_consistency(
    arm_path: str,
    recipe: GridLinesReferenceRecipe,
    runner: Mapping[str, Any],
    bridge: Mapping[str, Any],
) -> None:
    for bridge_field, recipe_field in _BRIDGE_RECIPE_FIELDS.items():
        if bridge[bridge_field] != getattr(recipe, recipe_field):
            raise StudyRequestError(
                f"{arm_path}.bridge.{bridge_field} does not match the dataset "
                f"recipe {recipe_field}"
            )
    for field in BRIDGE_COMMAND_FIELDS:
        if bridge[field] != runner[field]:
            raise StudyRequestError(
                f"{arm_path}: bridge command operand {field!r} does not match "
                "the runner operand"
            )
    for name in ("N", "gridsize"):
        if runner[name] != getattr(recipe, name):
            raise StudyRequestError(
                f"{arm_path}.runner.{name} does not match the dataset recipe"
            )
    if bridge["generic_position_crop_border"] != runner["position_crop_border"]:
        raise StudyRequestError(
            f"{arm_path}: bridge generic_position_crop_border does not match "
            "runner position_crop_border"
        )
    crop_pairs = (
        ("historical_effective_patch_n", "historical_crop_border"),
        ("generic_effective_patch_n", "generic_position_crop_border"),
    )
    for patch_key, border_key in crop_pairs:
        if bridge[patch_key] != recipe.N - 2 * bridge[border_key]:
            raise StudyRequestError(
                f"{arm_path}.bridge.{patch_key} is inconsistent with "
                f"{border_key} at N={recipe.N}"
            )
    if bridge["loader_kind"] != "dictionary":
        raise StudyRequestError(
            f"{arm_path}.bridge.loader_kind must be 'dictionary': the mmap "
            "loader is a later ladder rung, not the reference"
        )
    if (
        bridge["physics_forward_mode"] == "amplitude"
        and float(bridge["amplitude_physics_gain"]) != 16.0
    ):
        raise StudyRequestError(
            f"{arm_path}: amplitude reference arms require "
            "amplitude_physics_gain exactly 16"
        )


def _parse_expected_differences(value: Any, path: str) -> dict[str, ExpectedDifference]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise StudyRequestError(f"{path} must be a table")
    differences: dict[str, ExpectedDifference] = {}
    for field, entry in value.items():
        if field not in BRIDGE_DIFFERENCE_IDS:
            raise StudyRequestError(
                f"{path}.{field} is not a classifiable bridge difference"
            )
        entry_table = _closed_table(
            entry,
            path=f"{path}.{field}",
            allowed={"classification", "justification"},
            required={"classification", "justification"},
        )
        classification = entry_table["classification"]
        justification = entry_table["justification"]
        if classification not in _DIFFERENCE_CLASSIFICATIONS:
            raise StudyRequestError(
                f"{path}.{field}.classification must be one of "
                + ", ".join(sorted(_DIFFERENCE_CLASSIFICATIONS))
            )
        if not isinstance(justification, str) or not justification:
            raise StudyRequestError(
                f"{path}.{field}.justification must be nonempty text"
            )
        differences[field] = ExpectedDifference(field, classification, justification)
    return differences


def _parse_reference_arm(
    index: int,
    value: Any,
    recipe: GridLinesReferenceRecipe,
    base_dir: Path,
) -> ReferenceArm:
    path = f"references[{index}]"
    table = _closed_table(
        value,
        path=path,
        allowed={
            "id",
            "runner",
            "bridge",
            "expected_differences",
            "frozen_reference_artifact",
        },
        required={"id", "runner", "bridge"},
    )
    arm_id = table["id"]
    if not isinstance(arm_id, str) or not arm_id:
        raise StudyRequestError(f"{path}.id must be nonempty text")
    runner = _validate_runner(table["runner"], f"{path}.runner")
    bridge = _normalized_bridge(table["bridge"], f"{path}.bridge")
    if bridge["bridge_id"] != arm_id:
        raise StudyRequestError(f"{path}.bridge.bridge_id must equal {arm_id!r}")
    declared_artifact = table.get("frozen_reference_artifact")
    if declared_artifact is not None and (
        not isinstance(declared_artifact, str) or not declared_artifact
    ):
        raise StudyRequestError(
            f"{path}.frozen_reference_artifact must be nonempty text"
        )
    floor_source = bridge.get("ssim_floor_source")
    if floor_source == "fixture":
        if declared_artifact is not None:
            raise StudyRequestError(
                f"{path}: fixture-sourced floors must not declare a frozen "
                "reference artifact"
            )
        try:
            IntegrationBridgeRequirement.from_mapping(bridge)
        except VerdictInputError as error:
            raise StudyRequestError(f"{path}.bridge is invalid: {error}") from error
    elif floor_source == "frozen_reference_artifact":
        if declared_artifact is None:
            raise StudyRequestError(
                f"{path}: frozen-source floors must declare "
                "frozen_reference_artifact (the path Task 20b freezes)"
            )
        if bridge["ssim_floor_reference_artifact_sha256"] is not None:
            try:
                IntegrationBridgeRequirement.from_mapping(bridge)
            except VerdictInputError as error:
                raise StudyRequestError(f"{path}.bridge is invalid: {error}") from error
    else:
        raise StudyRequestError(
            f"{path}.bridge.ssim_floor_source must be 'fixture' or "
            "'frozen_reference_artifact'"
        )
    _validate_arm_consistency(path, recipe, runner, bridge)
    return ReferenceArm(
        id=arm_id,
        runner=MappingProxyType(runner),
        bridge=MappingProxyType(bridge),
        expected_differences=MappingProxyType(
            _parse_expected_differences(
                table.get("expected_differences"), f"{path}.expected_differences"
            )
        ),
        frozen_reference_artifact=(
            None
            if declared_artifact is None
            else (Path(base_dir) / declared_artifact).resolve()
        ),
        frozen_reference_artifact_declared=declared_artifact,
    )


def load_reference_spec(
    path: str | Path, *, base_dir: str | Path | None = None
) -> ReferenceQualificationSpec:
    """Parse and cross-validate the checked reference-qualification spec."""
    spec_path = Path(path).resolve()
    resolved_base = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
    try:
        raw = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise StudyRequestError(
            f"cannot load reference spec {spec_path}: {exc}"
        ) from exc
    _closed_table(
        raw,
        path="spec",
        allowed={"schema", "study", "dataset", "references"},
        required={"schema", "study", "dataset", "references"},
    )
    schema = _closed_table(
        raw["schema"],
        path="schema",
        allowed={"kind", "version"},
        required={"kind", "version"},
    )
    if schema["kind"] != REFERENCE_SPEC_KIND or schema["version"] != 1:
        raise StudyRequestError(
            f"reference spec schema kind/version must be "
            f"{REFERENCE_SPEC_KIND!r}/1, got {schema['kind']!r}/{schema['version']!r}"
        )
    study = _closed_table(raw["study"], path="study", allowed={"id"}, required={"id"})
    study_id = study["id"]
    if not isinstance(study_id, str) or not study_id:
        raise StudyRequestError("study.id must be nonempty text")
    dataset = raw["dataset"]
    if not isinstance(dataset, Mapping) or "id" not in dataset:
        raise StudyRequestError("dataset must be a recipe table with an id")
    try:
        recipe = parse_grid_lines_reference_recipe(
            str(dataset["id"]), dataset, base_dir=resolved_base
        )
    except DatasetError as error:
        raise StudyRequestError(f"dataset recipe is invalid: {error}") from error
    references = raw["references"]
    if not isinstance(references, list) or not references:
        raise StudyRequestError("references must be a nonempty array of tables")
    arms = tuple(
        _parse_reference_arm(index, value, recipe, resolved_base)
        for index, value in enumerate(references)
    )
    ids = [arm.id for arm in arms]
    if len(set(ids)) != len(ids):
        raise StudyRequestError("references declare duplicate ids")
    return ReferenceQualificationSpec(
        study_id=study_id,
        spec_path=spec_path,
        base_dir=resolved_base,
        recipe=recipe,
        arms=arms,
    )


def render_reference_dry_run(spec: ReferenceQualificationSpec) -> str:
    """Render the qualification plan without NPZ, Torch, or accelerator access."""
    lines = [
        f"reference_qualification {spec.study_id}",
        f"spec {spec.spec_path}",
        f"dataset_recipe {spec.recipe.id} fingerprint={spec.recipe.fingerprint_sha256}",
        f"probe_archive {spec.recipe.probe_archive_declared} "
        f"sha256={spec.recipe.probe_archive_sha256}",
        f"condition N={spec.recipe.N} gridsize={spec.recipe.gridsize} "
        f"loader=dictionary",
        f"bridge_schema contract={INTEGRATION_BRIDGE_CONTRACT_SCHEMA_VERSION} "
        f"evidence={INTEGRATION_BRIDGE_EVIDENCE_SCHEMA_VERSION}",
    ]
    for arm in spec.arms:
        lines.append(
            f"reference {arm.id} architecture={arm.runner['architecture']} "
            f"seed={arm.runner['seed']} epochs={arm.runner['epochs']} "
            f"amplitude_physics_gain={arm.runner['amplitude_physics_gain']} "
            f"floor_source={arm.floor_source} "
            f"floors_locked={'true' if arm.floors_locked else 'false'}"
        )
        if arm.floor_source == "fixture":
            lines.append(
                f"  floors amp_ssim>={arm.bridge['fixture_amp_ssim_min']} "
                f"phase_ssim>={arm.bridge['fixture_phase_ssim_min']} "
                f"(mae guard <= {arm.bridge['fixture_amp_mae_max']}/"
                f"{arm.bridge['fixture_phase_mae_max']})"
            )
        else:
            lines.append(
                f"  floors locked_from={arm.frozen_reference_artifact_declared} "
                f"declared_sha256="
                f"{arm.bridge['ssim_floor_reference_artifact_sha256']}"
            )
    return "\n".join(lines)
