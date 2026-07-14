"""Checked-spec parsing for the one-variable configuration bridge ladder.

The spec (``scripts/studies/specs/grid_lines_bridge_ladder.toml``) walks from
the qualified grid-lines Hybrid ResNet reference (plan Task 20, rung 0) toward
the withdrawn study endpoint, changing exactly one configuration group per
rung (design gate 3, "One-variable ladder"). The single-group property is
machine-checked at load time: each rung's resolved config must differ from
its predecessor's in exactly the fields of its declared group, every declared
change must be effective, and consecutive rung dataset recipes may only step
in the recipe fields the group owns.

Everything in this module is dry-run safe: no NPZ content is read, no Torch
or TensorFlow import happens, and no accelerator is touched.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .dataset_reference import LadderDatasetRecipe, parse_ladder_dataset
from .datasets import DatasetError
from .runtime_errors import StudyRequestError
from .runtime_ladder_config import (
    CONFIG_FIELDS,
    INVARIANT_CONFIG_FIELDS,
    LADDER_DIFFERENCE_IDS,
    MUTABLE_CONFIG_FIELDS,
    LadderGate,
    closed_table,
    config_delta,
    parse_config,
    parse_ladder_expected_differences,
    required_text,
    validate_config_value,
    validate_dataset_step,
)
from .runtime_ladder_sections import (
    parse_baseline,
    parse_gate,
    parse_groups,
    parse_residuals,
)
from .runtime_ladder_spec_types import LadderBaseline, LadderResidual, LadderRung
from .runtime_reference_spec import ExpectedDifference

__all__ = [
    "CONFIG_FIELDS",
    "INVARIANT_CONFIG_FIELDS",
    "LADDER_DIFFERENCE_IDS",
    "LADDER_SPEC_KIND",
    "MUTABLE_CONFIG_FIELDS",
    "BridgeLadderSpec",
    "ExpectedDifference",
    "LadderBaseline",
    "LadderGate",
    "LadderResidual",
    "LadderRung",
    "config_delta",
    "load_ladder_spec",
    "render_ladder_dry_run",
]

LADDER_SPEC_KIND = "grid_lines_bridge_ladder_v1"
_SHA256_ALPHABET = set("0123456789abcdef")  # baseline evidence pin format


@dataclass(frozen=True)
class BridgeLadderSpec:
    """Parsed and cross-validated bridge-ladder spec."""

    study_id: str
    spec_path: Path
    spec_declared: str
    base_dir: Path
    gate: LadderGate
    baseline: LadderBaseline
    groups: Mapping[str, tuple[str, ...]]
    datasets: Mapping[str, LadderDatasetRecipe]
    rungs: tuple[LadderRung, ...]
    endpoint_config: Mapping[str, Any]
    residuals: tuple[LadderResidual, ...]

    def rung(self, rung_id: str) -> LadderRung:
        for rung in self.rungs:
            if rung.id == rung_id:
                return rung
        raise StudyRequestError(f"unknown ladder rung {rung_id!r}")

    def dataset(self, dataset_id: str) -> LadderDatasetRecipe:
        if dataset_id not in self.datasets:
            raise StudyRequestError(f"unknown ladder dataset {dataset_id!r}")
        return self.datasets[dataset_id]


def _validate_config_coherence(
    path: str, config: Mapping[str, Any], dataset: LadderDatasetRecipe
) -> None:
    loader_is_dictionary = config["loader"] == "dictionary"
    if loader_is_dictionary != (dataset.expression == "dictionary"):
        raise StudyRequestError(
            f"{path}: loader {config['loader']!r} is incoherent with dataset "
            f"expression {dataset.expression!r}"
        )
    if loader_is_dictionary and config["probe_normalize"] is not False:
        raise StudyRequestError(
            f"{path}: probe_normalize must be false with the dictionary "
            "loader — the ladder config declares EFFECTIVE semantics and the "
            "dictionary loader is legacy passthrough (Task 19 bridge)"
        )
    if loader_is_dictionary and config["mmap_bounds_filter"] != "off":
        raise StudyRequestError(
            f"{path}: mmap_bounds_filter must be 'off' with the dictionary "
            "loader — the dictionary path never bounds-filters scan positions"
        )
    # mmap_scale_convention is an inert ownership declaration under dictionary
    # loading. The closed enum still validates it, and the canonical baseline
    # carries dictionary_parity so its first mmap successor resolves unit scaling.
    if loader_is_dictionary and config["mmap_train_sampler"] != "sequential":
        raise StudyRequestError(
            f"{path}: mmap_train_sampler must stay at its default with the "
            "dictionary loader (the field selects mmap-loader behavior)"
        )
    count_domain = config["measurement_domain"] == "count_intensity"
    if count_domain != (dataset.expression == "generic_count_intensity"):
        raise StudyRequestError(
            f"{path}: measurement_domain {config['measurement_domain']!r} is "
            f"incoherent with dataset expression {dataset.expression!r}"
        )
    if config["N"] != dataset.recipe.N:
        raise StudyRequestError(
            f"{path}: config N={config['N']} does not match the dataset "
            f"recipe N={dataset.recipe.N}"
        )
    pair = (config["scale_contract_version"], config["measurement_domain"])
    if pair not in {
        ("legacy_v1", "normalized_amplitude"),
        ("ci_intensity_v2", "count_intensity"),
    }:
        raise StudyRequestError(
            f"{path}: scale contract pair {pair} is unsupported — the runtime "
            "(scaling_contract.resolve_scale_contract) validates version and "
            "measurement_domain as an inseparable pair, so they must step in "
            "the same rung"
        )


def _parse_rung(
    index: int,
    value: Any,
    *,
    groups: Mapping[str, tuple[str, ...]],
    datasets: Mapping[str, LadderDatasetRecipe],
    previous_config: Mapping[str, Any],
    previous_dataset: LadderDatasetRecipe,
    parsed_rungs: Mapping[str, LadderRung],
) -> LadderRung:
    path = f"rungs[{index}]"
    table = closed_table(
        value,
        path=path,
        allowed={
            "id",
            "group",
            "dataset",
            "changes",
            "expected_differences",
            "requires_scan_accounting",
            "requires_normalization_evidence",
            "requires_count_error_evidence",
            "diagnostic",
            "control_rung",
            "execution_status",
        },
        required={"id", "group", "dataset", "changes"},
    )
    rung_id = required_text(table["id"], f"{path}.id")
    diagnostic = bool(table.get("diagnostic", False))
    execution_status = required_text(
        table.get("execution_status", "runnable"), f"{path}.execution_status"
    )
    if execution_status not in {"runnable", "historical_only"}:
        raise StudyRequestError(
            f"{path}.execution_status must be 'runnable' or 'historical_only'"
        )
    if execution_status == "historical_only" and not diagnostic:
        raise StudyRequestError(
            f"{path}: historical_only execution status is valid only on "
            "diagnostic rungs"
        )
    control_rung = table.get("control_rung")
    if control_rung is not None:
        control_rung = required_text(control_rung, f"{path}.control_rung")
        if not diagnostic:
            raise StudyRequestError(
                f"{path}.control_rung is only valid on diagnostic rungs"
            )
        if control_rung not in parsed_rungs:
            raise StudyRequestError(
                f"{path}.control_rung {control_rung!r} must name an earlier "
                "rung (prior in the spec order)"
            )
        # The control is also the RESOLUTION BASE: the diagnostic's delta is
        # single-group against its control, which is what keeps 1e-style
        # rungs one-variable despite being two groups from the chain.
        previous_config = parsed_rungs[control_rung].resolved_config
        previous_dataset = datasets[parsed_rungs[control_rung].dataset]
    group = required_text(table["group"], f"{path}.group")
    if group not in groups:
        raise StudyRequestError(f"{path}.group {group!r} is not a declared group")
    dataset_id = required_text(table["dataset"], f"{path}.dataset")
    if dataset_id not in datasets:
        raise StudyRequestError(f"{path}.dataset {dataset_id!r} is undeclared")
    changes = table["changes"]
    if not isinstance(changes, Mapping) or not changes:
        raise StudyRequestError(f"{path}.changes must be a nonempty table")
    if set(changes) != set(groups[group]):
        raise StudyRequestError(
            f"{path}.changes must cover exactly the {group!r} group fields "
            f"{sorted(groups[group])}; got {sorted(changes)}"
        )
    for field, item in changes.items():
        validate_config_value(field, item, f"{path}.changes")
        if previous_config[field] == item:
            raise StudyRequestError(
                f"{path}.changes.{field} is a no-op: every declared change "
                "must be effective against the predecessor rung"
            )
    if changes.get("dataset", dataset_id) != dataset_id:
        raise StudyRequestError(
            f"{path}: changes.dataset must equal the rung dataset {dataset_id!r}"
        )
    resolved = dict(previous_config)
    resolved.update(changes)
    if resolved["dataset"] != dataset_id:
        raise StudyRequestError(
            f"{path}: resolved config dataset does not match {dataset_id!r}"
        )
    if (
        resolved["loader"] == "dictionary"
        and "mmap_scale_convention" in changes
    ):
        raise StudyRequestError(
            f"{path}.changes.mmap_scale_convention is inert with the dictionary "
            "loader; dictionary normalization ownership may be declared only "
            "by the baseline config"
        )
    dataset = datasets[dataset_id]
    validate_dataset_step(path, group, previous_dataset, dataset)
    _validate_config_coherence(path, resolved, dataset)
    flags = (
        "requires_scan_accounting",
        "requires_normalization_evidence",
        "requires_count_error_evidence",
        "diagnostic",
    )
    for flag in flags:
        if flag in table and type(table[flag]) is not bool:
            raise StudyRequestError(f"{path}.{flag} must be boolean")
    return LadderRung(
        id=rung_id,
        group=group,
        dataset=dataset_id,
        changes=MappingProxyType(dict(changes)),
        expected_differences=MappingProxyType(
            parse_ladder_expected_differences(
                table.get("expected_differences"), f"{path}.expected_differences"
            )
        ),
        requires_scan_accounting=bool(table.get("requires_scan_accounting", False)),
        requires_normalization_evidence=bool(
            table.get("requires_normalization_evidence", False)
        ),
        requires_count_error_evidence=bool(
            table.get("requires_count_error_evidence", False)
        ),
        execution_status=execution_status,
        diagnostic=diagnostic,
        control_rung=control_rung,
        resolved_config=MappingProxyType(resolved),
    )


def load_ladder_spec(
    path: str | Path, *, base_dir: str | Path | None = None
) -> BridgeLadderSpec:
    """Parse and cross-validate the checked bridge-ladder spec."""
    spec_path = Path(path).resolve()
    resolved_base = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
    try:
        spec_declared = spec_path.relative_to(resolved_base).as_posix()
    except ValueError:
        spec_declared = spec_path.name
    try:
        raw = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise StudyRequestError(f"cannot load ladder spec {spec_path}: {exc}") from exc
    sections = {
        "schema",
        "study",
        "gate",
        "baseline",
        "groups",
        "datasets",
        "rungs",
        "endpoint",
        "residuals",
    }
    closed_table(raw, path="spec", allowed=sections, required=sections)
    schema = closed_table(
        raw["schema"], path="schema", allowed={"kind", "version"},
        required={"kind", "version"},
    )
    if schema["kind"] != LADDER_SPEC_KIND or schema["version"] != 1:
        raise StudyRequestError(
            f"ladder spec schema kind/version must be {LADDER_SPEC_KIND!r}/1, "
            f"got {schema['kind']!r}/{schema['version']!r}"
        )
    study = closed_table(raw["study"], path="study", allowed={"id"}, required={"id"})
    gate = parse_gate(raw["gate"])
    baseline = parse_baseline(raw["baseline"], resolved_base)
    groups = parse_groups(raw["groups"])
    if not isinstance(raw["datasets"], Mapping) or not raw["datasets"]:
        raise StudyRequestError("datasets must be a nonempty table")
    datasets: dict[str, LadderDatasetRecipe] = {}
    for dataset_id, table in raw["datasets"].items():
        try:
            datasets[dataset_id] = parse_ladder_dataset(
                str(dataset_id), table, base_dir=resolved_base
            )
        except DatasetError as error:
            raise StudyRequestError(
                f"datasets.{dataset_id} is invalid: {error}"
            ) from error
    if baseline.dataset not in datasets:
        raise StudyRequestError(f"baseline.dataset {baseline.dataset!r} is undeclared")
    if baseline.config["dataset"] != baseline.dataset:
        raise StudyRequestError("baseline.config.dataset must match baseline.dataset")
    _validate_config_coherence("baseline", baseline.config, datasets[baseline.dataset])
    raw_rungs = raw["rungs"]
    if not isinstance(raw_rungs, list) or not raw_rungs:
        raise StudyRequestError("rungs must be a nonempty array of tables")
    rungs: list[LadderRung] = []
    parsed_rungs: dict[str, LadderRung] = {}
    previous_config: Mapping[str, Any] = baseline.config
    previous_dataset = datasets[baseline.dataset]
    for index, value in enumerate(raw_rungs):
        rung = _parse_rung(
            index,
            value,
            groups=groups,
            datasets=datasets,
            previous_config=previous_config,
            previous_dataset=previous_dataset,
            parsed_rungs=parsed_rungs,
        )
        parsed_rungs[rung.id] = rung
        rungs.append(rung)
        if not rung.diagnostic:  # diagnostics never propagate into the chain
            previous_config = rung.resolved_config
            previous_dataset = datasets[rung.dataset]
    ids = [rung.id for rung in rungs]
    if len(set(ids)) != len(ids) or baseline.id in ids:
        raise StudyRequestError("rungs declare duplicate ids")
    used_groups = [rung.group for rung in rungs if not rung.diagnostic]
    if len(set(used_groups)) != len(used_groups):
        raise StudyRequestError(
            "each configuration group may own at most one rung; pre-split "
            "bundled groups instead of repeating one"
        )
    endpoint = closed_table(
        raw["endpoint"], path="endpoint", allowed={"config"}, required={"config"})
    endpoint_config = parse_config(endpoint["config"], "endpoint.config")
    chain = [rung for rung in rungs if not rung.diagnostic]
    if not chain:
        raise StudyRequestError("the ladder needs a non-diagnostic rung")
    mismatch = config_delta(chain[-1].resolved_config, endpoint_config)
    if mismatch:
        raise StudyRequestError(
            "endpoint.config does not equal the final rung's resolved config; "
            f"mismatched fields {sorted(mismatch)}"
        )
    return BridgeLadderSpec(
        study_id=required_text(study["id"], "study.id"),
        spec_path=spec_path,
        spec_declared=spec_declared,
        base_dir=resolved_base,
        gate=gate,
        baseline=baseline,
        groups=MappingProxyType(groups),
        datasets=MappingProxyType(datasets),
        rungs=tuple(rungs),
        endpoint_config=MappingProxyType(endpoint_config),
        residuals=parse_residuals(raw["residuals"]),
    )


def render_ladder_dry_run(spec: BridgeLadderSpec) -> str:
    """Render the ladder plan without NPZ, Torch, or accelerator access."""
    gate = spec.gate
    if gate.policy == "absolute_ssim_delta_v1":
        gate_line = (
            f"gate {gate.policy} abs_amp_delta<={gate.max_abs_amp_ssim_delta} "
            f"abs_phase_delta<={gate.max_abs_phase_ssim_delta} "
            f"provenance={gate.threshold_provenance}"
        )
    else:
        gate_line = (
            f"gate {gate.policy} retained_amp>="
            f"{gate.retained_amp_ssim_min_fraction} retained_phase>="
            f"{gate.retained_phase_ssim_min_fraction} abs_amp_floor>="
            f"{gate.absolute_amp_ssim_floor} provenance={gate.threshold_provenance}"
        )
    lines = [
        f"bridge_ladder {spec.study_id}",
        f"spec {spec.spec_declared}",
        gate_line,
        (
            f"baseline {spec.baseline.id} reference={spec.baseline.reference_id} "
            f"status={spec.baseline.status} "
            f"evidence={spec.baseline.evidence_declared} "
            f"evidence_sha256={spec.baseline.evidence_sha256}"
        ),
    ]
    for rung in spec.rungs:
        dataset = spec.dataset(rung.dataset)
        changes = " ".join(
            f"{field}={rung.changes[field]!r}" for field in sorted(rung.changes)
        )
        lines.append(
            f"rung {rung.id} execution_status={rung.execution_status} "
            f"runnable={'true' if rung.runnable else 'false'} "
            f"group={rung.group} dataset={rung.dataset} "
            f"expression={dataset.expression} "
            f"recipe_fingerprint={dataset.fingerprint_sha256}"
        )
        lines.append(f"  changes {changes}")
    lines.append("residuals " + ", ".join(residual.id for residual in spec.residuals))
    return "\n".join(lines)
