"""Framework-neutral reconstruction policy identities.

The records in this module describe how existing reconstruction operations are
composed. They do not implement numerical assembly, calibration, or rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias


AssemblyAlgorithm: TypeAlias = Literal["legacy_position_v1", "barycentric_v1"]
PatchWeighting: TypeAlias = Literal["uniform", "probe"]
AssemblyWindow: TypeAlias = Literal["legacy_stitch_crop", "middle_trim"]
CalibrationMethod: TypeAlias = Literal["identity_v1", "varpro_s1s2_v1"]
OutputRepresentation: TypeAlias = Literal["amplitude_phase"]
FinalCrop: TypeAlias = Literal["none"]
GaugePresentation: TypeAlias = Literal["preserve_calibrated_canvas"]
DiagnosticsMode: TypeAlias = Literal["none"]
CompatibilityRoute: TypeAlias = Literal["uniform", "barycentric"]

RECONSTRUCTION_POLICY_VERSION = "reconstruction_policy_v1"

_ASSEMBLY_ALGORITHMS = {"legacy_position_v1", "barycentric_v1"}
_PATCH_WEIGHTINGS = {"uniform", "probe"}
_ASSEMBLY_WINDOWS = {"legacy_stitch_crop", "middle_trim"}
_CALIBRATION_METHODS = {"identity_v1", "varpro_s1s2_v1"}


@dataclass(frozen=True)
class AssemblySpec:
    """Placement, overlap weighting, and window contract for one assembly."""

    algorithm: AssemblyAlgorithm
    weighting: PatchWeighting
    window: AssemblyWindow

    def __post_init__(self) -> None:
        if self.algorithm not in _ASSEMBLY_ALGORITHMS:
            raise ValueError(f"Unsupported assembly algorithm: {self.algorithm!r}")
        if self.weighting not in _PATCH_WEIGHTINGS:
            raise ValueError(f"Unsupported patch weighting: {self.weighting!r}")
        if self.window not in _ASSEMBLY_WINDOWS:
            raise ValueError(f"Unsupported assembly window: {self.window!r}")
        if self.algorithm == "legacy_position_v1":
            if self.weighting != "uniform":
                raise ValueError(
                    "legacy_position_v1 assembly requires weighting='uniform'"
                )
            if self.window != "legacy_stitch_crop":
                raise ValueError(
                    "legacy_position_v1 assembly requires "
                    "window='legacy_stitch_crop'"
                )
        elif self.window != "middle_trim":
            raise ValueError(
                "barycentric_v1 assembly requires window='middle_trim'"
            )


@dataclass(frozen=True)
class CalibrationSpec:
    """Post-assembly scale-estimation contract."""

    method: CalibrationMethod

    def __post_init__(self) -> None:
        if self.method not in _CALIBRATION_METHODS:
            raise ValueError(f"Unsupported calibration method: {self.method!r}")


@dataclass(frozen=True)
class OutputSpec:
    """Presentation applied after assembly and optional calibration."""

    representation: OutputRepresentation = "amplitude_phase"
    final_crop: FinalCrop = "none"
    gauge_presentation: GaugePresentation = "preserve_calibrated_canvas"
    diagnostics: DiagnosticsMode = "none"

    def __post_init__(self) -> None:
        if self.representation != "amplitude_phase":
            raise ValueError(
                f"Unsupported output representation: {self.representation!r}"
            )
        if self.final_crop != "none":
            raise ValueError(f"Unsupported final crop: {self.final_crop!r}")
        if self.gauge_presentation != "preserve_calibrated_canvas":
            raise ValueError(
                "Unsupported gauge presentation: "
                f"{self.gauge_presentation!r}"
            )
        if self.diagnostics != "none":
            raise ValueError(f"Unsupported diagnostics mode: {self.diagnostics!r}")


@dataclass(frozen=True)
class ReconstructionPolicy:
    """Versioned composition of assembly, calibration, and output contracts."""

    assembly: AssemblySpec
    calibration: CalibrationSpec
    output: OutputSpec = OutputSpec()
    version: str = RECONSTRUCTION_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.version != RECONSTRUCTION_POLICY_VERSION:
            raise ValueError(f"Unsupported reconstruction policy: {self.version!r}")
        if (
            self.calibration.method == "varpro_s1s2_v1"
            and self.assembly.algorithm != "barycentric_v1"
        ):
            raise ValueError("VarPro calibration requires barycentric_v1 assembly")

    @property
    def identity(self) -> str:
        return "/".join(
            (
                self.version,
                self.assembly.algorithm,
                self.assembly.weighting,
                self.assembly.window,
                self.calibration.method,
                self.output.representation,
                self.output.final_crop,
                self.output.gauge_presentation,
                self.output.diagnostics,
            )
        )

    @property
    def compatibility_route(self) -> CompatibilityRoute:
        if self.assembly.algorithm == "legacy_position_v1":
            return "uniform"
        return "barycentric"


def resolve_cli_reconstruction_policy(
    patch_weighting: str,
    varpro_scaling: bool,
) -> ReconstructionPolicy:
    """Resolve the existing native CLI knobs into one explicit composition."""

    if patch_weighting not in _PATCH_WEIGHTINGS:
        raise ValueError(
            "patch_weighting must be 'uniform' or 'probe', got "
            f"{patch_weighting!r}"
        )

    calibration = CalibrationSpec(
        method="varpro_s1s2_v1" if bool(varpro_scaling) else "identity_v1"
    )
    if patch_weighting == "uniform" and not bool(varpro_scaling):
        assembly = AssemblySpec(
            algorithm="legacy_position_v1",
            weighting="uniform",
            window="legacy_stitch_crop",
        )
    else:
        assembly = AssemblySpec(
            algorithm="barycentric_v1",
            weighting=patch_weighting,
            window="middle_trim",
        )
    return ReconstructionPolicy(assembly=assembly, calibration=calibration)
