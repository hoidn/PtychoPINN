"""CLI helper to inspect IntensityScaler weights inside scenario checkpoints.

Loads the TensorFlow archive saved by the scenario runner, extracts the trained
IntensityScaler / IntensityScaler_inv gain (exp(log_scale)), compares it to the
recorded ``intensity_scale`` parameter, and writes JSON + Markdown summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import dill

from ptycho import params
from ptycho.model import _get_log_scale  # pylint: disable=protected-access
from ptycho.model_manager import ModelManager


@dataclass
class InspectionResult:
    scenario: str
    scenario_dir: Path
    model_archive: Path
    recorded_params: Dict[str, Any]
    log_scale: float
    layer_scale: float

    def to_json(self) -> Dict[str, Any]:
        recorded_scale = self.recorded_params.get("intensity_scale")
        delta = (
            None
            if recorded_scale is None
            else self.layer_scale - float(recorded_scale)
        )
        ratio = (
            None
            if recorded_scale in (None, 0)
            else self.layer_scale / float(recorded_scale)
        )

        intensity_scaler_gain = 1.0 / self.layer_scale if self.layer_scale else None

        return {
            "scenario": self.scenario,
            "scenario_dir": str(self.scenario_dir),
            "model_archive": str(self.model_archive),
            "log_scale": self.log_scale,
            "exp_log_scale": self.layer_scale,
            "recorded_params": self.recorded_params,
            "intensity_scaler": {
                "operation": "x / exp(log_scale)",
                "gain": intensity_scaler_gain,
                "bias": 0.0,
            },
            "intensity_scaler_inv": {
                "operation": "exp(log_scale) * x",
                "gain": self.layer_scale,
                "bias": 0.0,
            },
            "diff_vs_recorded": {
                "absolute_delta": delta,
                "ratio": ratio,
            },
        }

    def to_markdown(self) -> str:
        payload = self.to_json()
        recorded_scale = payload["recorded_params"].get("intensity_scale")
        delta = payload["diff_vs_recorded"]["absolute_delta"]
        ratio = payload["diff_vs_recorded"]["ratio"]

        recorded_lines = "\n".join(
            f"- **{key}**: {value}"
            for key, value in payload["recorded_params"].items()
        )

        ratio_line = (
            f"{ratio:.6f}" if isinstance(ratio, (float, int)) else "n/a"
        )
        delta_line = (
            f"{delta:+.6f}" if isinstance(delta, (float, int)) else "n/a"
        )

        divider = "-" * 80
        return textwrap.dedent(
            f"""
            # Intensity Scaler Inspection — {self.scenario}

            * Model archive: `{self.model_archive}`
            * Recorded `intensity_scale` (params.dill): {recorded_scale}
            * Layer exp(log_scale): {self.layer_scale:.6f}
            * Δ(layer - recorded): {delta_line}
            * Ratio (layer / recorded): {ratio_line}

            ## Recorded params snapshot
            {recorded_lines or '(no params found)'}

            ## Layer gains
            - IntensityScaler divides by **{1.0 / self.layer_scale:.6f}**
            - IntensityScaler_inv multiplies by **{self.layer_scale:.6f}**
            {divider}
            """
        ).strip()


def load_recorded_params(archive_path: Path) -> Dict[str, Any]:
    with zipfile.ZipFile(archive_path, "r") as zf:
        candidate = next(
            (
                name
                for name in zf.namelist()
                if name.endswith("autoencoder/params.dill")
            ),
            None,
        )
        if candidate is None:
            raise FileNotFoundError(
                f"No params.dill found inside {archive_path}"
            )
        with zf.open(candidate) as fh:
            return dill.load(fh)


def infer_archive_path(scenario_dir: Path) -> Path:
    candidate = scenario_dir / "train_outputs" / "wts.h5.zip"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Expected checkpoint archive at {candidate}, found nothing"
        )
    return candidate


def inspect_scenario(
    scenario_dir: Path, scenario_name: Optional[str] = None
) -> InspectionResult:
    archive_path = infer_archive_path(scenario_dir)
    recorded_params = load_recorded_params(archive_path)

    # Load the checkpoint via ModelManager so IntensityScaler weights populate.
    base_path = archive_path.with_suffix("")  # drop only the .zip suffix
    ModelManager.load_multiple_models(str(base_path))

    log_scale = float(_get_log_scale().numpy())
    layer_scale = math.exp(log_scale)
    params_snapshot = {
        key: recorded_params.get(key)
        for key in (
            "intensity_scale",
            "intensity_scale.trainable",
            "nphotons",
            "N",
            "gridsize",
        )
        if key in recorded_params
    }

    return InspectionResult(
        scenario=scenario_name or scenario_dir.name,
        scenario_dir=scenario_dir,
        model_archive=archive_path,
        recorded_params=params_snapshot,
        log_scale=log_scale,
        layer_scale=layer_scale,
    )


def write_outputs(result: InspectionResult, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_json(), f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(result.to_markdown() + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect IntensityScaler weights for a scenario checkpoint."
    )
    parser.add_argument(
        "--scenario-dir",
        required=True,
        type=Path,
        help="Path to the scenario run directory (e.g., .../gs1_ideal)",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        type=Path,
        help="Path to write the JSON report",
    )
    parser.add_argument(
        "--output-markdown",
        required=True,
        type=Path,
        help="Path to write the Markdown summary",
    )
    parser.add_argument(
        "--scenario-name",
        required=False,
        help="Override scenario name in outputs (defaults to directory name)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = inspect_scenario(args.scenario_dir, args.scenario_name)
    write_outputs(result, args.output_json, args.output_markdown)
    print(
        f"Scenario {result.scenario}: exp(log_scale)={result.layer_scale:.6f}, "
        f"recorded intensity_scale={result.recorded_params.get('intensity_scale')}"
    )


if __name__ == "__main__":
    main()
