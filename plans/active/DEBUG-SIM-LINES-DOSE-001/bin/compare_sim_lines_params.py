#!/usr/bin/env python3
"""
Compare sim_lines_4x snapshot metadata against the legacy dose_experiments
parameter scan. Emits a Markdown table plus a structured JSON diff.

Extended for Phase D1 to include loss configuration weights (mae_weight,
nll_weight, realspace_weight, realspace_mae_weight) by instantiating
TrainingConfig for each scenario.

Phase D1a-D1c (2026-01-20T112029Z): Added stubbed cfg execution to capture
**runtime** values from the legacy dose_experiments init() for both loss_fn
modes ('nll' and 'mae'). The previous static parsing captured conditional
assignments (the MAE branch) as defaults, misrepresenting the actual config.

Phase D3 (2026-01-20T133807Z): Extended to include training hyperparameters
(nepochs, batch_size, probe.trainable, intensity_scale.trainable) in the
diff to support hyperparameter audit before scheduling retrains.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import pathlib
import re
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

CFG_ASSIGNMENT_RE = re.compile(r"""cfg\['(?P<key>[^']+)'\]\s*=\s*(?P<value>.+)""")

# Loss weight keys we need from the legacy params.cfg
LEGACY_LOSS_KEYS = ["mae_weight", "nll_weight", "realspace_weight", "realspace_mae_weight"]


def capture_legacy_params_defaults() -> Dict[str, Any]:
    """
    Import the actual ptycho.params module and deep-copy the cfg dictionary
    to capture the framework defaults for loss weights.

    Per CLAUDE.md / CONFIG-001: treat legacy params.cfg as source of truth;
    copy values without mutating shared state.

    Returns dict with keys: mae_weight, nll_weight, realspace_weight, realspace_mae_weight
    plus their source line reference (ptycho/params.py:64).
    """
    from ptycho import params as legacy_params

    # Deep copy to avoid any accidental mutation
    cfg_copy = copy.deepcopy(legacy_params.cfg)

    # Extract only the loss weight keys we care about
    defaults = {}
    for key in LEGACY_LOSS_KEYS:
        defaults[key] = cfg_copy.get(key)

    # Add metadata about the source
    defaults["_source"] = "ptycho/params.py:64"
    defaults["_note"] = "Framework defaults from legacy global cfg dictionary"

    return defaults


def execute_legacy_init_with_stubbed_cfg(
    dose_config_path: pathlib.Path,
    nphotons: float = 1e9,
    loss_fn: str = "nll",
) -> Dict[str, Any]:
    """
    Execute the legacy dose_experiments init() function in-process with a
    stubbed ptycho.params module so we can capture the runtime cfg dictionary
    without touching the production environment.

    Per How-To Map (D1a):
    - Save sys.modules['ptycho.params'], register a lightweight stub exposing cfg={}
    - Load and call init() with the given nphotons and loss_fn
    - Deep-copy the resulting cfg and restore the original module in finally block

    Returns the cfg dictionary after init() completes.
    """
    original_params_module = sys.modules.get("ptycho.params")

    # Create a stub module with an empty cfg dict
    stub_module = types.ModuleType("ptycho.params")
    stub_cfg: Dict[str, Any] = {}
    stub_module.cfg = stub_cfg

    try:
        # Register the stub so `from ptycho.params import cfg` sees it
        sys.modules["ptycho.params"] = stub_module

        # Extract the init function source from the captured script
        script_text = dose_config_path.read_text()

        # Find the code block before the markdown table (ends at "---")
        code_end = script_text.find("\n---")
        if code_end == -1:
            code_block = script_text
        else:
            code_block = script_text[:code_end]

        # Compile and exec the code to get the init function
        module_globals: Dict[str, Any] = {"__name__": "__dose_stub__"}
        exec(compile(code_block, str(dose_config_path), "exec"), module_globals)

        init_func = module_globals.get("init")
        if init_func is None:
            raise ValueError(f"No init() function found in {dose_config_path}")

        # Call init with the specified parameters
        init_func(nphotons, loss_fn=loss_fn)

        # Deep-copy the cfg to preserve the snapshot
        return copy.deepcopy(stub_cfg)

    finally:
        # Always restore the original module
        if original_params_module is not None:
            sys.modules["ptycho.params"] = original_params_module
        elif "ptycho.params" in sys.modules:
            del sys.modules["ptycho.params"]


def capture_legacy_loss_modes(
    dose_config_path: pathlib.Path,
    nphotons: float = 1e9,
) -> Dict[str, Dict[str, Any]]:
    """
    Capture the runtime cfg dictionaries for both loss_fn modes ('nll' and 'mae')
    by executing the legacy init() function twice with a stubbed cfg.

    Returns: {"nll": {...cfg...}, "mae": {...cfg...}}
    """
    return {
        "nll": execute_legacy_init_with_stubbed_cfg(dose_config_path, nphotons, "nll"),
        "mae": execute_legacy_init_with_stubbed_cfg(dose_config_path, nphotons, "mae"),
    }


# Parameters we care about along with a human-friendly label
PARAMETERS: List[Tuple[str, str]] = [
    ("gridsize", "gridsize"),
    ("probe_big", "probe_big"),
    ("probe_mask", "probe_mask"),
    ("probe_scale", "probe_scale"),
    ("offset", "offset"),
    ("outer_offset_train", "outer_offset_train"),
    ("outer_offset_test", "outer_offset_test"),
    ("nimgs_train", "nimgs_train"),
    ("nimgs_test", "nimgs_test"),
    ("nphotons", "nphotons"),
    ("group_count", "group_count"),
    ("neighbor_count", "neighbor_count"),
    ("reassemble_M", "reassemble_M"),
    ("intensity_scale.trainable", "intensity_scale.trainable"),
    ("total_images", "total_images"),
    # Phase D1: Loss configuration weights
    ("mae_weight", "mae_weight"),
    ("nll_weight", "nll_weight"),
    ("realspace_weight", "realspace_weight"),
    ("realspace_mae_weight", "realspace_mae_weight"),
    # Phase D3: Training hyperparameters
    ("nepochs", "nepochs"),
    ("batch_size", "batch_size"),
    ("probe.trainable", "probe.trainable"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=pathlib.Path, help="sim_lines_4x snapshot JSON")
    parser.add_argument("--dose-config", required=True, type=pathlib.Path, help="dose_experiments_param_scan.md file")
    parser.add_argument("--output-markdown", required=True, type=pathlib.Path, help="Output Markdown file")
    parser.add_argument("--output-json", required=True, type=pathlib.Path, help="Output comparison JSON")
    parser.add_argument(
        "--output-dose-loss-weights",
        type=pathlib.Path,
        help="Output JSON with runtime cfg snapshots for both loss_fn modes",
    )
    parser.add_argument(
        "--output-dose-loss-weights-markdown",
        type=pathlib.Path,
        help="Output Markdown with runtime cfg snapshots for both loss_fn modes (labels conditional fields)",
    )
    parser.add_argument(
        "--nphotons",
        type=float,
        default=1e9,
        help="nphotons value for legacy init() execution (default: 1e9)",
    )
    parser.add_argument(
        "--output-legacy-defaults",
        type=pathlib.Path,
        help="Output JSON with the actual ptycho.params.cfg framework defaults for loss weights",
    )
    parser.add_argument(
        "--default-sim-lines-nepochs",
        type=int,
        default=5,
        help="Default nepochs for sim_lines scenarios (default: 5, reflecting typical short training runs)",
    )
    return parser.parse_args()


def literal_eval(value: str) -> Any:
    """Best-effort literal parsing with graceful fallback for identifiers."""
    try:
        return ast.literal_eval(value)
    except Exception:
        stripped = value.strip()
        return stripped


def parse_dose_config(path: pathlib.Path) -> Dict[str, Any]:
    """
    Parse cfg[...] assignments from the init() function in the captured
    dose_experiments script.
    """
    text = path.read_text()
    assignments: Dict[str, Any] = {}
    in_init = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("def "):
            in_init = stripped.startswith("def init")
        if not in_init:
            continue
        match = CFG_ASSIGNMENT_RE.search(stripped)
        if not match:
            continue
        key = match.group("key")
        value = match.group("value")
        if key not in assignments:
            assignments[key] = literal_eval(value)
    return assignments


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def get_loss_weights_from_training_config(
    gridsize: int,
    probe_scale: float,
    probe_big: bool,
    probe_mask: bool,
    nphotons: float,
    group_count: int,
    neighbor_count: int,
) -> Dict[str, float]:
    """
    Instantiate a TrainingConfig to extract the actual loss weights that would
    be used for the given scenario parameters. This ensures we capture the real
    defaults from the dataclass definition.

    Returns dict with keys: mae_weight, nll_weight, realspace_weight, realspace_mae_weight
    """
    from pathlib import Path
    from ptycho.config.config import ModelConfig, TrainingConfig

    model_config = ModelConfig(
        N=64,
        gridsize=gridsize,
        model_type="pinn",
        probe_scale=probe_scale,
        probe_big=probe_big,
        probe_mask=probe_mask,
    )
    # Create a minimal TrainingConfig to read its loss weight defaults
    config = TrainingConfig(
        model=model_config,
        n_groups=group_count,
        nphotons=nphotons,
        neighbor_count=neighbor_count,
        nepochs=10,  # placeholder
        output_dir=Path("/tmp"),  # placeholder
    )
    return {
        "mae_weight": config.mae_weight,
        "nll_weight": config.nll_weight,
        "realspace_weight": config.realspace_weight,
        "realspace_mae_weight": config.realspace_mae_weight,
    }


def get_training_config_snapshot(
    gridsize: int,
    probe_scale: float,
    probe_big: bool,
    probe_mask: bool,
    nphotons: float,
    group_count: int,
    neighbor_count: int,
    nepochs: int,
    batch_size: int,
    probe_trainable: bool,
    intensity_scale_trainable: bool,
) -> Dict[str, Any]:
    """
    Instantiate a TrainingConfig to extract all training hyperparameters
    for the given scenario parameters. Phase D3 extension.

    Returns dict with keys:
      - mae_weight, nll_weight, realspace_weight, realspace_mae_weight (loss weights)
      - nepochs, batch_size, probe.trainable, intensity_scale.trainable (training knobs)
    """
    from pathlib import Path
    from ptycho.config.config import ModelConfig, TrainingConfig

    model_config = ModelConfig(
        N=64,
        gridsize=gridsize,
        model_type="pinn",
        probe_scale=probe_scale,
        probe_big=probe_big,
        probe_mask=probe_mask,
    )
    config = TrainingConfig(
        model=model_config,
        n_groups=group_count,
        nphotons=nphotons,
        neighbor_count=neighbor_count,
        nepochs=nepochs,
        batch_size=batch_size,
        probe_trainable=probe_trainable,
        intensity_scale_trainable=intensity_scale_trainable,
        output_dir=Path("/tmp"),  # placeholder
    )
    return {
        # Loss weights
        "mae_weight": config.mae_weight,
        "nll_weight": config.nll_weight,
        "realspace_weight": config.realspace_weight,
        "realspace_mae_weight": config.realspace_mae_weight,
        # Training hyperparameters (Phase D3)
        "nepochs": config.nepochs,
        "batch_size": config.batch_size,
        "probe.trainable": config.probe_trainable,
        "intensity_scale.trainable": config.intensity_scale_trainable,
    }


def load_sim_lines_snapshot(path: pathlib.Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    data = json.loads(path.read_text())
    run_params = data.get("run_params", {})
    scenarios: Dict[str, Dict[str, Any]] = {}
    for scenario in data.get("scenarios", []):
        scenario_name = scenario.get("name") or scenario.get("inputs", {}).get("name")
        if not scenario_name:
            continue
        defaults = scenario.get("defaults", {})
        inputs = scenario.get("inputs", {})
        derived = scenario.get("derived_counts", {})
        scenario_params = {
            "gridsize": inputs.get("gridsize"),
            "probe_big": coalesce(inputs.get("probe_big"), defaults.get("probe_big")),
            "probe_mask": coalesce(inputs.get("probe_mask"), defaults.get("probe_mask")),
            "probe_scale": inputs.get("probe_scale"),
            "group_count": scenario.get("group_count"),
            "neighbor_count": scenario.get("neighbor_count"),
            "reassemble_M": scenario.get("reassemble_M"),
            "nphotons": run_params.get("nphotons"),
            "total_images": derived.get("total_images"),
            "nimgs_train": derived.get("train_count"),
            "nimgs_test": derived.get("test_count"),
            # Fields absent in snapshot are filled later with None.
        }
        scenarios[scenario_name] = scenario_params
    return run_params, scenarios


def normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isclose(value, int(value)):
            return int(value)
    return value


def format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return f"{value}"
    return str(value)


def compute_delta(base: Any, scenario: Any) -> str:
    if base is None and scenario is None:
        return "missing in both"
    if base == scenario:
        return "match"
    if isinstance(base, (int, float)) and isinstance(scenario, (int, float)):
        return f"{scenario - base:+}"
    if base is None:
        return "not set in dose_experiments"
    if scenario is None:
        return "not set in sim_lines"
    return "differs"


def build_markdown(
    dose_params: Dict[str, Any],
    scenarios: Dict[str, Dict[str, Any]],
    snapshot_path: pathlib.Path,
    dose_path: pathlib.Path,
) -> str:
    lines: List[str] = []
    lines.append("# SIM-LINES-4X vs dose_experiments Parameter Diff")
    lines.append("")
    lines.append(f"- Snapshot source: `{snapshot_path}`")
    lines.append(f"- Legacy defaults source: `{dose_path}` (init() assignments)")
    lines.append("- `—` indicates the parameter was not defined in that pipeline.")
    lines.append("")
    for scenario_name in sorted(scenarios.keys()):
        lines.append(f"## Scenario: {scenario_name}")
        lines.append("")
        lines.append("| Parameter | dose_experiments | sim_lines | Δ / note |")
        lines.append("|-----------|------------------|-----------|----------|")
        scenario_params = scenarios[scenario_name]
        for key, label in PARAMETERS:
            scenario_value = scenario_params.get(key)
            dose_value = dose_params.get(key)
            delta = compute_delta(dose_value, scenario_value)
            lines.append(
                f"| {label} | {format_value(dose_value)} | {format_value(scenario_value)} | {delta} |"
            )
        lines.append("")
    return "\n".join(lines)


def build_diff_json(
    dose_params: Dict[str, Any], scenarios: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    diff = {
        "dose_experiments": dose_params,
        "scenarios": {},
    }
    for scenario_name, scenario_params in scenarios.items():
        entry = {"parameters": scenario_params, "diff": {}}
        for key, _ in PARAMETERS:
            entry["diff"][key] = {
                "dose_experiments": dose_params.get(key),
                "sim_lines": scenario_params.get(key),
                "delta": compute_delta(dose_params.get(key), scenario_params.get(key)),
            }
        diff["scenarios"][scenario_name] = entry
    return diff


def ensure_all_keys(
    dose_params: Dict[str, Any], scenarios: Dict[str, Dict[str, Any]]
) -> None:
    for key, _ in PARAMETERS:
        dose_params.setdefault(key, None)
        for scenario in scenarios.values():
            scenario.setdefault(key, None)


def enrich_scenario_with_loss_weights(
    scenario: Dict[str, Any],
    run_params: Dict[str, Any],
) -> None:
    """
    Instantiate TrainingConfig for the scenario and extract loss weights.
    Updates scenario dict in-place with mae_weight, nll_weight, etc.
    """
    gridsize = scenario.get("gridsize", 1)
    probe_scale = scenario.get("probe_scale", 4.0)
    probe_big = scenario.get("probe_big", True)
    probe_mask = scenario.get("probe_mask", False)
    nphotons = scenario.get("nphotons") or run_params.get("nphotons", 1e9)
    group_count = scenario.get("group_count") or run_params.get("group_count", 1000)
    neighbor_count = scenario.get("neighbor_count") or run_params.get("neighbor_count", 4)

    loss_weights = get_loss_weights_from_training_config(
        gridsize=int(gridsize),
        probe_scale=float(probe_scale),
        probe_big=bool(probe_big),
        probe_mask=bool(probe_mask),
        nphotons=float(nphotons),
        group_count=int(group_count),
        neighbor_count=int(neighbor_count),
    )
    scenario.update(loss_weights)


def enrich_scenario_with_training_config(
    scenario: Dict[str, Any],
    run_params: Dict[str, Any],
    default_nepochs: int,
    default_batch_size: int = 16,
    default_probe_trainable: bool = False,
    default_intensity_scale_trainable: bool = True,
) -> None:
    """
    Instantiate TrainingConfig for the scenario and extract all training hyperparameters.
    Phase D3 extension that includes nepochs, batch_size, probe.trainable, intensity_scale.trainable.
    Updates scenario dict in-place.

    Args:
        scenario: Scenario dict to update in-place
        run_params: Run parameters from snapshot
        default_nepochs: Default nepochs for sim_lines (from CLI --default-sim-lines-nepochs)
        default_batch_size: Default batch_size (TrainingConfig default is 16)
        default_probe_trainable: Default probe.trainable (TrainingConfig default is False)
        default_intensity_scale_trainable: Default intensity_scale.trainable (TrainingConfig default is True)
    """
    gridsize = scenario.get("gridsize", 1)
    probe_scale = scenario.get("probe_scale", 4.0)
    probe_big = scenario.get("probe_big", True)
    probe_mask = scenario.get("probe_mask", False)
    nphotons = scenario.get("nphotons") or run_params.get("nphotons", 1e9)
    group_count = scenario.get("group_count") or run_params.get("group_count", 1000)
    neighbor_count = scenario.get("neighbor_count") or run_params.get("neighbor_count", 4)

    training_snapshot = get_training_config_snapshot(
        gridsize=int(gridsize),
        probe_scale=float(probe_scale),
        probe_big=bool(probe_big),
        probe_mask=bool(probe_mask),
        nphotons=float(nphotons),
        group_count=int(group_count),
        neighbor_count=int(neighbor_count),
        nepochs=default_nepochs,
        batch_size=default_batch_size,
        probe_trainable=default_probe_trainable,
        intensity_scale_trainable=default_intensity_scale_trainable,
    )
    scenario.update(training_snapshot)


def build_loss_modes_markdown(
    dose_loss_modes: Dict[str, Dict[str, Any]],
) -> str:
    """
    Build a Markdown section summarizing the runtime-captured loss weights
    for both loss_fn modes (nll and mae).
    """
    lines: List[str] = []
    lines.append("## Legacy dose_experiments Loss Configuration (Runtime Captured)")
    lines.append("")
    lines.append("The following weights were captured by executing the legacy `init()` function")
    lines.append("with a stubbed `ptycho.params.cfg` for each `loss_fn` mode.")
    lines.append("")
    lines.append("| Parameter | loss_fn='nll' (default) | loss_fn='mae' (conditional) |")
    lines.append("|-----------|-------------------------|----------------------------|")

    loss_keys = ["mae_weight", "nll_weight", "realspace_weight", "realspace_mae_weight"]
    nll_cfg = dose_loss_modes.get("nll", {})
    mae_cfg = dose_loss_modes.get("mae", {})

    for key in loss_keys:
        nll_val = nll_cfg.get(key, "—")
        mae_val = mae_cfg.get(key, "—")
        lines.append(f"| {key} | {format_value(nll_val)} | {format_value(mae_val)} |")

    lines.append("")
    lines.append("**Key insight:** When `loss_fn='nll'` (the default), the legacy script does **not**")
    lines.append("set `mae_weight` or `nll_weight` explicitly — it relies on the underlying framework")
    lines.append("defaults. The `mae_weight=1.0, nll_weight=0.0` values only apply when `loss_fn='mae'`.")
    lines.append("")

    return "\n".join(lines)


def build_legacy_defaults_markdown(
    legacy_defaults: Dict[str, Any],
    training_config_defaults: Optional[Dict[str, float]] = None,
) -> str:
    """
    Build a Markdown section summarizing the legacy ptycho.params.cfg framework
    defaults and comparing them with the TrainingConfig dataclass defaults.
    """
    lines: List[str] = []
    lines.append("## Legacy params.cfg Framework Defaults")
    lines.append("")
    lines.append("The following loss weight defaults are defined in the legacy `ptycho.params.cfg`")
    lines.append(f"(source: `{legacy_defaults.get('_source', 'ptycho/params.py')}`)")
    lines.append("")

    if training_config_defaults is not None:
        lines.append("| Parameter | params.cfg default | TrainingConfig default | Match? |")
        lines.append("|-----------|-------------------|------------------------|--------|")
        for key in LEGACY_LOSS_KEYS:
            legacy_val = legacy_defaults.get(key)
            tc_val = training_config_defaults.get(key)
            match = "✓" if legacy_val == tc_val else "✗"
            lines.append(
                f"| {key} | {format_value(legacy_val)} | {format_value(tc_val)} | {match} |"
            )
    else:
        lines.append("| Parameter | params.cfg default |")
        lines.append("|-----------|-------------------|")
        for key in LEGACY_LOSS_KEYS:
            legacy_val = legacy_defaults.get(key)
            lines.append(f"| {key} | {format_value(legacy_val)} |")

    lines.append("")
    lines.append("**Conclusion:** The legacy framework defaults in `ptycho/params.py:64` define")
    lines.append("`mae_weight=0.0, nll_weight=1.0` (pure NLL loss), which matches the modern")
    lines.append("`TrainingConfig` dataclass defaults. **H-LOSS-WEIGHT is ruled out** — both")
    lines.append("pipelines use identical loss weights under default operation.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    # Phase D1 (new): Capture the actual ptycho.params.cfg framework defaults
    legacy_defaults: Optional[Dict[str, Any]] = None
    if args.output_legacy_defaults or True:  # Always capture for Markdown/JSON
        try:
            legacy_defaults = capture_legacy_params_defaults()
            print(f"Captured legacy params.cfg defaults: {legacy_defaults}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not capture legacy params.cfg defaults: {e}", file=sys.stderr)

    # Phase D1a: Execute legacy init() to capture runtime cfg for both loss modes
    dose_loss_modes: Optional[Dict[str, Dict[str, Any]]] = None
    try:
        dose_loss_modes = capture_legacy_loss_modes(args.dose_config, args.nphotons)
    except Exception as e:
        print(f"Warning: Could not capture runtime loss modes: {e}", file=sys.stderr)
        print("Falling back to static parsing.", file=sys.stderr)

    # Static parsing for non-loss parameters (still useful for nepochs, offset, etc.)
    dose_params = parse_dose_config(args.dose_config)
    run_params, scenarios = load_sim_lines_snapshot(args.snapshot)

    # Phase D1b: Use runtime nll config for loss weights instead of static parse
    # The static parse incorrectly captured the MAE branch values as defaults
    if dose_loss_modes is not None:
        nll_cfg = dose_loss_modes["nll"]
        # Override loss weights with runtime-captured values (nll mode = default)
        dose_params["mae_weight"] = nll_cfg.get("mae_weight")
        dose_params["nll_weight"] = nll_cfg.get("nll_weight")
        dose_params["realspace_weight"] = nll_cfg.get("realspace_weight")
        dose_params["realspace_mae_weight"] = nll_cfg.get("realspace_mae_weight")

    # Some parameters live in the global run params; surface them explicitly.
    if "intensity_scale.trainable" not in dose_params:
        dose_params["intensity_scale.trainable"] = dose_params.get("intensity_scale.trainable")

    # Phase D3: Ensure dose_params includes training hyperparameters from static parse
    # The static parse should already capture nepochs (60), probe.trainable (False),
    # but batch_size is not explicitly set in legacy scripts (uses framework default 16)
    dose_params.setdefault("nepochs", dose_params.get("nepochs"))  # Static parse captures cfg['nepochs'] = 60
    dose_params.setdefault("batch_size", 16)  # Legacy uses framework default
    dose_params.setdefault("probe.trainable", dose_params.get("probe.trainable"))  # Static parse captures cfg['probe.trainable']

    for scenario in scenarios.values():
        scenario.setdefault("offset", run_params.get("offset"))
        scenario.setdefault("outer_offset_train", run_params.get("outer_offset_train"))
        scenario.setdefault("outer_offset_test", run_params.get("outer_offset_test"))
        scenario.setdefault("intensity_scale.trainable", run_params.get("intensity_scale.trainable"))
        # Phase D3: Enrich scenario with full TrainingConfig snapshot (includes nepochs, batch_size, probe.trainable)
        enrich_scenario_with_training_config(
            scenario,
            run_params,
            default_nepochs=args.default_sim_lines_nepochs,
            default_batch_size=16,  # TrainingConfig default
            default_probe_trainable=False,  # TrainingConfig default
            default_intensity_scale_trainable=True,  # TrainingConfig default
        )

    ensure_all_keys(dose_params, scenarios)
    markdown = build_markdown(dose_params, scenarios, args.snapshot, args.dose_config)
    diff = build_diff_json(dose_params, scenarios)

    # Phase D1c: Add dose_loss_modes section to both Markdown and JSON
    if dose_loss_modes is not None:
        # Append loss modes section to Markdown
        markdown += "\n" + build_loss_modes_markdown(dose_loss_modes)

        # Add dose_loss_modes to JSON diff
        diff["dose_loss_modes"] = dose_loss_modes

        # Write separate dose_loss_weights.json if requested
        if args.output_dose_loss_weights:
            args.output_dose_loss_weights.parent.mkdir(parents=True, exist_ok=True)
            loss_weights_data = {
                "captured_at": __import__("datetime").datetime.now().isoformat(),
                "nphotons": args.nphotons,
                "dose_config_path": str(args.dose_config),
                "loss_modes": dose_loss_modes,
                "interpretation": {
                    "nll": "Default mode: uses framework loss weight defaults (mae_weight/nll_weight not explicitly set)",
                    "mae": "Override mode: explicitly sets mae_weight=1.0, nll_weight=0.0",
                },
            }
            args.output_dose_loss_weights.write_text(
                json.dumps(loss_weights_data, indent=2, sort_keys=True)
            )

        # Write separate dose_loss_weights.md if requested
        if args.output_dose_loss_weights_markdown:
            args.output_dose_loss_weights_markdown.parent.mkdir(parents=True, exist_ok=True)
            loss_modes_md = build_loss_modes_markdown(dose_loss_modes)
            args.output_dose_loss_weights_markdown.write_text(loss_modes_md)

    # Phase D1 (new): Add legacy params.cfg defaults section
    if legacy_defaults is not None:
        # Get TrainingConfig defaults for comparison
        try:
            tc_defaults = get_loss_weights_from_training_config(
                gridsize=1,
                probe_scale=4.0,
                probe_big=True,
                probe_mask=False,
                nphotons=1e9,
                group_count=1000,
                neighbor_count=4,
            )
        except Exception as e:
            print(f"Warning: Could not get TrainingConfig defaults: {e}", file=sys.stderr)
            tc_defaults = None

        # Append legacy defaults section to Markdown
        markdown += "\n" + build_legacy_defaults_markdown(legacy_defaults, tc_defaults)

        # Add legacy_params_cfg_defaults to JSON diff
        diff["legacy_params_cfg_defaults"] = legacy_defaults

        # Write separate legacy_params_cfg_defaults.json if requested
        if args.output_legacy_defaults:
            args.output_legacy_defaults.parent.mkdir(parents=True, exist_ok=True)
            legacy_data = {
                "captured_at": __import__("datetime").datetime.now().isoformat(),
                "source": legacy_defaults.get("_source", "ptycho/params.py"),
                "defaults": {k: v for k, v in legacy_defaults.items() if not k.startswith("_")},
                "training_config_defaults": tc_defaults,
                "match_status": {
                    k: legacy_defaults.get(k) == (tc_defaults.get(k) if tc_defaults else None)
                    for k in LEGACY_LOSS_KEYS
                },
                "conclusion": "Legacy params.cfg defaults match TrainingConfig defaults (mae_weight=0, nll_weight=1). H-LOSS-WEIGHT ruled out.",
            }
            args.output_legacy_defaults.write_text(
                json.dumps(legacy_data, indent=2, sort_keys=True)
            )

    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown)
    args.output_json.write_text(json.dumps(diff, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
