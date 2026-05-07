"""Generate the NeurIPS manuscript model-configuration appendix table."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


NEURIPS_DIR = Path("docs") / "plans" / "NEURIPS-HYBRID-RESNET-2026"
TABLES_DIR = NEURIPS_DIR / "tables"

CDI_BASE_ROOT = (
    Path(".artifacts")
    / "work"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-04-30-cdi-lines128-uno-table-extension"
    / "runs"
    / "complete_table_plus_uno_20260504T100347Z"
)
CDI_FFNO_NO_REFINER_ROOT = (
    Path(".artifacts")
    / "work"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun"
    / "runs"
    / "ffno_no_refiner_20260506T223454Z"
)
CDI_SUPERVISED_FFNO_ROOT = (
    Path(".artifacts")
    / "work"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun"
    / "runs"
    / "supervised_ffno_no_refiner_20260506T232535Z"
)
BRDT_40EP_ROOT = (
    Path(".artifacts")
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-05-06-brdt-corrected-ffno-40ep-rerun"
)

CDI_DISPLAY_LABELS = {
    "baseline": ("CNN", "supervised"),
    "pinn": ("CNN", "PINN"),
    "pinn_fno_vanilla": ("FNO", "PINN"),
    "pinn_ffno": ("FFNO", "PINN"),
    "supervised_ffno": ("FFNO", "supervised"),
    "pinn_hybrid_resnet": ("SRU-Net", "PINN"),
    "pinn_neuralop_uno": ("U-NO", "PINN"),
    "supervised_neuralop_uno": ("U-NO", "supervised"),
}

CNS_DISPLAY_LABELS = {
    "author_ffno_cns_base": "FFNO",
    "spectral_resnet_bottleneck_base": "SRU-Net",
    "fno_base": "FNO",
    "unet_strong": "U-Net",
}

BRDT_DISPLAY_LABELS = {
    "hybrid_resnet": "SRU-Net",
    "ffno": "FFNO",
}


@dataclass(frozen=True)
class ParameterCountResult:
    unique_trainable_params: int
    raw_recorded_parameter_count: int
    parameter_count_kind: str
    duplicate_groups: list[str]


@dataclass(frozen=True)
class ModelConfigRow:
    benchmark: str
    display_model: str
    row_id: str
    internal_architecture: str
    training_objective: str
    input_output_contract: str
    width: str
    fourier_modes: str
    encoder_blocks: str
    bottleneck_blocks: str
    downsampling: str
    skip_or_fusion: str
    unique_trainable_params: int
    raw_recorded_parameter_count: int | None
    parameter_count_kind: str
    parameter_count_source: str
    config_source: str
    notes: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_rel(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _as_str(value: object, *, missing: str = "not_recorded") -> str:
    if value is None:
        return missing
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _tensor_numel(value: Any) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    return 0


def _tensor_shape(value: Any) -> tuple[int, ...] | None:
    if hasattr(value, "shape"):
        return tuple(int(dim) for dim in value.shape)
    return None


def _state_tensor_items(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state_dict.items() if _tensor_numel(value) > 0}


def count_unique_state_dict_params(state_dict: Mapping[str, Any]) -> ParameterCountResult:
    """Count effective generator parameters while deduping saved wrapper aliases.

    CDI PyTorch rows save the same generator under both ``model.generator`` and
    ``model.autoencoder``. The manuscript convention counts the effective
    prediction model once, not every state-dict alias.
    """

    tensors = _state_tensor_items(state_dict)
    generator = {
        key.removeprefix("model.generator."): value
        for key, value in tensors.items()
        if key.startswith("model.generator.")
    }
    autoencoder = {
        key.removeprefix("model.autoencoder."): value
        for key, value in tensors.items()
        if key.startswith("model.autoencoder.")
    }

    if generator or autoencoder:
        raw_total = sum(_tensor_numel(value) for value in generator.values()) + sum(
            _tensor_numel(value) for value in autoencoder.values()
        )
    else:
        raw_total = sum(_tensor_numel(value) for value in tensors.values())

    duplicate_aliases = False
    if generator and autoencoder and set(generator) == set(autoencoder):
        duplicate_aliases = all(
            _tensor_shape(generator[name]) == _tensor_shape(autoencoder[name])
            and _tensor_numel(generator[name]) == _tensor_numel(autoencoder[name])
            for name in generator
        )

    if duplicate_aliases:
        unique_total = sum(_tensor_numel(value) for value in generator.values())
        return ParameterCountResult(
            unique_trainable_params=unique_total,
            raw_recorded_parameter_count=raw_total,
            parameter_count_kind="deduped_unique_effective_trainable_params",
            duplicate_groups=["model.generator/model.autoencoder"],
        )

    return ParameterCountResult(
        unique_trainable_params=raw_total,
        raw_recorded_parameter_count=raw_total,
        parameter_count_kind="unique_effective_trainable_params",
        duplicate_groups=[],
    )


def _load_torch_state_dict(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    try:
        import torch
    except ImportError:
        return None

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, Mapping):
        if "state_dict" in loaded and isinstance(loaded["state_dict"], Mapping):
            return loaded["state_dict"]
        return loaded
    return None


def row_to_dict(row: ModelConfigRow) -> dict[str, object]:
    return asdict(row)


def _manifest_rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["model_id"]): row for row in _read_json(path).get("rows", [])}


def _load_runner_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    return dict(payload.get("torch_runner_config") or payload)


def _cdi_skip_or_fusion(architecture: str, cfg: Mapping[str, Any]) -> str:
    if architecture == "hybrid_resnet":
        skip = _as_str(cfg.get("hybrid_skip_connections"))
        style = _as_str(cfg.get("hybrid_skip_style"))
        conv_scale = _as_str(cfg.get("hybrid_encoder_conv_hidden_scale"))
        spectral_scale = _as_str(cfg.get("hybrid_encoder_spectral_hidden_scale"))
        return f"encoder conv/spectral; skip={skip}; style={style}; scales={conv_scale}/{spectral_scale}"
    if architecture == "spectral_resnet_bottleneck_net":
        gate = _as_str(cfg.get("spectral_bottleneck_gate_mode"))
        return f"spectral bottleneck gated_add; gate={gate}"
    if architecture == "ffno":
        cnn_blocks = int(cfg.get("fno_cnn_blocks") or 0)
        if cnn_blocks > 0:
            return "factorized Fourier + local CNN refiners"
        return "factorized Fourier"
    if architecture == "fno_vanilla":
        return "spectral conv + pointwise mixing"
    if architecture == "neuralop_uno":
        return "U-NO multiscale operator"
    if architecture == "cnn":
        return "CNN local convolutions"
    return "not_recorded"


def _cdi_bottleneck_blocks(architecture: str, cfg: Mapping[str, Any]) -> str:
    if architecture == "hybrid_resnet":
        return _as_str(cfg.get("hybrid_resnet_blocks"))
    if architecture == "spectral_resnet_bottleneck_net":
        return _as_str(cfg.get("spectral_bottleneck_blocks") or cfg.get("hybrid_resnet_blocks"))
    return "not_applicable"


def _cdi_downsampling(architecture: str, cfg: Mapping[str, Any]) -> str:
    if architecture in {"hybrid_resnet", "spectral_resnet_bottleneck_net"}:
        return _as_str(cfg.get("hybrid_downsample_steps"))
    return "not_applicable"


def _cdi_param_count(
    run_dir: Path,
    manifest_row: Mapping[str, Any],
    repo_root: Path,
) -> tuple[int, int | None, str, str, str]:
    model_path = run_dir / "model.pt"
    recorded = manifest_row.get("parameter_count")
    if model_path.exists():
        state = _load_torch_state_dict(model_path)
        if state is not None:
            result = count_unique_state_dict_params(state)
            raw_recorded = int(recorded) if recorded is not None else result.raw_recorded_parameter_count
            notes = (
                "deduped generator/autoencoder aliases"
                if result.duplicate_groups
                else "counted from model state"
            )
            if str(manifest_row.get("architecture_id") or run_dir.name) == "ffno":
                cfg = _load_runner_config(run_dir / "config.json")
                cnn_blocks = int(cfg.get("fno_cnn_blocks") or 0)
                if cnn_blocks > 0:
                    notes = (
                        f"historical fno_cnn_blocks={cnn_blocks} local-refiner proxy; "
                        "corrected no-refiner rerun pending"
                    )
                else:
                    notes = "corrected no-refiner active CDI paper row"
            return (
                result.unique_trainable_params,
                raw_recorded,
                result.parameter_count_kind,
                _repo_rel(repo_root, model_path),
                notes,
            )
    if recorded is not None:
        return (
            int(recorded),
            int(recorded),
            "uses_recorded_count",
            "model_manifest.json",
            "recorded count; unique extractor unavailable for this row",
        )
    return (0, None, "not_recorded", "not_recorded", "parameter count not recorded")


def load_cdi_config_rows(repo_root: Path) -> list[ModelConfigRow]:
    table_path = repo_root / TABLES_DIR / "cdi_lines128_metrics_extended.json"
    table_rows = _read_json(table_path).get("rows", [])
    base_root = repo_root / CDI_BASE_ROOT
    corrected_ffno_root = repo_root / CDI_FFNO_NO_REFINER_ROOT
    supervised_ffno_root = repo_root / CDI_SUPERVISED_FFNO_ROOT
    manifest_rows = _manifest_rows_by_id(base_root / "model_manifest.json")
    manifest_rows.update(_manifest_rows_by_id(corrected_ffno_root / "model_manifest.json"))
    manifest_rows.update(_manifest_rows_by_id(supervised_ffno_root / "model_manifest.json"))
    rows: list[ModelConfigRow] = []

    for metric_row in table_rows:
        row_id = str(metric_row["row_id"])
        if row_id == "pinn_ffno":
            row_root = corrected_ffno_root
        elif row_id == "supervised_ffno":
            row_root = supervised_ffno_root
        else:
            row_root = base_root
        run_dir = row_root / "runs" / row_id
        config_path = run_dir / "config.json"
        cfg = _load_runner_config(config_path)
        manifest_row = manifest_rows.get(row_id, {})
        architecture = str(manifest_row.get("architecture_id") or cfg.get("architecture") or "not_recorded")
        display_model, training = CDI_DISPLAY_LABELS.get(
            row_id,
            (str(metric_row.get("model", row_id)), str(metric_row.get("training", "not_recorded"))),
        )
        unique_params, raw_params, count_kind, count_source, notes = _cdi_param_count(
            run_dir,
            manifest_row,
            repo_root,
        )
        if count_source == "model_manifest.json":
            count_source = _repo_rel(repo_root, row_root / "model_manifest.json")
        rows.append(
            ModelConfigRow(
                benchmark="Synthetic CDI",
                display_model=display_model,
                row_id=row_id,
                internal_architecture=architecture,
                training_objective=training,
                input_output_contract="diffraction intensity -> complex object",
                width=_as_str(cfg.get("fno_width") or cfg.get("hidden_channels")),
                fourier_modes=_as_str(cfg.get("fno_modes")),
                encoder_blocks=_as_str(cfg.get("fno_blocks")),
                bottleneck_blocks=_cdi_bottleneck_blocks(architecture, cfg),
                downsampling=_cdi_downsampling(architecture, cfg),
                skip_or_fusion=_cdi_skip_or_fusion(architecture, cfg),
                unique_trainable_params=unique_params,
                raw_recorded_parameter_count=raw_params,
                parameter_count_kind=count_kind,
                parameter_count_source=count_source,
                config_source=_repo_rel(repo_root, config_path) if config_path.exists() else "not_recorded",
                notes=notes,
            )
        )
    return rows


def _load_model_profile(repo_root: Path, source_root: str, row_id: str) -> tuple[dict[str, Any], Path | None]:
    root = repo_root / source_root
    path = root / f"model_profile_{row_id}.json"
    if path.exists():
        return _read_json(path), path
    return {}, None


def _cns_skip_or_fusion(row_id: str, cfg: Mapping[str, Any]) -> str:
    if row_id == "author_ffno_cns_base":
        return "factorized Fourier; shared_weight=" + _as_str(
            cfg.get("author_ffno_share_weight") or cfg.get("share_weight")
        )
    if row_id == "spectral_resnet_bottleneck_base":
        return "spectral encoder + ResNet decoder; skip=" + _as_str(
            cfg.get("hybrid_skip_connections")
        )
    if row_id == "fno_base":
        return "spectral conv + pointwise mixing"
    if row_id == "unet_strong":
        return "U-Net local convolutions"
    return "not_recorded"


def load_cns_config_rows(repo_root: Path) -> list[ModelConfigRow]:
    table_path = repo_root / TABLES_DIR / "pdebench_cns_matched_condition_metrics.json"
    payload = _read_json(table_path)
    rows: list[ModelConfigRow] = []
    for table_row in payload.get("rows", []):
        row_id = str(table_row["row_id"])
        profile, profile_path = _load_model_profile(
            repo_root,
            str(table_row.get("source_run_root", "")),
            row_id,
        )
        cfg = dict(profile.get("profile_config") or {})
        param_count = int(table_row["parameter_count"])
        rows.append(
            ModelConfigRow(
                benchmark="PDEBench CNS",
                display_model=CNS_DISPLAY_LABELS.get(row_id, str(table_row.get("manuscript_label", row_id))),
                row_id=row_id,
                internal_architecture=str(profile.get("base_model") or cfg.get("base_model") or row_id),
                training_objective=f"supervised {table_row.get('training_loss', 'mse')}",
                input_output_contract=(
                    f"history_len={table_row.get('history_len')} -> next frame; "
                    f"{table_row.get('split_label', 'split not recorded')}; "
                    f"{table_row.get('epochs')} epochs"
                ),
                width=_as_str(cfg.get("hidden_channels") or cfg.get("author_ffno_width") or cfg.get("unet_init_features")),
                fourier_modes=_as_str(cfg.get("fno_modes") or cfg.get("author_ffno_modes")),
                encoder_blocks=_as_str(cfg.get("fno_blocks") or cfg.get("author_ffno_layers")),
                bottleneck_blocks=_as_str(
                    cfg.get("hybrid_resnet_blocks") or cfg.get("spectral_bottleneck_blocks"),
                    missing="not_applicable",
                ),
                downsampling=_as_str(cfg.get("hybrid_downsample_steps")),
                skip_or_fusion=_cns_skip_or_fusion(row_id, cfg),
                unique_trainable_params=param_count,
                raw_recorded_parameter_count=param_count,
                parameter_count_kind="unique_effective_trainable_params",
                parameter_count_source=_repo_rel(repo_root, table_path),
                config_source=_repo_rel(repo_root, profile_path) if profile_path else _repo_rel(repo_root, table_path),
                notes="capped CNS row; runtime is training/evaluation provenance, not inference throughput",
            )
        )
    return rows


def load_brdt_config_rows(repo_root: Path) -> list[ModelConfigRow]:
    metrics_path = repo_root / BRDT_40EP_ROOT / "combined_metrics.json"
    payload = _read_json(metrics_path)
    rows: list[ModelConfigRow] = []
    for metric_row in payload.get("rows", []):
        row_id = str(metric_row["row_id"])
        if row_id not in BRDT_DISPLAY_LABELS:
            continue
        profile_path = repo_root / BRDT_40EP_ROOT / "rows" / row_id / "model_profile.json"
        profile = _read_json(profile_path) if profile_path.exists() else {}
        cfg = dict(profile.get("arch_kwargs") or {})
        param_count = int(profile.get("parameter_count") or metric_row.get("runtime", {}).get("parameter_count", 0))
        rows.append(
            ModelConfigRow(
                benchmark="BRDT",
                display_model=BRDT_DISPLAY_LABELS[row_id],
                row_id=row_id,
                internal_architecture=str(profile.get("architecture") or metric_row.get("architecture") or row_id),
                training_objective="supervised+Born",
                input_output_contract="born_init_image -> q_pred",
                width=_as_str(cfg.get("hidden_channels")),
                fourier_modes=_as_str(cfg.get("fno_modes")),
                encoder_blocks=_as_str(cfg.get("fno_blocks")),
                bottleneck_blocks=_as_str(cfg.get("resnet_blocks"), missing="not_applicable"),
                downsampling=_as_str(cfg.get("downsample_steps")),
                skip_or_fusion="factorized Fourier" if row_id == "ffno" else "spectral encoder + ResNet decoder",
                unique_trainable_params=param_count,
                raw_recorded_parameter_count=param_count,
                parameter_count_kind="unique_effective_trainable_params",
                parameter_count_source=_repo_rel(repo_root, profile_path) if profile_path.exists() else _repo_rel(repo_root, metrics_path),
                config_source=_repo_rel(repo_root, profile_path) if profile_path.exists() else _repo_rel(repo_root, metrics_path),
                notes="40-epoch BRDT additive-secondary paired rerun",
            )
        )
    return rows


def build_model_config_rows(repo_root: Path) -> list[ModelConfigRow]:
    return [
        *load_cdi_config_rows(repo_root),
        *load_cns_config_rows(repo_root),
        *load_brdt_config_rows(repo_root),
    ]


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _format_params(value: int) -> str:
    return f"{value:,}"


def render_model_config_tex(rows: Sequence[ModelConfigRow]) -> str:
    lines = [
        r"\begin{tabular}{lllrrrrll}",
        r"\toprule",
        r"Benchmark & Model row & Training & Width & Modes & Enc. & Bott. & Down & Unique params \\",
        r"\midrule",
    ]
    last_benchmark: str | None = None
    for row in rows:
        if last_benchmark is not None and row.benchmark != last_benchmark:
            lines.append(r"\midrule")
        last_benchmark = row.benchmark
        lines.append(
            " & ".join(
                [
                    _latex_escape(row.benchmark),
                    _latex_escape(f"{row.display_model} ({row.row_id})"),
                    _latex_escape(row.training_objective),
                    _latex_escape(row.width),
                    _latex_escape(row.fourier_modes),
                    _latex_escape(row.encoder_blocks),
                    _latex_escape(row.bottleneck_blocks),
                    _latex_escape(row.downsampling),
                    _latex_escape(_format_params(row.unique_trainable_params)),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_model_config_json(rows: Sequence[ModelConfigRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "model_config_by_benchmark_v1",
        "parameter_count_convention": (
            "unique trainable parameters in the effective prediction model; "
            "duplicate wrapper aliases are not counted twice"
        ),
        "rows": [row_to_dict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_model_config_csv(rows: Sequence[ModelConfigRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row_to_dict(rows[0]).keys()) if rows else list(ModelConfigRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row_to_dict(row))


def write_model_config_table(repo_root: Path, output_dir: Path) -> dict[str, str]:
    rows = build_model_config_rows(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "model_config_by_benchmark.json"
    csv_path = output_dir / "model_config_by_benchmark.csv"
    tex_path = output_dir / "model_config_by_benchmark.tex"
    write_model_config_json(rows, json_path)
    write_model_config_csv(rows, csv_path)
    tex_path.write_text(render_model_config_tex(rows), encoding="utf-8")
    return {
        "json": _repo_rel(repo_root, json_path),
        "csv": _repo_rel(repo_root, csv_path),
        "tex": _repo_rel(repo_root, tex_path),
    }
