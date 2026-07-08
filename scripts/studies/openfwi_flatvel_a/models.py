"""Tiny local model adapters and official InversionNet probe for OpenFWI."""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptycho_torch.generators.fno import SpatialLifter
from ptycho_torch.generators.hybrid_resnet import AvgPoolConvDownsample, HybridResnetEncoderBlock
from ptycho_torch.generators.resnet_components import CycleGanUpsampler, ResnetBottleneck
from scripts.studies.openfwi_flatvel_a.run_config import get_model_profile


class PadCropWrapper(nn.Module):
    def __init__(self, module: nn.Module, *, multiple: int = 1):
        super().__init__()
        self.module = module
        self.multiple = max(1, int(multiple))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        padded_height = ((height + self.multiple - 1) // self.multiple) * self.multiple
        padded_width = ((width + self.multiple - 1) // self.multiple) * self.multiple
        y = self.module(F.pad(x, (0, padded_width - width, 0, padded_height - height)))
        return y[..., :height, :width]


class HybridResnetSmoke(nn.Module):
    """Supervised real-channel adapter for the Hybrid ResNet body."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        modes: int,
        fno_blocks: int,
        resnet_blocks: int,
        downsample_steps: int,
    ):
        super().__init__()
        self.lifter = SpatialLifter(in_channels, hidden_channels)
        channels = hidden_channels
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for index in range(max(1, int(fno_blocks))):
            self.encoder_blocks.append(HybridResnetEncoderBlock(channels, modes=modes))
            if index < downsample_steps:
                self.downsample_layers.append(AvgPoolConvDownsample(channels, channels * 2))
                channels *= 2
        self.resnet = ResnetBottleneck(channels, n_blocks=max(1, int(resnet_blocks)))
        actual_downsample_steps = len(self.downsample_layers)
        upsample_widths = [channels]
        for _ in range(actual_downsample_steps):
            upsample_widths.append(max(hidden_channels, upsample_widths[-1] // 2))
        self.upsample_layers = nn.ModuleList(
            [
                CycleGanUpsampler(upsample_widths[index], upsample_widths[index + 1])
                for index in range(actual_downsample_steps)
            ]
        )
        self.output = nn.Conv2d(upsample_widths[-1], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifter(x)
        for index, block in enumerate(self.encoder_blocks):
            x = block(x)
            if index < len(self.downsample_layers):
                x = self.downsample_layers[index](x)
        x = self.resnet(x)
        for upsample in self.upsample_layers:
            x = upsample(x)
        return self.output(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.down = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, stride=2, padding=1)
        self.mid = nn.Sequential(nn.GELU(), nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1), nn.GELU())
        self.up = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=2, stride=2)
        self.out = nn.Conv2d(hidden_channels * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.enc(x)
        y = self.up(self.mid(self.down(skip)))
        if y.shape[-2:] != skip.shape[-2:]:
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(torch.cat([skip, y], dim=1))


def _build_fno(
    *,
    in_channels: int,
    out_channels: int,
    spatial_shape: tuple[int, int],
    hidden_channels: int,
    fno_modes: int,
    fno_blocks: int,
) -> nn.Module:
    try:
        from neuralop.models import FNO
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"neuralop FNO unavailable: {exc}") from exc
    modes = min(int(fno_modes), int(spatial_shape[0]), int(spatial_shape[1]))
    return FNO(
        n_modes=(modes, modes),
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layers=max(1, int(fno_blocks)),
    )


def build_model(
    profile_id: str,
    *,
    in_channels: int,
    out_channels: int,
    spatial_shape: tuple[int, int],
    profile_config: Mapping[str, Any] | None,
) -> nn.Module:
    """Build a supervised model with ordinary real-channel `model(x) -> y` semantics."""
    profile = get_model_profile(profile_id)
    config = {**profile.to_model_config(), **dict(profile_config or {})}
    hidden_channels = int(config.get("hidden_channels") or 8)
    if profile.base_model == "hybrid_resnet":
        downsample_steps = int(config.get("hybrid_downsample_steps") or 2)
        return PadCropWrapper(
            HybridResnetSmoke(
                in_channels,
                out_channels,
                hidden_channels,
                modes=int(config.get("fno_modes") or 4),
                fno_blocks=int(config.get("fno_blocks") or 2),
                resnet_blocks=int(config.get("hybrid_resnet_blocks") or 1),
                downsample_steps=downsample_steps,
            ),
            multiple=2 ** max(0, downsample_steps),
        )
    if profile.base_model == "unet":
        return PadCropWrapper(SmallUNet(in_channels, out_channels, hidden_channels), multiple=2)
    if profile.base_model == "fno":
        return _build_fno(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_shape=spatial_shape,
            hidden_channels=hidden_channels,
            fno_modes=int(config.get("fno_modes") or 4),
            fno_blocks=int(config.get("fno_blocks") or 2),
        )
    raise ValueError(f"profile {profile_id} does not map to a local smoke model")


def count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def describe_model(model: nn.Module, *, profile_id: str, profile_config: Mapping[str, Any]) -> dict[str, Any]:
    profile = get_model_profile(profile_id)
    config = {**profile.to_model_config(), **dict(profile_config)}
    return {
        "profile_id": profile_id,
        "class_name": model.__class__.__name__,
        "parameter_count": count_parameters(model),
        "profile_config": config,
    }


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except Exception:
        return None


def _load_module_from_path(module_path: Path, *, repo_path: Path) -> Any:
    module_name = f"_openfwi_official_{abs(hash(str(module_path.resolve())))}"
    previous_path = list(sys.path)
    sys.path.insert(0, str(repo_path))
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not create import spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path = previous_path
        sys.modules.pop(module_name, None)


def _find_inversionnet(module: Any) -> type[nn.Module] | None:
    candidate = getattr(module, "InversionNet", None)
    if isinstance(candidate, type):
        return candidate
    model_dict = getattr(module, "model_dict", None)
    if isinstance(model_dict, Mapping):
        candidate = model_dict.get("InversionNet")
        if isinstance(candidate, type):
            return candidate
    return None


def _constructor_kwargs(model_cls: type[nn.Module]) -> dict[str, Any]:
    try:
        signature = inspect.signature(model_cls)
    except (TypeError, ValueError):
        return {}
    tiny_defaults = {
        "dim1": 1,
        "dim2": 2,
        "dim3": 2,
        "dim4": 4,
        "dim5": 4,
        "sample_spatial": 1.0,
        "ratio": 1.0,
    }
    return {name: value for name, value in tiny_defaults.items() if name in signature.parameters}


def _probe_forward(model_cls: type[nn.Module]) -> dict[str, Any]:
    input_shape = [1, 5, 1000, 70]
    expected_output_shape = [1, 1, 70, 70]
    constructor_kwargs = _constructor_kwargs(model_cls)
    model = model_cls(**constructor_kwargs)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(*input_shape, dtype=torch.float32))
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"InversionNet forward returned {type(output).__name__}, not Tensor")
    output_shape = list(output.shape)
    if output_shape != expected_output_shape:
        raise ValueError(f"InversionNet output shape {output_shape} != {expected_output_shape}")
    return {
        "status": "succeeded",
        "input_shape": input_shape,
        "output_shape": output_shape,
        "constructor_kwargs": constructor_kwargs,
    }


def probe_official_inversionnet(repo_path: Path | None) -> dict[str, Any]:
    """Probe an optional external OpenFWI checkout without importing at module load."""
    if repo_path is None:
        return {
            "status": "blocked",
            "reason": "official_repo_missing",
            "message": "--official-openfwi-repo was not supplied",
        }
    repo_path = Path(repo_path).expanduser().resolve()
    if not repo_path.exists():
        return {
            "status": "blocked",
            "reason": "official_repo_missing",
            "message": f"official OpenFWI checkout does not exist: {repo_path}",
            "repo_path": str(repo_path),
        }
    license_candidates = [path for name in ("LICENSE", "LICENSE.txt") if (path := repo_path / name).exists()]
    base_payload: dict[str, Any] = {
        "repo_path": str(repo_path),
        "git_commit": _git_commit(repo_path),
        "license_path": str(license_candidates[0]) if license_candidates else None,
    }
    import_attempts = []
    for module_path in [repo_path / "network.py", repo_path / "model.py", repo_path / "models.py"]:
        if not module_path.exists():
            continue
        try:
            module = _load_module_from_path(module_path, repo_path=repo_path)
        except Exception as exc:
            import_attempts.append(
                {
                    "module_path": str(module_path),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue
        model_cls = _find_inversionnet(module)
        if model_cls is None:
            import_attempts.append(
                {
                    "module_path": str(module_path),
                    "status": "imported",
                    "message": "module imported but no InversionNet class or model_dict entry was found",
                }
            )
            continue
        import_attempt = {
            "module_path": str(module_path),
            "status": "imported",
            "class_name": model_cls.__name__,
        }
        try:
            forward_pass = _probe_forward(model_cls)
        except Exception as exc:
            return {
                **base_payload,
                "status": "blocked",
                "reason": "official_forward_probe_failed",
                "message": str(exc),
                "import_attempt": import_attempt,
                "import_attempts": import_attempts + [import_attempt],
                "forward_pass": {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        return {
            **base_payload,
            "status": "compatible",
            "import_attempt": import_attempt,
            "import_attempts": import_attempts + [import_attempt],
            "forward_pass": forward_pass,
        }
    if import_attempts:
        return {
            **base_payload,
            "status": "blocked",
            "reason": "official_import_failed",
            "message": "no importable official InversionNet implementation was found",
            "import_attempts": import_attempts,
        }
    return {
        **base_payload,
        "status": "blocked",
        "reason": "official_inversionnet_not_found",
        "message": "no network.py, model.py, or models.py file was found in the external checkout",
        "import_attempts": [],
    }
