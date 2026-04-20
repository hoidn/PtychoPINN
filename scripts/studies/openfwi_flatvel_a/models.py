"""Tiny local model adapters and official InversionNet probe for OpenFWI."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptycho_torch.generators.fno import SpatialLifter
from ptycho_torch.generators.hybrid_resnet import HybridResnetEncoderBlock
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
    """Small supervised adapter using existing Hybrid ResNet building blocks."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, modes: int, blocks: int):
        super().__init__()
        self.lifter = SpatialLifter(in_channels, hidden_channels)
        self.blocks = nn.Sequential(
            *[HybridResnetEncoderBlock(hidden_channels, modes=modes) for _ in range(max(1, int(blocks)))]
        )
        self.local = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.local(self.blocks(self.lifter(x)))


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
    profile = get_model_profile(profile_id)
    config = {**profile.to_model_config(), **dict(profile_config or {})}
    hidden_channels = int(config.get("hidden_channels") or 8)
    if profile.base_model == "hybrid_resnet":
        return PadCropWrapper(
            HybridResnetSmoke(
                in_channels,
                out_channels,
                hidden_channels,
                modes=int(config.get("fno_modes") or 4),
                blocks=int(config.get("fno_blocks") or 2),
            )
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
    return {
        "profile_id": profile_id,
        "class_name": model.__class__.__name__,
        "parameter_count": count_parameters(model),
        "profile_config": dict(profile_config),
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
    result = {
        "status": "blocked",
        "reason": "official_probe_not_implemented",
        "message": "external checkout exists, but this smoke gate records compatibility only after a controlled import/forward probe",
        "repo_path": str(repo_path),
        "git_commit": _git_commit(repo_path),
        "license_path": str(license_candidates[0]) if license_candidates else None,
    }
    return result
