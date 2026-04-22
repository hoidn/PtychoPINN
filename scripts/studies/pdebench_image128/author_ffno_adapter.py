"""Thin adapter that hosts the official author FFNO model under the CNS image-suite contract."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class AuthorFfnoAdapterBuildError(RuntimeError):
    """Raised when the external author FFNO source or its dependencies are unavailable."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = str(reason)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_author_ffno_root() -> Path:
    candidate = os.environ.get("AUTHOR_FFNO_ROOT")
    root = Path(candidate) if candidate else (_repo_root() / ".artifacts" / "external" / "fourierflow")
    if not root.exists():
        raise AuthorFfnoAdapterBuildError(
            "model_dependency_unavailable",
            f"Author FFNO source root is unavailable: {root}",
        )
    return root


def _ensure_package(module_name: str, package_path: Path) -> None:
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)
    module.__path__ = [str(package_path)]
    sys.modules[module_name] = module


def _load_module_from_path(module_name: str, path: Path):
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AuthorFfnoAdapterBuildError(
            "model_dependency_unavailable",
            f"Unable to build import spec for author FFNO module at {path}",
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _manual_namespace(root: Path) -> str:
    digest = hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"author_ffno_source_{digest}"


def _manual_load_author_ffno_modules(root: Path):
    modules_root = root / "fourierflow" / "modules"
    grid_path = modules_root / "factorized_fno" / "grid_2d.py"
    if not grid_path.exists():
        raise AuthorFfnoAdapterBuildError(
            "model_dependency_unavailable",
            f"Author FFNO entrypoint is missing: {grid_path}",
        )

    namespace = _manual_namespace(root)
    _ensure_package(namespace, modules_root)
    _ensure_package(f"{namespace}.factorized_fno", modules_root / "factorized_fno")

    try:
        linear_module = _load_module_from_path(f"{namespace}.linear", modules_root / "linear.py")
        _load_module_from_path(f"{namespace}.feedforward", modules_root / "feedforward.py")
        grid_module = _load_module_from_path(f"{namespace}.factorized_fno.grid_2d", grid_path)
    except ModuleNotFoundError as exc:
        raise AuthorFfnoAdapterBuildError(
            "model_dependency_unavailable",
            f"Author FFNO module load failed from {root}: missing dependency {exc.name}",
        ) from exc
    except Exception as exc:
        raise AuthorFfnoAdapterBuildError(
            "model_dependency_unavailable",
            f"Author FFNO module load failed from {root}: {exc}",
        ) from exc

    return grid_module.FNOFactorized2DBlock, linear_module.WNLinear


def _host_environment_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "torch_version": torch.__version__,
    }
    try:
        import einops

        payload["einops_version"] = einops.__version__
    except Exception:
        payload["einops_version"] = None
    return payload


def _source_provenance(root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "external_repo": "https://github.com/alasdairtran/fourierflow",
        "external_root": str(root.resolve()),
        "entrypoint_module": "fourierflow.modules.factorized_fno.grid_2d",
        "entrypoint_class": "FNOFactorized2DBlock",
        "entrypoint_config": "experiments/torus_li/markov/24_layers/config.yaml",
        "author_model_rationale": (
            "Official paper repository for Factorized Fourier Neural Operators; "
            "uses the authored FNOFactorized2DBlock implementation rather than a local proxy bottleneck."
        ),
        "import_mode": "manual_submodule_load",
        "host_environment": _host_environment_payload(),
        "dependency_notes": {
            "full_package_import_requires": ["hydra", "omegaconf", "xarray"],
            "adapter_runtime_requires": ["torch", "einops"],
            "manual_load_reason": (
                "The ptycho311 environment can host the authored module subtree directly, "
                "but it does not provide the full top-level fourierflow CLI stack."
            ),
        },
    }
    try:
        payload["external_commit"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # pragma: no cover - best effort provenance only
        pass
    return payload


def load_author_ffno_dependencies():
    root = resolve_author_ffno_root()
    ffno_cls, wnlinear_cls = _manual_load_author_ffno_modules(root)
    return ffno_cls, wnlinear_cls, _source_provenance(root)


def build_author_ffno_source_payload() -> dict[str, Any]:
    _, _, provenance = load_author_ffno_dependencies()
    return provenance


def _regular_unit_grid(spatial_shape: tuple[int, int]) -> torch.Tensor:
    height, width = (int(spatial_shape[0]), int(spatial_shape[1]))
    y = torch.linspace(0.0, 1.0, steps=height, dtype=torch.float32)
    x = torch.linspace(0.0, 1.0, steps=width, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([yy, xx], dim=-1)


class AuthorFfnoCnsModel(nn.Module):
    """Wrap the official author FFNO grid model behind a `B,C,H,W -> B,C,H,W` interface."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        spatial_shape: tuple[int, int],
        task_metadata: dict[str, Any],
        modes: int = 16,
        width: int = 64,
        n_layers: int = 24,
        share_weight: bool = True,
        factor: int = 4,
        ff_weight_norm: bool = True,
        n_ff_layers: int = 2,
        gain: float = 0.1,
        dropout: float = 0.0,
        in_dropout: float = 0.0,
        layer_norm: bool = False,
        use_position: bool = True,
        mode: str = "full",
    ):
        super().__init__()
        if str(task_metadata.get("task_id")) != "2d_cfd_cns":
            raise ValueError("AuthorFfnoCnsModel requires task_metadata['task_id'] == '2d_cfd_cns'")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.spatial_shape = (int(spatial_shape[0]), int(spatial_shape[1]))
        self.use_position = bool(use_position)
        input_dim = self.in_channels + (2 if self.use_position else 0)

        ffno_cls, wnlinear_cls, provenance = load_author_ffno_dependencies()
        self.external_source_provenance = {
            **provenance,
            "local_contract_adaptation": {
                "task_id": "2d_cfd_cns",
                "input_layout": "B,C,H,W -> B,H,W,C",
                "coordinate_features": "append unit-interval y/x channels",
                "output_head": (
                    "replace the authored scalar output projection with the same authored WNLinear head "
                    "widened to the local multi-field CNS target count"
                ),
            },
        }
        self.model = ffno_cls(
            modes=int(modes),
            width=int(width),
            input_dim=int(input_dim),
            dropout=float(dropout),
            in_dropout=float(in_dropout),
            n_layers=int(n_layers),
            share_weight=bool(share_weight),
            factor=int(factor),
            ff_weight_norm=bool(ff_weight_norm),
            n_ff_layers=int(n_ff_layers),
            gain=float(gain),
            layer_norm=bool(layer_norm),
            mode=str(mode),
        )
        self.model.out = nn.Sequential(
            wnlinear_cls(int(width), 128, wnorm=bool(ff_weight_norm)),
            wnlinear_cls(128, self.out_channels, wnorm=bool(ff_weight_norm)),
        )
        self.register_buffer(
            "_position_features",
            _regular_unit_grid(self.spatial_shape),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"AuthorFfnoCnsModel expected {self.in_channels} input channels, got {channels}")
        if (int(height), int(width)) != self.spatial_shape:
            raise ValueError(
                f"AuthorFfnoCnsModel expected spatial_shape={self.spatial_shape}, got {(int(height), int(width))}"
            )
        features = x.permute(0, 2, 3, 1)
        if self.use_position:
            coords = self._position_features.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(batch_size, -1, -1, -1)
            features = torch.cat([features, coords], dim=-1)
        output = self.model(features)["forecast"]
        return output.permute(0, 3, 1, 2).contiguous()
