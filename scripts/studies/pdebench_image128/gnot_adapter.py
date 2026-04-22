"""Thin adapter that hosts the official GNOT model under the CNS image-suite contract."""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class GnotAdapterBuildError(RuntimeError):
    """Raised when the external GNOT source or its dependencies are unavailable."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = str(reason)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_gnot_root() -> Path:
    candidate = os.environ.get("GNOT_ROOT")
    root = Path(candidate) if candidate else (_repo_root() / ".artifacts" / "external" / "gnot")
    if not root.exists():
        raise GnotAdapterBuildError(
            "model_dependency_unavailable",
            f"GNOT source root is unavailable: {root}",
        )
    return root


@contextmanager
def _prepend_sys_path(path: Path):
    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:  # pragma: no cover - defensive cleanup
            pass


def _source_provenance(root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "external_repo": "https://github.com/HaoZhongkai/GNOT",
        "external_root": str(root.resolve()),
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


def load_gnot_dependencies():
    os.environ.setdefault("DGLBACKEND", "pytorch")
    try:
        import dgl  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise GnotAdapterBuildError(
            "model_dependency_unavailable",
            f"dgl is unavailable for the GNOT adapter: {exc}",
        ) from exc

    root = resolve_gnot_root()
    with _prepend_sys_path(root):
        try:
            from models.mmgpt import GNOT  # type: ignore
            from utils import MultipleTensors  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise GnotAdapterBuildError(
                "model_dependency_unavailable",
                f"GNOT import failed from {root}: {exc}",
            ) from exc
    return dgl, GNOT, MultipleTensors, _source_provenance(root)


def _regular_center_grid(spatial_shape: tuple[int, int], *, dx: float, dy: float) -> torch.Tensor:
    height, width = (int(spatial_shape[0]), int(spatial_shape[1]))
    x = torch.arange(height, dtype=torch.float32) * float(dx) + float(dx) / 2.0
    y = torch.arange(width, dtype=torch.float32) * float(dy) + float(dy) / 2.0
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)


class GnotCnsModel(nn.Module):
    """Wrap the official GNOT model behind a `B,C,H,W -> B,C,H,W` interface."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        spatial_shape: tuple[int, int],
        task_metadata: dict[str, Any],
        n_hidden: int = 64,
        n_layers: int = 3,
        n_head: int = 1,
        n_experts: int = 1,
        n_inner: int = 4,
        mlp_layers: int = 3,
        attn_type: str = "linear",
    ):
        super().__init__()
        if str(task_metadata.get("task_id")) != "2d_cfd_cns":
            raise ValueError("GnotCnsModel requires task_metadata['task_id'] == '2d_cfd_cns'")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.spatial_shape = (int(spatial_shape[0]), int(spatial_shape[1]))
        self.num_nodes = int(self.spatial_shape[0] * self.spatial_shape[1])
        self.dgl, gnot_cls, self._multiple_tensors_cls, provenance = load_gnot_dependencies()
        self.external_source_provenance = provenance
        self.model = gnot_cls(
            trunk_size=2,
            branch_sizes=[2 + self.in_channels],
            output_size=self.out_channels,
            n_layers=int(n_layers),
            n_hidden=int(n_hidden),
            n_head=int(n_head),
            space_dim=2,
            n_experts=int(n_experts),
            n_inner=int(n_inner),
            mlp_layers=int(mlp_layers),
            attn_type=str(attn_type),
        )
        self.register_buffer(
            "_query_points",
            _regular_center_grid(
                self.spatial_shape,
                dx=float(task_metadata["dx"]),
                dy=float(task_metadata["dy"]),
            ),
            persistent=False,
        )
        self._graph_cache: dict[tuple[int, str], Any] = {}

    def _graph_for_batch(self, batch_size: int, device: torch.device):
        key = (int(batch_size), str(device))
        graph = self._graph_cache.get(key)
        if graph is None:
            graph = self.dgl.batch([self.dgl.graph(([], []), num_nodes=self.num_nodes) for _ in range(batch_size)]).to(device)
            self._graph_cache[key] = graph
        return graph

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        if (int(height), int(width)) != self.spatial_shape:
            raise ValueError(
                f"GnotCnsModel expected spatial_shape={self.spatial_shape}, got {(int(height), int(width))}"
            )
        coords = self._query_points.to(device=x.device, dtype=x.dtype)
        query_points = coords.unsqueeze(0).expand(batch_size, -1, -1)
        state_points = x.permute(0, 2, 3, 1).reshape(batch_size, self.num_nodes, self.in_channels)
        branch_inputs = torch.cat([query_points, state_points], dim=-1)
        graph = self._graph_for_batch(batch_size, x.device)
        graph.ndata["x"] = query_points.reshape(batch_size * self.num_nodes, 2)
        global_params = torch.zeros(batch_size, 0, device=x.device, dtype=x.dtype)
        outputs = self.model(graph, global_params, self._multiple_tensors_cls([branch_inputs]))
        return outputs.view(batch_size, self.spatial_shape[0], self.spatial_shape[1], self.out_channels).permute(0, 3, 1, 2).contiguous()
