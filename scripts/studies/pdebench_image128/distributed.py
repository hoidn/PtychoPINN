"""Distributed-runtime helpers for PDEBench image-suite study runners."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    return int(raw)


@dataclass
class DistributedRuntime:
    requested_device: str
    device: torch.device
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str | None = None
    distributed_enabled: bool = False
    launched_via_torchrun: bool = False

    @property
    def is_rank_zero(self) -> bool:
        return int(self.rank) == 0

    def barrier(self) -> None:
        if self.distributed_enabled:
            dist.barrier()

    def broadcast_object(self, value: Any, *, src: int = 0) -> Any:
        if not self.distributed_enabled:
            return value
        payload = [value if self.rank == src else None]
        dist.broadcast_object_list(payload, src=src)
        return payload[0]

    def reduce_sum(self, value: float) -> float:
        if not self.distributed_enabled:
            return float(value)
        tensor = torch.tensor(float(value), device=self.device if self.device.type == "cuda" else "cpu")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item())

    def reduce_sum_dict(self, values: Mapping[str, float]) -> dict[str, float]:
        payload = {str(key): float(value) for key, value in values.items()}
        if not payload or not self.distributed_enabled:
            return payload
        keys = sorted(payload)
        tensor = torch.tensor(
            [payload[key] for key in keys],
            dtype=torch.float32,
            device=self.device if self.device.type == "cuda" else "cpu",
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return {key: float(tensor[index].item()) for index, key in enumerate(keys)}

    def max_cuda_memory_bytes(self) -> int | None:
        if self.device.type != "cuda":
            return None
        value = int(torch.cuda.max_memory_allocated(self.device))
        if not self.distributed_enabled:
            return value
        tensor = torch.tensor(value, dtype=torch.int64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return int(tensor.item())

    def maybe_reset_peak_memory_stats(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def build_training_loader(
        self,
        dataset,
        *,
        batch_size: int,
        num_workers: int,
        collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
        shuffle: bool = False,
    ) -> tuple[DataLoader, DistributedSampler | None]:
        sampler = None
        if self.distributed_enabled:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                drop_last=False,
            )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=bool(shuffle) if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        return loader, sampler

    def training_runtime_payload(self) -> dict[str, Any]:
        return {
            "requested_device": str(self.requested_device),
            "resolved_device": str(self.device),
            "distributed_enabled": bool(self.distributed_enabled),
            "distributed_world_size": int(self.world_size),
            "distributed_rank": int(self.rank),
            "distributed_local_rank": int(self.local_rank),
            "distributed_backend": self.backend,
            "launched_via_torchrun": bool(self.launched_via_torchrun),
        }

    def finalize(self) -> None:
        if self.distributed_enabled and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def initialize_runtime(requested_device: str) -> DistributedRuntime:
    requested_device = str(requested_device)
    launched_via_torchrun = any(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    world_size = max(1, _env_int("WORLD_SIZE", 1))

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            if world_size > 1:
                raise RuntimeError("multi-process PDEBench run requested --device cuda but CUDA is unavailable")
            device = torch.device("cpu")
        else:
            device_index = local_rank if world_size > 1 else 0
            torch.cuda.set_device(device_index)
            device = torch.device("cuda", device_index)
    else:
        device = torch.device(requested_device)

    distributed_enabled = world_size > 1
    backend = None
    if dist.is_initialized():
        distributed_enabled = dist.get_world_size() > 1
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        backend = dist.get_backend()
    elif distributed_enabled:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    return DistributedRuntime(
        requested_device=requested_device,
        device=device,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
        distributed_enabled=distributed_enabled,
        launched_via_torchrun=launched_via_torchrun,
    )


def prepare_output_root(
    output_root: Path,
    *,
    allow_existing: bool,
    runtime: DistributedRuntime,
) -> None:
    output_root = Path(output_root)
    error_payload = None
    if runtime.is_rank_zero:
        try:
            if output_root.exists() and any(output_root.iterdir()) and not allow_existing:
                raise FileExistsError(f"refusing to write into non-empty output root: {output_root}")
            output_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - exercised through broadcast on worker ranks
            error_payload = {"type": exc.__class__.__name__, "message": str(exc)}
    error_payload = runtime.broadcast_object(error_payload, src=0)
    if error_payload is not None:
        if error_payload.get("type") == "FileExistsError":
            raise FileExistsError(str(error_payload.get("message", "output-root preparation failed")))
        raise RuntimeError(str(error_payload.get("message", "output-root preparation failed")))
    runtime.barrier()
    if not runtime.is_rank_zero:
        output_root.mkdir(parents=True, exist_ok=True)
