"""Hybrid ResNet schematic manifest extraction utilities."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import torch

from ptycho_torch.generators.hybrid_resnet import HybridResnetGeneratorModule


def _shape_payload(value: Any) -> Any:
    """Return a JSON-friendly shape payload for tensor-like values."""
    if isinstance(value, torch.Tensor):
        return [int(v) for v in value.shape]
    if isinstance(value, (list, tuple)):
        payload = [_shape_payload(v) for v in value]
        payload = [p for p in payload if p is not None]
        return payload if payload else None
    return None


def _primary_shape(value: Any) -> list[int] | None:
    """Extract the first tensor shape from nested inputs/outputs."""
    payload = _shape_payload(value)
    if payload is None:
        return None
    if payload and isinstance(payload[0], int):
        return payload
    if payload and isinstance(payload[0], list):
        return payload[0]
    return None


def _make_node_record(
    *,
    name: str,
    module: torch.nn.Module,
    input_shape: list[int] | None,
    output_shape: list[int] | None,
) -> dict[str, Any]:
    return {
        "name": name,
        "module_class": module.__class__.__name__,
        "input_shape": input_shape,
        "output_shape": output_shape,
    }


def build_hybrid_resnet_manifest(
    *,
    N: int = 64,
    gridsize: int = 2,
    fno_width: int = 32,
    fno_blocks: int = 4,
    fno_modes: int = 12,
    resnet_width: int | None = None,
    output_mode: str = "real_imag",
    max_hidden_channels: int | None = None,
    resnet_blocks: int = 6,
    input_transform: str = "none",
    batch_size: int = 1,
) -> dict[str, Any]:
    """Build shape/flow manifest for HybridResnetGeneratorModule."""
    C = int(gridsize) * int(gridsize)
    model = HybridResnetGeneratorModule(
        in_channels=1,
        out_channels=2,
        hidden_channels=int(fno_width),
        n_blocks=int(fno_blocks),
        modes=int(fno_modes),
        C=C,
        input_transform=input_transform,
        output_mode=output_mode,
        max_hidden_channels=max_hidden_channels,
        resnet_blocks=int(resnet_blocks),
        resnet_width=resnet_width,
    )
    model.eval()

    tracked: list[tuple[str, torch.nn.Module]] = [("lifter", model.lifter)]
    tracked.extend((f"encoder_block_{i}", block) for i, block in enumerate(model.encoder_blocks))
    tracked.extend((f"downsample_{i}", down) for i, down in enumerate(model.downsample))
    tracked.append(("adapter", model.adapter))
    tracked.append(("resnet", model.resnet))
    if hasattr(model, "up1") and hasattr(model, "up2"):
        tracked.append(("up1", model.up1))
        tracked.append(("up2", model.up2))
    elif hasattr(model, "upsample_layers"):
        for i, upsample in enumerate(model.upsample_layers):
            tracked.append((f"up{i + 1}", upsample))
    if output_mode == "amp_phase":
        tracked.append(("output_amp", model.output_amp))
        tracked.append(("output_phase", model.output_phase))
    else:
        tracked.append(("output_proj", model.output_proj))

    seen_records: dict[str, dict[str, Any]] = {}
    call_order: list[str] = []
    handles: list[Any] = []

    def _register_hook(name: str, module: torch.nn.Module) -> None:
        def _hook(_module: torch.nn.Module, inputs: Any, outputs: Any) -> None:
            call_order.append(name)
            seen_records[name] = _make_node_record(
                name=name,
                module=module,
                input_shape=_primary_shape(inputs),
                output_shape=_primary_shape(outputs),
            )

        handles.append(module.register_forward_hook(_hook))

    for node_name, node_module in tracked:
        _register_hook(node_name, node_module)

    x = torch.randn(batch_size, C, int(N), int(N))
    with torch.no_grad():
        forward_out = model(x)

    for handle in handles:
        handle.remove()

    dedup_call_order = list(dict.fromkeys(call_order))
    nodes = [seen_records[name] for name in dedup_call_order]
    edges = [{"src": src, "dst": dst} for src, dst in pairwise(dedup_call_order)]

    tensor_contract: dict[str, Any] = {"input": [batch_size, C, int(N), int(N)]}
    if output_mode == "amp_phase":
        amp, phase = forward_out
        tensor_contract["output"] = {
            "amp": [int(v) for v in amp.shape],
            "phase": [int(v) for v in phase.shape],
        }
    else:
        tensor_contract["output"] = [int(v) for v in forward_out.shape]

    stage_shapes = {
        name: record.get("output_shape")
        for name, record in seen_records.items()
    }

    return {
        "architecture": "hybrid_resnet",
        "config": {
            "N": int(N),
            "gridsize": int(gridsize),
            "C": C,
            "fno_width": int(fno_width),
            "fno_blocks": int(fno_blocks),
            "fno_modes": int(fno_modes),
            "resnet_blocks": int(resnet_blocks),
            "resnet_width": resnet_width,
            "max_hidden_channels": max_hidden_channels,
            "input_transform": input_transform,
            "output_mode": output_mode,
        },
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "nodes": nodes,
        "edges": edges,
        "execution_order": dedup_call_order,
        "tensor_contract": tensor_contract,
        "stage_shapes": stage_shapes,
    }
