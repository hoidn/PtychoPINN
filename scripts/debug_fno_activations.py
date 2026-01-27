import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from ptycho.log_config import setup_logging
from ptycho_torch.generators.fno import CascadedFNOGenerator, HybridUNOGenerator, PtychoBlock, SpatialLifter


class ActivationMonitor:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.stats: Dict[str, List[float]] = defaultdict(list)
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.branch_norms: Dict[str, Dict[str, float]] = defaultdict(dict)

    def _record(self, key: str, value: float) -> None:
        self.stats[key].append(float(value))

    def _tensor_stats(self, name: str, tensor: torch.Tensor) -> None:
        values = tensor.detach().abs().flatten()
        if values.numel() == 0:
            return
        self._record(f"{name}/mean", values.mean().item())
        self._record(f"{name}/std", values.std().item())
        self._record(f"{name}/p50", torch.quantile(values, 0.5).item())
        self._record(f"{name}/p95", torch.quantile(values, 0.95).item())
        self._record(f"{name}/p99", torch.quantile(values, 0.99).item())

    def _low_freq_ratio(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.ndim != 4:
            return
        x_ft = torch.fft.rfft2(tensor.detach(), norm="ortho")
        power = x_ft.real.pow(2) + x_ft.imag.pow(2)
        h = power.shape[-2]
        w = power.shape[-1]
        low_h = max(1, h // 8)
        low_w = max(1, w // 8)
        low = power[..., :low_h, :low_w].mean()
        total = power.mean()
        if total > 0:
            self._record(f"{name}/low_freq_ratio", (low / total).item())

    def _maybe_record_ratio(self, prefix: str) -> None:
        spectral = self.branch_norms[prefix].get("spectral")
        local = self.branch_norms[prefix].get("local")
        if spectral is None or local is None or local == 0:
            return
        self._record(f"{prefix}/spectral_local_ratio", spectral / local)

    def _hook_fn(self, module, inputs, output, name: str) -> None:
        if not isinstance(output, torch.Tensor):
            return
        self._tensor_stats(name, output)
        self._low_freq_ratio(name, output)
        if isinstance(module, (torch.nn.ReLU, torch.nn.GELU)):
            sparsity = (output <= 0).float().mean().item()
            self._record(f"{name}/sparsity", sparsity)

        if name.endswith(".spectral"):
            prefix = name.rsplit(".", 1)[0]
            self.branch_norms[prefix]["spectral"] = output.detach().abs().mean().item()
            self._maybe_record_ratio(prefix)
        elif name.endswith(".local_conv"):
            prefix = name.rsplit(".", 1)[0]
            self.branch_norms[prefix]["local"] = output.detach().abs().mean().item()
            self._maybe_record_ratio(prefix)

    def register(self) -> None:
        for module_name, module in self.model.named_modules():
            if isinstance(module, (SpatialLifter, PtychoBlock)):
                hook = module.register_forward_hook(
                    lambda m, i, o, n=module_name: self._hook_fn(m, i, o, n)
                )
                self.hooks.append(hook)
            elif module_name.endswith(".spectral") or module_name.endswith(".local_conv"):
                hook = module.register_forward_hook(
                    lambda m, i, o, n=module_name: self._hook_fn(m, i, o, n)
                )
                self.hooks.append(hook)

    def close(self) -> None:
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.branch_norms.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture FNO activation statistics")
    parser.add_argument("--input", required=True, help="Path to train.npz")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--architecture", default="fno", choices=["fno", "hybrid"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _load_diffraction(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if "diffraction" not in data:
            raise KeyError("NPZ missing diffraction array")
        diffraction = np.asarray(data["diffraction"], dtype=np.float32)
    if diffraction.ndim == 3:
        diffraction = diffraction[..., np.newaxis]
    if diffraction.ndim != 4:
        raise ValueError(f"Expected diffraction with 3 or 4 dims, got {diffraction.shape}")
    return diffraction


def _build_model(architecture: str, n: int, channels: int) -> torch.nn.Module:
    modes = min(12, max(1, n // 4))
    if architecture == "hybrid":
        return HybridUNOGenerator(
            in_channels=1,
            out_channels=2,
            hidden_channels=32,
            n_blocks=4,
            modes=modes,
            C=channels,
        )
    return CascadedFNOGenerator(
        in_channels=1,
        out_channels=2,
        hidden_channels=32,
        fno_blocks=4,
        cnn_blocks=2,
        modes=modes,
        C=channels,
    )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    diffraction = _load_diffraction(Path(args.input))
    n_samples = min(args.max_samples, diffraction.shape[0])
    batch_size = min(args.batch_size, n_samples)

    batch = diffraction[:batch_size]
    n = batch.shape[1]
    channels = batch.shape[-1]

    model = _build_model(args.architecture, n, channels)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    monitor = ActivationMonitor(model)
    monitor.register()

    with torch.no_grad():
        tensor_batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        _ = model(tensor_batch)

    monitor.close()

    report = {
        "summary": {
            "architecture": args.architecture,
            "n_samples": int(n_samples),
            "batch_size": int(batch_size),
            "channels": int(channels),
        },
        "layers": {key: values for key, values in monitor.stats.items()},
    }

    report_path = out_dir / "activation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
