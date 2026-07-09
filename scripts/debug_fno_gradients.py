import argparse
import json
from pathlib import Path

import torch

from ptycho.log_config import setup_logging
from ptycho_torch.generators.fno import PtychoBlock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FNO gradient diagnostic")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    torch.manual_seed(args.seed)
    block = PtychoBlock(channels=args.channels, modes=args.modes)
    block.train()

    x = torch.randn(2, args.channels, 32, 32, requires_grad=True)
    y = block(x)
    loss = y.abs().mean()
    loss.backward()

    spectral_grad = block.spectral.weights.grad.abs().mean().item()
    local_grad = block.local_conv.weight.grad.abs().mean().item()
    ratio = spectral_grad / local_grad if local_grad else None

    report = {
        "spectral_grad_mean": spectral_grad,
        "local_grad_mean": local_grad,
        "spectral_local_ratio": ratio,
    }
    (out_dir / "gradient_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
