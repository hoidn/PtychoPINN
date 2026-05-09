from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OUTPUT_ROOT_RELATIVE = (
    ".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/"
    "2026-04-29-wavebench-shared-encoder-supervised-benchmark"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.exists(), f"missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


VALID_STATUSES = {
    "completed",
    "smoke_pass",
    "blocked",
    "not_protocol_compatible",
}
COMPLETION_LIKE_STATUSES = {"completed", "blocked", "not_protocol_compatible"}
LOCKED_ROWS = (
    "cnn",
    "hybrid_resnet",
    "spectral_resnet_bottleneck_net",
    "fno",
    "ffno",
)
LOCKED_LATENT_CHANNELS = (32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate WaveBench shared-encoder artifacts.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-root", default=OUTPUT_ROOT_RELATIVE)
    parser.add_argument(
        "--require-benchmark-completion",
        action="store_true",
        help=(
            "Require every row in the locked roster to have a benchmark-mode entry for"
            " each latent width that is either status=completed (with mode=benchmark)"
            " or carries an explicit row-level blocker."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_root = (
        Path(args.output_root).resolve()
        if Path(args.output_root).is_absolute()
        else (repo_root / args.output_root).resolve()
    )

    row_contract = load_json(output_root / "row_contract.json")
    manifest = load_json(output_root / "shared_encoder_execution_manifest.json")

    require(
        row_contract["selected_variant"] == "time_varying/is/thick_lines_gaussian_lens",
        "row_contract selected_variant drifted from the locked WaveBench contract",
    )
    require(
        row_contract["latent_channel_settings"] == [32, 64],
        "row_contract must preserve the required C=32/C=64 latent widths",
    )
    require(
        row_contract["rows"]
        == [
            "cnn",
            "hybrid_resnet",
            "spectral_resnet_bottleneck_net",
            "fno",
            "ffno",
        ],
        "row_contract must preserve the locked row roster",
    )
    require(
        manifest["selected_variant"] == row_contract["selected_variant"],
        "manifest selected_variant must match row_contract",
    )

    for row, channel_map in manifest.get("rows", {}).items():
        for channel_key, record in channel_map.items():
            status = record["status"]
            require(
                status in VALID_STATUSES,
                f"invalid row status for {row}/{channel_key}: {status}",
            )
            artifact_path = output_root / record["artifact_path"]
            if status in {"completed", "smoke_pass"}:
                metrics = load_json(artifact_path)
                require(metrics["row"] == row, f"metrics row mismatch for {artifact_path}")
                require(
                    metrics["latent_channels"] == int(channel_key.removeprefix("c")),
                    f"metrics latent width mismatch for {artifact_path}",
                )
                require(
                    metrics.get("mode") in {"smoke", "benchmark"},
                    f"metrics payload missing valid mode for {artifact_path}",
                )
                if status == "completed":
                    require(
                        metrics["mode"] == "benchmark",
                        f"status=completed requires mode=benchmark for {artifact_path}",
                    )
                if status == "smoke_pass":
                    require(
                        metrics["mode"] == "smoke",
                        f"status=smoke_pass requires mode=smoke for {artifact_path}",
                    )
                for metric_name in ("MAE", "RMSE", "RelL2", "SSIM"):
                    require(
                        metric_name in metrics["metrics"],
                        f"metrics payload missing {metric_name} for {artifact_path}",
                    )

    if args.require_benchmark_completion:
        rows = manifest.get("rows", {})
        for row in LOCKED_ROWS:
            for latent in LOCKED_LATENT_CHANNELS:
                channel_key = f"c{latent}"
                record = rows.get(row, {}).get(channel_key)
                require(
                    record is not None,
                    f"missing manifest entry for {row}/{channel_key} under benchmark-completion gate",
                )
                require(
                    record["status"] in COMPLETION_LIKE_STATUSES,
                    (
                        f"benchmark-completion gate requires {row}/{channel_key} to be one of "
                        f"{sorted(COMPLETION_LIKE_STATUSES)}, found status={record['status']}"
                    ),
                )

    print("wavebench shared-encoder contract validated")


if __name__ == "__main__":
    main()
