"""CLI for rendering model-agnostic PDEBench CNS rollout videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.studies.pdebench_image128.cns_rollout import autoregressive_rollout
from scripts.studies.pdebench_image128.cns_rollout_data import load_cns_trajectory_window
from scripts.studies.pdebench_image128.cns_rollout_render import render_all_field_rollouts, render_field_rollout_gif


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--start-time", type=int)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--field", default="density")
    parser.add_argument("--include-error", action="store_true")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--format", choices=["gif"], default="gif")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _manifest_path(output_root: Path, row_id: str, sample_id: int) -> Path:
    return output_root / f"{row_id}_sample{int(sample_id):03d}_rollout_manifest.json"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    window = load_cns_trajectory_window(
        run_root=args.run_root,
        split=args.split,
        sample_id=args.sample_id,
        start_time=args.start_time,
        steps=args.steps,
        data_file=args.data_file,
    )
    checkpoint = args.checkpoint_path or args.run_root / f"model_state_{args.row_id}.pt"
    base_manifest = {
        "schema_version": "pdebench_cns_rollout_video_manifest_v1",
        "row_id": args.row_id,
        "run_root": str(args.run_root),
        "checkpoint_path": str(checkpoint),
        "checkpoint_exists": bool(checkpoint.exists()),
        "requires_checkpoint": True,
        "split": args.split,
        "sample_id": int(args.sample_id),
        "trajectory_id": int(window.trajectory_id),
        "start_time": int(window.start_time),
        "steps": int(args.steps),
        "field": args.field,
        "field_order": list(window.field_order),
        "format": args.format,
        "include_error": bool(args.include_error),
        "fps": float(args.fps),
        "device": args.device,
    }
    if args.dry_run:
        _write_json(_manifest_path(args.output_root, args.row_id, args.sample_id), base_manifest)
        return 0
    from scripts.studies.pdebench_image128.cns_rollout_models import (
        MissingCnsCheckpointError,
        load_cns_predictor,
    )

    try:
        predictor = load_cns_predictor(
            run_root=args.run_root,
            row_id=args.row_id,
            checkpoint_path=args.checkpoint_path,
            device=args.device,
        )
    except MissingCnsCheckpointError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    state_stats = json.loads((args.run_root / "normalization_stats_state.json").read_text(encoding="utf-8"))
    result = autoregressive_rollout(window=window, predictor=predictor, state_stats=state_stats)
    stem = f"{args.row_id}_sample{int(args.sample_id):03d}"
    if args.field == "all":
        outputs = render_all_field_rollouts(
            result=result,
            output_root=args.output_root,
            stem=stem,
            fps=args.fps,
            include_error=args.include_error,
        )
    else:
        outputs = [
            render_field_rollout_gif(
                result=result,
                field=args.field,
                output_path=args.output_root / f"{stem}_{args.field}_rollout.gif",
                fps=args.fps,
                include_error=args.include_error,
            )
        ]
    _write_json(_manifest_path(args.output_root, args.row_id, args.sample_id), {**base_manifest, "outputs": [str(path) for path in outputs]})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
