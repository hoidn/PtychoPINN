"""Lightweight search interface for PtychoPINN experiment runs.

Usage:
    python -m ptycho_torch.experiment_search /path/to/outputs
    python -m ptycho_torch.experiment_search /path/to/outputs --experiment CCNF_Runs --dataset pinn_velo
    python -m ptycho_torch.experiment_search /path/to/outputs --notes "fno" --format json
"""

import argparse
import json
from pathlib import Path
from typing import Optional


def _load_json_safe(path: Path) -> Optional[dict]:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _collect_run_info(run_dir: Path, experiment_name: str) -> Optional[dict]:
    """Read metadata and key config fields from a single run directory."""
    metadata = _load_json_safe(run_dir / "metadata.json")
    if metadata is None:
        return None

    info = {
        "experiment_name": experiment_name,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
    }
    info.update(metadata)

    config_dir = run_dir / "configs"
    model_cfg = _load_json_safe(config_dir / "model_config.json")
    if model_cfg:
        info["architecture"] = model_cfg.get("architecture", "")
        info["encoder_type"] = model_cfg.get("encoder_type", "")
        info["loss_function"] = model_cfg.get("loss_function", "")

    training_cfg = _load_json_safe(config_dir / "training_config.json")
    if training_cfg:
        info["learning_rate"] = training_cfg.get("learning_rate")
        info["epochs"] = training_cfg.get("epochs")
        info["batch_size"] = training_cfg.get("batch_size")

    return info


def search_runs(
    output_dir: str,
    experiment_name: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    notes_contains: Optional[str] = None,
    status: Optional[str] = None,
    model_name: Optional[str] = None,
    architecture: Optional[str] = None,
    max_val_loss: Optional[float] = None,
) -> list:
    """Search experiment runs by filtering metadata.

    Walks output_dir for the expected hierarchy:
        output_dir/experiment_name/run_name/metadata.json

    Args:
        output_dir: Root output directory containing experiment folders.
        experiment_name: Filter by experiment name (exact match).
        dataset_dir: Filter by dataset directory (substring match on path).
        notes_contains: Filter runs whose notes contain this substring (case-insensitive).
        status: Filter by run status ('completed', 'running', 'failed').
        model_name: Filter by model name (exact match).
        architecture: Filter by architecture (exact match, from model_config.json).
        max_val_loss: Only include runs with best_val_loss <= this value.

    Returns:
        List of dicts with metadata, config hyperparams, and directory info,
        sorted by start_time descending.
    """
    root = Path(output_dir)
    if not root.is_dir():
        return []

    if experiment_name:
        experiment_dirs = [root / experiment_name]
    else:
        experiment_dirs = [d for d in root.iterdir() if d.is_dir()]

    results = []
    for exp_dir in experiment_dirs:
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if not (run_dir / "metadata.json").exists():
                continue

            info = _collect_run_info(run_dir, exp_name)
            if info is None:
                continue

            if dataset_dir and dataset_dir not in info.get("dataset_dir", ""):
                continue
            if notes_contains and notes_contains.lower() not in info.get("notes", "").lower():
                continue
            if status and info.get("status") != status:
                continue
            if model_name and info.get("model_name") != model_name:
                continue
            if architecture and info.get("architecture") != architecture:
                continue
            if max_val_loss is not None:
                val_loss = info.get("best_val_loss")
                if val_loss is None or val_loss > max_val_loss:
                    continue

            results.append(info)

    results.sort(key=lambda r: r.get("start_time", ""), reverse=True)
    return results


def _format_table(results: list) -> str:
    if not results:
        return "No runs found."

    header = f"{'Experiment':<25} {'Run':<35} {'Status':<10} {'Val Loss':<10} {'Arch':<12} {'Encoder':<10} {'Dataset':<30} {'Notes'}"
    sep = "-" * len(header)
    lines = [header, sep]

    for r in results:
        val_loss = r.get("best_val_loss")
        val_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        dataset = r.get("dataset_dir", "")
        if len(dataset) > 28:
            dataset = "..." + dataset[-25:]
        notes = r.get("notes", "")
        if len(notes) > 40:
            notes = notes[:37] + "..."

        lines.append(
            f"{r.get('experiment_name', ''):<25} "
            f"{r.get('run_name', ''):<35} "
            f"{r.get('status', '?'):<10} "
            f"{val_str:<10} "
            f"{r.get('architecture', ''):<12} "
            f"{r.get('encoder_type', ''):<10} "
            f"{dataset:<30} "
            f"{notes}"
        )

    lines.append(sep)
    lines.append(f"{len(results)} run(s) found.")
    return "\n".join(lines)


def cli_main():
    parser = argparse.ArgumentParser(
        description="Search PtychoPINN experiment runs",
    )
    parser.add_argument("output_dir", help="Root output directory")
    parser.add_argument("--experiment", "-e", help="Filter by experiment name")
    parser.add_argument("--dataset", "-d", help="Filter by dataset directory (substring)")
    parser.add_argument("--notes", "-n", help="Filter by notes (substring, case-insensitive)")
    parser.add_argument("--status", "-s", choices=["completed", "running", "failed"])
    parser.add_argument("--model", "-m", help="Filter by model name")
    parser.add_argument("--architecture", "-a", help="Filter by architecture")
    parser.add_argument("--max-loss", type=float, help="Maximum best_val_loss")
    parser.add_argument("--format", "-f", choices=["table", "json"], default="table")
    args = parser.parse_args()

    results = search_runs(
        output_dir=args.output_dir,
        experiment_name=args.experiment,
        dataset_dir=args.dataset,
        notes_contains=args.notes,
        status=args.status,
        model_name=args.model,
        architecture=args.architecture,
        max_val_loss=args.max_loss,
    )

    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        print(_format_table(results))


if __name__ == "__main__":
    cli_main()
