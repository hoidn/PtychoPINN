#!/usr/bin/env python3
"""CLI wrapper for grid_lines_workflow."""

import argparse
from dataclasses import replace
import os
import sys
from pathlib import Path

from ptycho.config import load_simulation_config, validate_simulation_config
from ptycho.simulation.probe_transform import (
    parse_probe_transform_pipeline,
    serialize_probe_transform_pipeline,
)
from ptycho.workflows.grid_lines_workflow import GridLinesConfig, run_grid_lines_workflow


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-config", type=Path)
    parser.add_argument("--N", type=int, choices=[64, 128])
    parser.add_argument("--gridsize", type=int, choices=[1, 2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-npz", type=Path)
    parser.add_argument("--nimgs-train", type=int)
    parser.add_argument("--nimgs-test", type=int)
    parser.add_argument("--nphotons", type=float)
    parser.add_argument("--nepochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nll-weight", type=float, default=0.0)
    parser.add_argument("--mae-weight", type=float, default=1.0)
    parser.add_argument("--realspace-weight", type=float, default=0.0)
    parser.add_argument("--probe-smoothing-sigma", type=float)
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument("--probe-mask-diameter", type=int)
    mask_group.add_argument("--no-probe-mask", action="store_true")
    parser.add_argument("--probe-transform-pipeline")
    parser.add_argument(
        "--probe-source",
        choices=["custom", "ideal_disk"],
        default=None,
        help="Probe source for grid-lines datasets.",
    )
    parser.add_argument(
        "--probe-scale-mode",
        choices=["pad_preserve", "pad_extrapolate", "interpolate"],
        default=None,
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> GridLinesConfig:
    if args.probe_npz is not None and args.probe_source == "ideal_disk":
        raise ValueError(
            "--probe-npz conflicts with --probe-source=ideal_disk"
        )
    if args.probe_transform_pipeline is not None:
        if args.probe_scale_mode is not None:
            raise ValueError(
                "--probe-transform-pipeline conflicts with --probe-scale-mode"
            )
        if args.probe_smoothing_sigma is not None:
            raise ValueError(
                "--probe-transform-pipeline conflicts with --probe-smoothing-sigma"
            )
    if args.simulation_config is not None:
        simulation = load_simulation_config(args.simulation_config)
        if args.N is not None and args.N != simulation.N:
            resized_steps = [
                ({**step, "target_N": args.N} if "target_N" in step else step)
                for step in parse_probe_transform_pipeline(
                    simulation.probe.transform_pipeline
                )
            ]
            simulation = replace(
                simulation,
                N=args.N,
                probe=replace(
                    simulation.probe,
                    transform_pipeline=serialize_probe_transform_pipeline(
                        resized_steps
                    ),
                ),
            )
        if args.gridsize is not None:
            simulation = replace(
                simulation,
                scan=replace(
                    simulation.scan,
                    grid_size=(args.gridsize, args.gridsize),
                ),
            )
        if args.probe_npz is not None:
            simulation = replace(
                simulation,
                probe=replace(
                    simulation.probe,
                    source="custom",
                    source_path=args.probe_npz,
                ),
            )
        if args.probe_source is not None:
            source = "ideal" if args.probe_source == "ideal_disk" else "custom"
            simulation = replace(
                simulation,
                probe=replace(
                    simulation.probe,
                    source=source,
                    source_path=(
                        None if source == "ideal" else simulation.probe.source_path
                    ),
                ),
            )
        if args.probe_transform_pipeline is not None:
            simulation = replace(
                simulation,
                probe=replace(
                    simulation.probe,
                    transform_pipeline=args.probe_transform_pipeline,
                ),
            )
        elif args.probe_scale_mode is not None or args.probe_smoothing_sigma is not None:
            mode = args.probe_scale_mode or "pad_preserve"
            sigma = (
                args.probe_smoothing_sigma
                if args.probe_smoothing_sigma is not None
                else 0.5
            )
            smooth = f"smooth:{sigma:g}"
            if mode == "pad_preserve":
                steps = ([smooth] if sigma > 0 else []) + [f"pad_preserve:{simulation.N}"]
            elif mode == "pad_extrapolate":
                steps = [f"pad_extrapolate:{simulation.N}"] + ([smooth] if sigma > 0 else [])
            else:
                steps = [f"interp:{simulation.N}"] + ([smooth] if sigma > 0 else [])
            simulation = replace(
                simulation,
                probe=replace(
                    simulation.probe,
                    transform_pipeline="|".join(steps),
                ),
            )
        if args.probe_mask_diameter is not None or args.no_probe_mask:
            simulation = replace(
                simulation,
                probe=replace(
                    simulation.probe,
                    mask_diameter=(
                        None if args.no_probe_mask else args.probe_mask_diameter
                    ),
                ),
            )
        if args.nimgs_train is not None or args.nimgs_test is not None:
            simulation = replace(
                simulation,
                scan=replace(
                    simulation.scan,
                    train_groups=(
                        args.nimgs_train
                        if args.nimgs_train is not None
                        else simulation.scan.train_groups
                    ),
                    test_groups=(
                        args.nimgs_test
                        if args.nimgs_test is not None
                        else simulation.scan.test_groups
                    ),
                ),
            )
        if args.nphotons is not None:
            simulation = replace(
                simulation,
                detector=replace(
                    simulation.detector,
                    photons_per_pattern=args.nphotons,
                ),
            )
        validate_simulation_config(simulation)
        return GridLinesConfig(
            output_dir=args.output_dir,
            simulation=simulation,
            nepochs=args.nepochs,
            batch_size=args.batch_size,
            nll_weight=args.nll_weight,
            mae_weight=args.mae_weight,
            realspace_weight=args.realspace_weight,
        )

    if args.N is None or args.gridsize is None:
        raise ValueError(
            "--N and --gridsize are required when --simulation-config is omitted"
        )
    return GridLinesConfig(
        N=args.N,
        gridsize=args.gridsize,
        output_dir=args.output_dir,
        probe_npz=(
            args.probe_npz
            or Path("datasets/Run1084_recon3_postPC_shrunk_3.npz")
        ),
        nimgs_train=args.nimgs_train if args.nimgs_train is not None else 2,
        nimgs_test=args.nimgs_test if args.nimgs_test is not None else 2,
        nphotons=args.nphotons if args.nphotons is not None else 1e9,
        nepochs=args.nepochs,
        batch_size=args.batch_size,
        nll_weight=args.nll_weight,
        mae_weight=args.mae_weight,
        realspace_weight=args.realspace_weight,
        probe_smoothing_sigma=(
            args.probe_smoothing_sigma
            if args.probe_smoothing_sigma is not None
            else 0.5
        ),
        probe_mask_diameter=(
            None if args.no_probe_mask else args.probe_mask_diameter
        ),
        probe_source=args.probe_source or "custom",
        probe_scale_mode=args.probe_scale_mode or "pad_preserve",
        probe_transform_pipeline=args.probe_transform_pipeline,
    )


def main(argv=None) -> None:
    os.environ.setdefault("PTYCHO_MEMOIZE_KEY_MODE", "dataset")
    args = parse_args(argv)
    from scripts.studies.invocation_logging import write_invocation_artifacts

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    write_invocation_artifacts(
        output_dir=args.output_dir,
        script_path="scripts/studies/grid_lines_workflow.py",
        argv=raw_argv,
        parsed_args=vars(args),
    )
    cfg = build_config(args)
    run_grid_lines_workflow(cfg)


if __name__ == "__main__":
    main()
