#!/usr/bin/env python

"""Unified training CLI with a direct Torch door and bounded TensorFlow branch."""

import argparse
from dataclasses import replace
import logging
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _configure_logging() -> None:
    root = logging.getLogger()
    if any(getattr(handler, "_ptycho_training_cli", False) for handler in root.handlers):
        return
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler("train_debug.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler._ptycho_training_cli = True
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler._ptycho_training_cli = True
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)


def parse_arguments(argv=None):
    from ptycho.cli_args import add_logging_arguments
    from ptycho.workflows.config_cli import add_public_training_config_arguments

    parser = argparse.ArgumentParser(description="Non-grid CDI Example Script")
    parser.add_argument("--config", type=str)
    parser.add_argument("--do_stitching", action="store_true", default=False)
    add_logging_arguments(parser)
    add_public_training_config_arguments(parser)
    parser.add_argument(
        "--scale-contract-version",
        choices=["ci_intensity_v2", "legacy_v1"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--measurement-domain",
        choices=["count_intensity", "normalized_amplitude"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--torch-accelerator",
        choices=["auto", "cpu", "cuda", "gpu", "mps", "tpu"],
        default="cuda",
    )
    parser.add_argument("--torch-deterministic", action="store_true", default=True)
    parser.add_argument("--torch-num-workers", type=int, default=0)
    parser.add_argument("--torch-accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--torch-learning-rate", type=float)
    parser.add_argument(
        "--torch-scheduler",
        choices=["Default", "Exponential", "WarmupCosine", "ReduceLROnPlateau"],
        default="Default",
    )
    parser.add_argument("--torch-plateau-factor", type=float)
    parser.add_argument("--torch-plateau-patience", type=int)
    parser.add_argument("--torch-plateau-min-lr", type=float)
    parser.add_argument("--torch-plateau-threshold", type=float)
    parser.add_argument(
        "--torch-logger",
        choices=["csv", "tensorboard", "mlflow", "none"],
        default="csv",
    )
    parser.add_argument("--torch-recon-log-every-n-epochs", type=int)
    parser.add_argument("--torch-recon-log-num-patches", type=int, default=4)
    parser.add_argument("--torch-recon-log-fixed-indices", type=int, nargs="+")
    parser.add_argument("--torch-recon-log-stitch", action="store_true", default=False)
    parser.add_argument("--torch-recon-log-max-stitch-samples", type=int)
    parser.add_argument("--torch-enable-checkpointing", action="store_true", default=True)
    parser.add_argument("--torch-checkpoint-save-top-k", type=int, default=1)
    return parser.parse_args(argv)


def _metadata_photon_count(path: Path) -> float | None:
    from collections.abc import Mapping
    from ptycho.metadata import MetadataManager

    try:
        _, metadata = MetadataManager.load_with_metadata(str(path))
    except Exception as error:
        logging.getLogger(__name__).debug("No dataset photon metadata: %s", error)
        return None
    if not metadata:
        return None
    if "nphotons" in metadata:
        return float(metadata["nphotons"])
    physics = metadata.get("physics_parameters")
    if isinstance(physics, Mapping) and "nphotons" in physics:
        return float(physics["nphotons"])
    return None


def main(argv=None):
    """Parse one request and execute its selected backend directly."""
    _configure_logging()
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    args = parse_arguments(raw_argv)
    try:
        from ptycho.workflows.config_cli import load_yaml_config, setup_configuration

        raw_yaml = load_yaml_config(args.config) if args.config else {}
        config = setup_configuration(args, args.config)
        metadata_photons = _metadata_photon_count(Path(config.train_data_file))
        if metadata_photons is not None:
            config = replace(config, nphotons=metadata_photons)

        from ptycho.config import (
            validate_runnable_training_config,
            validate_training_config_structure,
        )

        validate_training_config_structure(config)
        validate_runnable_training_config(config)

        if config.backend == "pytorch":
            from ptycho_torch.cli.shared import (
                build_execution_request_from_args,
                build_training_config_patch_from_args,
            )
            from ptycho_torch.config_factory import build_training_factory_overrides
            from ptycho_torch.train import train

            execution = build_execution_request_from_args(
                args,
                mode="training",
                explicit_options=raw_argv,
                lane="unified-training",
            )
            settings = build_training_factory_overrides(config)
            settings.update(
                build_training_config_patch_from_args(
                    args,
                    explicit_options=raw_argv,
                    lane="unified-training",
                )
            )
            for name in ("scale_contract_version", "measurement_domain"):
                value = getattr(args, name, None)
                if value is None and isinstance(raw_yaml, dict):
                    value = raw_yaml.get(name)
                if value is not None:
                    settings[name] = value
            authored = set(vars(args)) | (
                set(raw_yaml) if isinstance(raw_yaml, dict) else set()
            )
            for name in ("nphotons", "neighbor_count"):
                if name not in authored:
                    settings.pop(name, None)
            legacy = (
                settings.get("scale_contract_version") == "legacy_v1"
                and settings.get("measurement_domain") == "normalized_amplitude"
            )
            bundle = train(
                config.train_data_file,
                config.output_dir,
                settings,
                profile=None if legacy else "ci",
                execution_config=execution,
            )
            logging.getLogger(__name__).info("PyTorch bundle saved at: %s", bundle)
            return bundle

        import numpy as np
        from ptycho import loader, model_manager, params
        from ptycho.config.config import update_legacy_dict
        from ptycho.config.legacy_state import legacy_params_scope
        from ptycho.workflows.config_cli import load_data
        from ptycho.workflows.workflow_orchestration import run_cdi_example, save_outputs

        raw_selection = (
            config.train_raw_selection
            if config.train_raw_selection is not None
            else config.training_groups
        )
        with legacy_params_scope():
            update_legacy_dict(params.cfg, config)
            train_raw = load_data(
                str(config.train_data_file),
                n_images=config.training_groups,
                n_subsample=raw_selection,
                subsample_seed=config.subsample_seed,
            )
            test_raw = (
                load_data(str(config.test_data_file), n_images=None, n_subsample=None)
                if config.test_data_file is not None
                else None
            )

            def materialize(raw, path):
                if raw is None:
                    return None
                grouped = raw.generate_grouped_data(
                    N=config.model.N,
                    K=config.neighbor_count,
                    nsamples=config.training_groups,
                    dataset_path=str(path),
                    seed=config.subsample_seed,
                    sequential_sampling=config.sequential_sampling,
                    gridsize=config.model.gridsize,
                )
                actual = int(np.asarray(grouped["nn_indices"]).shape[0])
                if actual != config.training_groups:
                    raise ValueError(
                        f"grouping produced {actual} groups; expected exactly "
                        f"{config.training_groups}"
                    )
                return loader.load(
                    lambda: grouped,
                    raw.probeGuess,
                    which=None,
                    create_split=False,
                )

            train_container = materialize(train_raw, config.train_data_file)
            validation_container = materialize(test_raw, config.test_data_file)
            update_legacy_dict(params.cfg, config)
            amplitude, phase, results = run_cdi_example(
                train_container,
                validation_container,
                config,
                do_stitching=args.do_stitching,
            )
            results["backend"] = "tensorflow"
            update_legacy_dict(params.cfg, config)
            model_manager.save(str(config.output_dir))
        save_outputs(amplitude, phase, dict(results), str(config.output_dir))
        logging.getLogger(__name__).info(
            "TensorFlow artifacts saved via model_manager and save_outputs"
        )
        return results
    except Exception as error:
        logging.getLogger(__name__).error(
            "An error occurred during execution: %s", error
        )
        raise


if __name__ == "__main__":
    main()
