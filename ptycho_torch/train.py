"""Public dataset-to-bundle training and its native Torch CLI."""

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys


def train(
    dataset,
    output_dir,
    settings,
    *,
    profile="ci",
    execution_config=None,
):
    """Train one NPZ dataset and return its ``wts.h5.zip`` bundle.

    ``settings`` uses the existing Torch resolver names. Common fields include
    ``architecture``, ``training_groups``, ``gridsize``, ``nphotons``, and
    ``epochs``; see ``docs/CONFIGURATION.md`` for the complete table.
    """
    dataset = Path(dataset)
    output_dir = Path(output_dir)
    if not isinstance(settings, Mapping):
        raise TypeError("settings must be a mapping")
    if dataset.suffix.lower() != ".npz" or not dataset.is_file():
        raise FileNotFoundError(f"training dataset must be an existing .npz: {dataset}")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output_dir must be a directory: {output_dir}")

    from ptycho_torch.config_factory import create_training_payload

    payload = create_training_payload(
        train_data_file=dataset,
        output_dir=output_dir,
        overrides=dict(settings),
        profile=profile,
        execution_config=execution_config,
    )

    from ptycho.raw_data import RawData

    train_data = RawData.from_file(str(dataset))
    validation_path = payload.pt_training_config.test_data_file
    if validation_path is not None:
        validation_path = Path(validation_path)
        if validation_path.suffix.lower() != ".npz" or not validation_path.is_file():
            raise FileNotFoundError(
                "validation dataset must be an existing .npz: "
                f"{validation_path}"
            )
        test_data = RawData.from_file(str(validation_path))
    else:
        test_data = None

    from ptycho_torch.scaling_contract import (
        CI_SCALE_CONTRACT,
        COUNT_INTENSITY,
        LEGACY_SCALE_CONTRACT,
        NORMALIZED_AMPLITUDE,
        rescale_amplitude_to_nphotons,
    )

    target = (
        payload.pt_data_config.scale_contract_version,
        payload.pt_data_config.measurement_domain,
    )
    converted_metadata_free_training = False
    for index, raw in enumerate((train_data, test_data)):
        if raw is None:
            continue
        source = (raw.scale_contract_version, raw.measurement_domain)
        if source == (CI_SCALE_CONTRACT, COUNT_INTENSITY):
            if target == (LEGACY_SCALE_CONTRACT, NORMALIZED_AMPLITUDE):
                raise ValueError(
                    "count-intensity source cannot target legacy normalized amplitude"
                )
            convert = False
        elif source == (LEGACY_SCALE_CONTRACT, NORMALIZED_AMPLITUDE):
            convert = target == (CI_SCALE_CONTRACT, COUNT_INTENSITY)
        else:
            convert = (
                source == (None, None)
                and target == (CI_SCALE_CONTRACT, COUNT_INTENSITY)
                and payload.overrides_applied["nphotons_source"] == "explicit"
            )
        if convert:
            raw.diff3d, raw.probeGuess, raw.probe_simulated = (
                rescale_amplitude_to_nphotons(
                    raw.diff3d,
                    raw.probeGuess,
                    payload.pt_data_config.nphotons,
                    raw.probe_simulated,
                )
            )
            raw.scale_contract_version, raw.measurement_domain = target
            converted_metadata_free_training |= index == 0 and source == (None, None)

    raw_cap = payload.pt_data_config.n_raw_frames_selected
    if raw_cap is not None:
        from ptycho.acquisition import select_acquisition

        selection = select_acquisition(
            train_data,
            raw_cap,
            seed=payload.pt_data_config.subsample_seed,
        )
        selected = selection.source_indices
        for name in (
            "xcoords", "ycoords", "xcoords_start", "ycoords_start", "diff3d",
            "scan_index", "object_index", "Y", "label",
        ):
            value = getattr(train_data, name, None)
            if value is not None:
                setattr(train_data, name, value[selected])
        train_data.sample_indices = selected.copy()
        train_data.subsample_seed = selection.seed

    from ptycho_torch.workflows.legacy import train_cdi_model_torch

    component_kwargs = {"resolved_payload": payload, "persist_bundle": True}
    if converted_metadata_free_training:
        import hashlib

        with dataset.open("rb") as source_file:
            component_kwargs["rescaled_source_sha256"] = hashlib.file_digest(
                source_file, "sha256"
            ).hexdigest()
    results = train_cdi_model_torch(
        train_data,
        test_data,
        payload.tf_training_config,
        **component_kwargs,
    )
    expected_bundle = output_dir / "wts.h5.zip"
    returned_bundle = results.get("bundle_path") if isinstance(results, Mapping) else None
    if (
        returned_bundle is None
        or Path(returned_bundle) != expected_bundle
        or not expected_bundle.is_file()
        or expected_bundle.stat().st_size == 0
    ):
        raise RuntimeError(
            f"training did not return a nonempty training bundle at {expected_bundle}"
        )
    return expected_bundle


def cli_main():
    """Parse native Torch syntax and delegate once to :func:`train`."""
    raw_argv = tuple(sys.argv[1:])
    parser = argparse.ArgumentParser(
        description="PyTorch Lightning training for ptychographic reconstruction"
    )
    parser.add_argument("--train_data_file")
    parser.add_argument("--test_data_file")
    parser.add_argument("--output_dir")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--n_images", type=int, default=512)
    parser.add_argument("--gridsize", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--nphotons", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--neighbor-count",
        dest="neighbor_count",
        type=int,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--log-patch-stats", action="store_true")
    parser.add_argument("--patch-stats-limit", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--disable_mlflow", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--torch-loss-mode", choices=["poisson", "mae"], default="poisson"
    )
    parser.add_argument("--profile", choices=["ci"])
    parser.add_argument(
        "--rect-s1s2-init",
        choices=["ones", "dose_closure"],
        help=(
            "Select ones or dose-closure-initialized trainable s1/s2 startup; "
            "separate from whether s1/s2 remain trainable."
        ),
    )
    parser.add_argument(
        "--scale-contract-version", choices=["ci_intensity_v2", "legacy_v1"]
    )
    parser.add_argument(
        "--measurement-domain",
        choices=["count_intensity", "normalized_amplitude"],
    )
    parser.add_argument("--probe-mask", dest="probe_mask", action="store_true")
    parser.add_argument("--no-probe-mask", dest="probe_mask", action="store_false")
    parser.set_defaults(probe_mask=False)
    parser.add_argument("--probe-mask-sigma", type=float, default=1.0)
    parser.add_argument("--probe-mask-diameter", type=float)
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu", "cuda", "tpu", "mps"],
        default="auto",
    )
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument(
        "--no-deterministic", dest="deterministic", action="store_false"
    )
    parser.set_defaults(deterministic=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--logger",
        dest="logger_backend",
        choices=["none", "csv", "tensorboard", "mlflow"],
        default="csv",
    )
    parser.add_argument(
        "--enable-checkpointing", dest="enable_checkpointing", action="store_true"
    )
    parser.add_argument(
        "--disable-checkpointing", dest="enable_checkpointing", action="store_false"
    )
    parser.set_defaults(enable_checkpointing=True)
    parser.add_argument("--checkpoint-save-top-k", type=int, default=1)
    parser.add_argument("--checkpoint-monitor", dest="checkpoint_monitor_metric", default="val_loss")
    parser.add_argument("--checkpoint-mode", choices=["min", "max"], default="min")
    parser.add_argument("--early-stop-patience", type=int, default=100)
    parser.add_argument(
        "--scheduler",
        choices=[
            "Default", "Exponential", "MultiStage", "Adaptive",
            "WarmupCosine", "ReduceLROnPlateau",
        ],
        default="Default",
    )
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    args = parser.parse_args()

    if not args.train_data_file or not args.output_dir:
        parser.error("--train_data_file and --output_dir are required")

    from ptycho_torch.cli.shared import (
        build_execution_request_from_args,
        build_training_config_patch_from_args,
    )

    try:
        execution_request = build_execution_request_from_args(
            args,
            mode="training",
            explicit_options=raw_argv,
            lane="native-training",
        )
        settings = {
            "training_groups": args.n_images,
            "batch_size": args.batch_size,
            "gridsize": args.gridsize,
            "epochs": args.max_epochs,
            "torch_loss_mode": args.torch_loss_mode,
            "probe_mask": args.probe_mask,
            "probe_mask_sigma": args.probe_mask_sigma,
            "probe_mask_diameter": args.probe_mask_diameter,
            "log_patch_stats": args.log_patch_stats,
            "patch_stats_limit": args.patch_stats_limit,
            "object_big": args.gridsize > 1,
        }
        settings.update(
            build_training_config_patch_from_args(
                args,
                explicit_options=raw_argv,
                lane="native-training",
            )
        )
        for name in (
            "nphotons", "neighbor_count", "scale_contract_version",
            "measurement_domain", "rect_s1s2_init",
        ):
            value = getattr(args, name, None)
            if value is not None:
                settings[name] = value
        if args.test_data_file:
            settings["test_data_file"] = Path(args.test_data_file)
        legacy = (
            args.scale_contract_version == "legacy_v1"
            and args.measurement_domain == "normalized_amplitude"
        )
        bundle = train(
            args.train_data_file,
            args.output_dir,
            settings,
            profile=None if legacy else (args.profile or "ci"),
            execution_config=execution_request,
        )
    except Exception as error:
        print(f"ERROR: {error}")
        raise SystemExit(1) from error
    print(f"Model bundle saved at: {bundle}")
    return bundle


if __name__ == "__main__":
    cli_main()
