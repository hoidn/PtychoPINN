#!/usr/bin/env python

"""Legacy CLI adapter for the shared generic training workflow."""

import argparse
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from ptycho.workflows.components import (  # noqa: E402
    add_public_training_config_arguments,
)
from ptycho.workflows.training import (  # noqa: E402
    TrainingWorkflowRequest,
    run_training_workflow,
)


def _configure_logging() -> None:
    """Install the historical CLI handlers exactly once per process."""

    root = logging.getLogger()
    if any(getattr(handler, "_ptycho_training_cli", False) for handler in root.handlers):
        return
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
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
    """Extend the public parser with Torch runtime and optimizer flags."""

    from ptycho.cli_args import add_logging_arguments

    parser = argparse.ArgumentParser(description="Non-grid CDI Example Script")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "--do_stitching",
        action="store_true",
        default=False,
        help="Perform image stitching after training (default: False)",
    )
    add_logging_arguments(parser)
    add_public_training_config_arguments(parser)

    parser.add_argument(
        "--torch-accelerator",
        type=str,
        choices=["auto", "cpu", "cuda", "gpu", "mps", "tpu"],
        default="cuda",
        help=(
            "PyTorch accelerator for training (only applies when --backend "
            "pytorch). Default: cuda."
        ),
    )
    parser.add_argument(
        "--torch-deterministic",
        action="store_true",
        default=True,
        help="Enable deterministic PyTorch training (default: True).",
    )
    parser.add_argument(
        "--torch-num-workers",
        type=int,
        default=0,
        help="PyTorch dataloader workers (default: 0).",
    )
    parser.add_argument(
        "--torch-accumulate-grad-batches",
        type=int,
        default=1,
        help="Accumulate gradients over N batches (default: 1).",
    )
    parser.add_argument(
        "--torch-learning-rate",
        type=float,
        default=None,
        help="Explicit PyTorch learning rate.",
    )
    parser.add_argument(
        "--torch-scheduler",
        type=str,
        default="Default",
        choices=[
            "Default",
            "Exponential",
            "WarmupCosine",
            "ReduceLROnPlateau",
        ],
        help="PyTorch scheduler (default: Default).",
    )
    parser.add_argument("--torch-plateau-factor", type=float, default=None)
    parser.add_argument("--torch-plateau-patience", type=int, default=None)
    parser.add_argument("--torch-plateau-min-lr", type=float, default=None)
    parser.add_argument("--torch-plateau-threshold", type=float, default=None)
    parser.add_argument(
        "--torch-logger",
        type=str,
        default="csv",
        choices=["csv", "tensorboard", "mlflow", "none"],
        help="PyTorch logger backend (default: csv).",
    )
    parser.add_argument("--torch-recon-log-every-n-epochs", type=int, default=None)
    parser.add_argument("--torch-recon-log-num-patches", type=int, default=4)
    parser.add_argument(
        "--torch-recon-log-fixed-indices",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--torch-recon-log-stitch",
        action="store_true",
        default=False,
    )
    parser.add_argument("--torch-recon-log-max-stitch-samples", type=int, default=None)
    parser.add_argument(
        "--torch-enable-checkpointing",
        action="store_true",
        default=True,
    )
    parser.add_argument("--torch-checkpoint-save-top-k", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    """Parse CLI arguments and delegate to the shared generic workflow."""

    _configure_logging()
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    args = parse_arguments(raw_argv)
    try:
        result = run_training_workflow(
            TrainingWorkflowRequest(
                legacy_args=args,
                raw_argv=raw_argv,
                do_stitching=args.do_stitching,
            )
        )
    except Exception as exc:
        logging.getLogger(__name__).error(
            "An error occurred during execution: %s",
            exc,
        )
        raise

    config = getattr(result, "public_config", None)
    if config is not None and config.backend == "pytorch":
        logging.getLogger(__name__).info(
            "PyTorch backend completed. Check %s for saved bundles.",
            config.output_dir,
        )
        bundle_path = getattr(result, "bundle_path", None)
        if bundle_path is not None:
            logging.getLogger(__name__).info(
                "PyTorch bundle saved at: %s",
                bundle_path,
            )
    elif config is not None:
        logging.getLogger(__name__).info(
            "TensorFlow artifacts saved via model_manager and save_outputs"
        )
    return result


if __name__ == "__main__":
    main()
