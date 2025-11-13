#!/usr/bin/env python3
"""Programmatic example: run Ptychodus API with PyTorch and TensorFlow backends."""
from __future__ import annotations

from pathlib import Path

from ptycho.raw_data import RawData
from ptycho.config.config import (
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    PyTorchExecutionConfig,
)
from ptycho.workflows.backend_selector import (
    run_cdi_example_with_backend,
    load_inference_bundle_with_backend,
)
from ptycho.workflows import components as tf_components
from ptycho_torch.inference import (
    _run_inference_and_reconstruct as torch_infer,
    save_individual_reconstructions,
)

DATA = Path("tests/fixtures/pytorch_integration/minimal_dataset_v1.npz")
WORKDIR = Path("tmp/api_demo")


def run_backend(backend: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = RawData.from_file(str(DATA))
    test_data = RawData.from_file(str(DATA))

    train_cfg = TrainingConfig(
        model=ModelConfig(N=64, gridsize=2),
        train_data_file=DATA,
        test_data_file=DATA,
        backend=backend,
        output_dir=out_dir / "train_outputs",
        n_groups=4,
        n_subsample=16,
        neighbor_count=7,
        batch_size=4,
        nepochs=1,
    )

    run_cdi_example_with_backend(train_data, test_data, train_cfg, do_stitching=False)
    print(f"[{backend}] training complete: bundle → {train_cfg.output_dir}")

    infer_cfg = InferenceConfig(
        model=train_cfg.model,
        model_path=train_cfg.output_dir,
        test_data_file=DATA,
        backend=backend,
        n_groups=4,
        n_subsample=16,
        output_dir=out_dir / "inference_outputs",
    )
    model, params_dict = load_inference_bundle_with_backend(infer_cfg.model_path, infer_cfg)

    if backend == "pytorch":
        exec_cfg = PyTorchExecutionConfig(accelerator="cpu", inference_batch_size=2)
        amp, phase = torch_infer(model, RawData.from_file(str(DATA)), infer_cfg, exec_cfg, exec_cfg.accelerator)
        save_individual_reconstructions(amp, phase, infer_cfg.output_dir)
    else:
        container = tf_components.create_ptycho_data_container(test_data, infer_cfg.model)
        amp, phase = tf_components.perform_inference(model, container, params_dict, infer_cfg, quiet=True)
        tf_components.save_outputs(amp, phase, {}, infer_cfg.output_dir)

    print(f"[{backend}] inference outputs stored in {infer_cfg.output_dir}\n")


def main() -> None:
    run_backend("pytorch", WORKDIR / "torch")
    run_backend("tensorflow", WORKDIR / "tf")


if __name__ == "__main__":
    main()
