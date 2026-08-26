"""Train and evaluate the maintained TensorFlow supervised baseline."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow direct ``python scripts/run_baseline.py`` execution.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

from ptycho import evaluation, misc
from ptycho.config.config import (
    TrainingConfig,
    dataclass_to_legacy_dict,
)
from ptycho.image.cropping import align_for_evaluation
from ptycho.loader import PtychoDataset
from ptycho.workflows.components import (
    create_ptycho_data_container,
    load_data,
    parse_arguments,
    setup_configuration,
)

logger = logging.getLogger(__name__)


BASELINE_STITCH_SIZE = 20
# The historical baseline evaluator trimmed the repository-default four-pixel
# offset after coordinate-based alignment. TrainingConfig does not own scan
# stride, so preserve that CLI metric contract explicitly.
BASELINE_EVALUATION_TRIM_OFFSET = 4


@dataclass(frozen=True)
class BaselineRunIdentity:
    label: str
    output_prefix: str
    timestamp: str


def _resolve_run_identity(
    config: TrainingConfig,
    *,
    timestamp: str,
) -> BaselineRunIdentity:
    """Resolve run naming from explicit CLI-owned values."""
    label = f"baseline_gs{config.model.gridsize}"
    return BaselineRunIdentity(
        label=label,
        output_prefix=misc.get_path_prefix_explicit(
            label=label,
            output_prefix=str(config.output_dir),
            timestamp=timestamp,
        ),
        timestamp=timestamp,
    )


@contextmanager
def _baseline_tensorflow_scope(
    config: TrainingConfig,
    *,
    intensity_scale=None,
):
    """Project config only while a protected TensorFlow leaf executes.

    Remove this adapter when baseline model construction/training and
    ``tf_helper.reassemble_position`` accept all required values explicitly.
    """
    from ptycho import params
    from ptycho.config.config import update_legacy_dict
    from ptycho.config.legacy_state import (
        configured_params_scope,
        legacy_params_scope,
    )

    with legacy_params_scope():
        with configured_params_scope():
            update_legacy_dict(params.cfg, config)
            if intensity_scale is not None:
                # Seal-visible write (whitelisted key) instead of a direct dict write.
                params.set("intensity_scale", intensity_scale)
            yield


def _prepare_baseline_data_inputs(ptycho_dataset, config):
    """Flatten grouped channels into independent baseline-model samples."""
    import tensorflow as tf
    from ptycho.tf_helper import _channel_to_flat

    gridsize = config.model.gridsize
    if gridsize not in {1, 2}:
        raise ValueError(
            "This baseline script only supports gridsize 1 or 2, "
            f"but got {gridsize}."
        )
    n_channels = gridsize**2
    X_train = ptycho_dataset.train_data.X[..., :n_channels]
    Y_I_train = ptycho_dataset.train_data.Y_I[..., :n_channels]
    Y_phi_train = ptycho_dataset.train_data.Y_phi[..., :n_channels]
    X_test = ptycho_dataset.test_data.X[..., :n_channels]
    global_offsets = ptycho_dataset.test_data.global_offsets

    if n_channels > 1:
        logger.info(
            "Flattening %s channels to independent samples for baseline model",
            n_channels,
        )
        X_train = _channel_to_flat(X_train)
        Y_I_train = _channel_to_flat(Y_I_train)
        Y_phi_train = _channel_to_flat(Y_phi_train)
        X_test = _channel_to_flat(X_test)
        if global_offsets is not None:
            original_shape = tf.shape(global_offsets)
            logger.info("DEBUG: global_offsets original shape: %s", original_shape)
            batch_size = original_shape[0]
            actual_channels = original_shape[-1] if len(original_shape) > 3 else 1
            if actual_channels == 1:
                global_offsets = tf.tile(
                    global_offsets,
                    [1, 1, 1, n_channels],
                )
            global_offsets = tf.reshape(
                global_offsets,
                [batch_size * n_channels, 1, 2, 1],
            )

    return X_train, Y_I_train, Y_phi_train, X_test, global_offsets


def _load_baseline_dataset(config: TrainingConfig):
    """Load explicit NPZ inputs or enter the declared simulation legacy leaf."""
    with _baseline_tensorflow_scope(config):
        from ptycho import probe as probe_module

        probe_module.set_default_probe()
        if not (config.train_data_file and config.test_data_file):
            from ptycho import generate_data

            legacy = generate_data.run()
            return legacy.ptycho_dataset, legacy.YY_ground_truth

    logger.info("Loading from .npz files: %s", config.train_data_file)
    train_data_raw = load_data(
        str(config.train_data_file),
        n_images=config.n_images,
    )
    test_data_raw = load_data(str(config.test_data_file))
    train_container = create_ptycho_data_container(train_data_raw, config)
    test_container = create_ptycho_data_container(test_data_raw, config)
    dataset = PtychoDataset(train_container, test_container)
    ground_truth = (
        test_data_raw.objectGuess[None, ..., None]
        if test_data_raw.objectGuess is not None
        else None
    )
    return dataset, ground_truth


def _scalar_intensity_scale(value: Any) -> float:
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(
            "baseline training intensity scale must be scalar, "
            f"got shape {array.shape}"
        )
    return float(array.reshape(()))


def _train_baseline_and_predict(
    X_train,
    Y_I_train,
    Y_phi_train,
    X_test,
    *,
    config: TrainingConfig,
    intensity_scale: float,
):
    """Execute model construction, training, and prediction in one TF leaf."""
    with _baseline_tensorflow_scope(
        config,
        intensity_scale=intensity_scale,
    ):
        from ptycho import baselines

        # n_filters_scale flows through update_legacy_dict -> params.cfg and is
        # read by baselines.build_model at call time (W3.2); the old module-
        # global projection/restore dance existed only for the import-time read.
        model, history = baselines.train(
            X_train,
            Y_I_train,
            Y_phi_train,
        )
        pred_I_patches, pred_phi_patches = model.predict(X_test)
    return model, history, pred_I_patches, pred_phi_patches


def _reassemble_predictions(
    pred_I_patches,
    pred_phi_patches,
    global_offsets,
    *,
    config: TrainingConfig,
    intensity_scale: float,
):
    """Call the protected coordinate reassembly with scoped geometry."""
    with _baseline_tensorflow_scope(
        config,
        intensity_scale=intensity_scale,
    ):
        import tensorflow as tf
        from ptycho.tf_helper import reassemble_position

        patches = tf.cast(pred_I_patches, tf.complex64) * tf.exp(
            1j * tf.cast(pred_phi_patches, tf.complex64)
        )
        stitched = reassemble_position(
            patches,
            global_offsets,
            M=BASELINE_STITCH_SIZE,
        )
        return stitched[None, ..., None]


def _save_reconstructions_legacy(
    *,
    output_prefix: str,
    stitched_obj,
    ground_truth_obj,
) -> None:
    """Contain the output-prefix read of the legacy reconstruction exporter.

    Remove this adapter when ``ptycho.export.save_recons`` accepts its output
    directory explicitly.
    """
    from ptycho import params
    from ptycho.config.legacy_state import legacy_params_scope
    from ptycho.export import save_recons

    with legacy_params_scope():
        # Seal-visible write instead of a direct dict write (W3.1 convergence).
        params.set("output_prefix", output_prefix)
        save_recons(
            model_type="supervised",
            stitched_obj=stitched_obj,
            ground_truth_obj=ground_truth_obj,
        )


def _evaluation_snapshot(
    config: TrainingConfig,
    *,
    identity: BaselineRunIdentity,
    intensity_scale: float,
) -> dict[str, Any]:
    snapshot = dataclass_to_legacy_dict(config)
    snapshot.update(
        {
            "label": identity.label,
            "output_prefix": identity.output_prefix,
            "timestamp": identity.timestamp,
            "intensity_scale": intensity_scale,
        }
    )
    return snapshot


def run_baseline(
    config: TrainingConfig,
    *,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Run the baseline from resolved configuration without global authority."""
    if config.model.model_type != "supervised":
        raise ValueError(
            "Baseline script requires model_type='supervised', "
            f"got {config.model.model_type!r}"
        )
    timestamp = timestamp or datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    identity = _resolve_run_identity(config, timestamp=timestamp)
    Path(identity.output_prefix).mkdir(parents=True, exist_ok=True)

    logger.info("--- Starting Supervised Baseline Run ---")
    logger.info("Results will be saved to: %s", identity.output_prefix)
    logger.info("Resolved configuration: %s", config)

    dataset, ground_truth = _load_baseline_dataset(config)
    intensity_scale = _scalar_intensity_scale(dataset.train_data.norm_Y_I)
    logger.info("Resolved intensity_scale from training data: %s", intensity_scale)

    prepared = _prepare_baseline_data_inputs(dataset, config)
    X_train, Y_I_train, Y_phi_train, X_test, reassembly_offsets = prepared
    logger.info("Final training input shape: %s", X_train.shape)
    logger.info(
        "Training the baseline for %s epochs with batch size %s",
        config.nepochs,
        config.batch_size,
    )
    model, history, pred_I_patches, pred_phi_patches = (
        _train_baseline_and_predict(
            X_train,
            Y_I_train,
            Y_phi_train,
            X_test,
            config=config,
            intensity_scale=intensity_scale,
        )
    )
    model_path = Path(identity.output_prefix) / "baseline_model.h5"
    model.save(model_path)
    logger.info("Trained model saved to %s", model_path)

    try:
        stitched_obj = _reassemble_predictions(
            pred_I_patches,
            pred_phi_patches,
            reassembly_offsets,
            config=config,
            intensity_scale=intensity_scale,
        )
        logger.info("Stitched object shape: %s", stitched_obj.shape)
    except Exception as exc:
        stitched_obj = None
        logger.error("Object stitching failed: %s", exc, exc_info=True)

    metrics: dict[str, Any] | None = None
    if stitched_obj is not None and ground_truth is not None:
        recon_complex = np.squeeze(stitched_obj)
        gt_complex = np.squeeze(ground_truth)
        scan_coords_xy = np.squeeze(dataset.test_data.global_offsets)
        scan_coords_yx = scan_coords_xy[:, [1, 0]]
        recon_cropped, gt_cropped = align_for_evaluation(
            reconstruction_image=recon_complex,
            ground_truth_image=gt_complex,
            scan_coords_yx=scan_coords_yx,
            stitch_patch_size=BASELINE_STITCH_SIZE,
        )
        recon_final = recon_cropped[None, ..., None]
        gt_final = gt_cropped[..., None]
        metrics = evaluation.eval_reconstruction_explicit(
            recon_final,
            gt_final,
            trim_offset=BASELINE_EVALUATION_TRIM_OFFSET,
        )
        logger.info("Evaluation Metrics (Amplitude, Phase):")
        logger.info("  MAE:  %s", metrics["mae"])
        logger.info("  PSNR: %s", metrics["psnr"])
        evaluation.save_metrics_explicit(
            recon_final,
            gt_final,
            label=identity.label,
            trim_offset=BASELINE_EVALUATION_TRIM_OFFSET,
            output_dir=identity.output_prefix,
            config_snapshot=_evaluation_snapshot(
                config,
                identity=identity,
                intensity_scale=intensity_scale,
            ),
        )
        _save_reconstructions_legacy(
            output_prefix=identity.output_prefix,
            stitched_obj=recon_final,
            ground_truth_obj=gt_final,
        )
    elif stitched_obj is not None:
        recon_final = np.squeeze(stitched_obj)[None, ..., None]
        _save_reconstructions_legacy(
            output_prefix=identity.output_prefix,
            stitched_obj=recon_final,
            ground_truth_obj=None,
        )
    else:
        logger.warning(
            "Skipping evaluation: stitched object or ground truth was not available."
        )

    logger.info("--- Baseline script finished successfully. ---")
    return {
        "history": history,
        "metrics": metrics,
        "model_path": str(model_path),
        "output_prefix": identity.output_prefix,
        "stitched_obj": stitched_obj,
    }


def main():
    """Resolve CLI configuration and execute the supervised baseline."""
    # The facade's import-time basicConfig side effect was deleted (Phase 1);
    # this CLI owns its logging configuration now.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_arguments()
    config = setup_configuration(args, args.config)
    config = replace(
        config,
        model=replace(config.model, model_type="supervised"),
    )
    return run_baseline(config)


if __name__ == "__main__":
    main()
