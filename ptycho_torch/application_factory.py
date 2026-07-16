"""Application composition from the sealed Torch structural model identity."""

from __future__ import annotations

from ptycho_torch.config_params import DataConfig, InferenceConfig, TrainingConfig
from ptycho_torch.model_spec import ModelSpec
from ptycho_torch.scaling_contract import validate_scale_contract


def build_ptychopinn_application(
    model_spec: ModelSpec,
    data_config: DataConfig,
    training_config: TrainingConfig,
    inference_config: InferenceConfig,
):
    """Compose model, scientific/data, training, and inference sections.

    Runtime execution remains with Trainer orchestration and cannot affect the
    module graph or its state-dict identity.
    """
    if not isinstance(model_spec, ModelSpec):
        raise TypeError("model_spec must be a ModelSpec")
    model_config = model_spec.to_model_config()
    if model_config.C_model != data_config.C or model_config.C_forward != data_config.C:
        raise ValueError(
            "ModelSpec channel joins conflict with data_config.C: "
            f"C_model={model_config.C_model}, C_forward={model_config.C_forward}, "
            f"data C={data_config.C}"
        )

    desired_loss = "Poisson" if training_config.torch_loss_mode == "poisson" else "MAE"
    if model_config.mode == "Supervised":
        if training_config.torch_loss_mode != "mae" or model_config.loss_function != "MAE":
            raise ValueError(
                "training torch_loss_mode and model loss_function conflict: "
                "Supervised construction requires torch_loss_mode='mae' and "
                "model loss_function='MAE'"
            )
    elif model_config.loss_function != desired_loss:
        raise ValueError(
            "training torch_loss_mode and model loss_function conflict: "
            f"torch_loss_mode={training_config.torch_loss_mode!r} requires "
            f"model loss_function={desired_loss!r}, got {model_config.loss_function!r}"
        )

    validate_scale_contract(data_config, model_config, training_config)

    from ptycho_torch.model import PtychoPINN_Lightning

    constructor_kwargs = dict(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        inference_config=inference_config,
    )
    if (
        model_spec.parity_scale_mode != "off"
        or model_spec.parity_fixed_delta != 0.0
        or model_spec.parity_init_scheme != "default"
    ):
        constructor_kwargs.update(
            parity_scale_mode=model_spec.parity_scale_mode,
            parity_fixed_delta=model_spec.parity_fixed_delta,
            parity_init_scheme=model_spec.parity_init_scheme,
        )
    return PtychoPINN_Lightning(**constructor_kwargs)
