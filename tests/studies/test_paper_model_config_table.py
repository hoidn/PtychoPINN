import json
from pathlib import Path

import torch

from scripts.studies.paper_model_config_table import (
    ModelConfigRow,
    count_unique_state_dict_params,
    load_brdt_config_rows,
    load_cdi_config_rows,
    render_model_config_tex,
    row_to_dict,
)


def test_count_unique_state_dict_params_dedupes_generator_autoencoder_aliases():
    state = {
        "model.generator.block.weight": torch.zeros(2, 3),
        "model.generator.block.bias": torch.zeros(2),
        "model.autoencoder.block.weight": torch.zeros(2, 3),
        "model.autoencoder.block.bias": torch.zeros(2),
        "model.forward_model.alpha": torch.zeros(1),
        "model.forward_model.beta": torch.zeros(1),
    }

    result = count_unique_state_dict_params(state)

    assert result.unique_trainable_params == 8
    assert result.raw_recorded_parameter_count == 16
    assert result.duplicate_groups == ["model.generator/model.autoencoder"]


def test_count_unique_state_dict_params_keeps_nonmatching_autoencoder():
    state = {
        "model.generator.block.weight": torch.zeros(2, 3),
        "model.autoencoder.block.weight": torch.zeros(4, 3),
    }

    result = count_unique_state_dict_params(state)

    assert result.unique_trainable_params == 18
    assert result.raw_recorded_parameter_count == 18
    assert result.duplicate_groups == []


def test_row_to_dict_contains_normalized_contract_fields():
    row = ModelConfigRow(
        benchmark="CDI Lines128",
        display_model="SRU-Net",
        row_id="pinn_hybrid_resnet",
        internal_architecture="hybrid_resnet",
        training_objective="PINN",
        input_output_contract="diffraction intensity -> complex object",
        width="32",
        fourier_modes="12",
        encoder_blocks="4",
        bottleneck_blocks="6",
        downsampling="2",
        skip_or_fusion="add",
        unique_trainable_params=9003299,
        raw_recorded_parameter_count=18006600,
        parameter_count_kind="deduped_unique_effective_trainable_params",
        parameter_count_source="runs/pinn_hybrid_resnet/model.pt",
        config_source="runs/pinn_hybrid_resnet/config.json",
        notes="deduped generator/autoencoder aliases",
    )

    payload = row_to_dict(row)

    for key in [
        "benchmark",
        "display_model",
        "row_id",
        "internal_architecture",
        "training_objective",
        "input_output_contract",
        "width",
        "fourier_modes",
        "encoder_blocks",
        "bottleneck_blocks",
        "downsampling",
        "skip_or_fusion",
        "unique_trainable_params",
        "parameter_count_source",
        "config_source",
    ]:
        assert key in payload
    assert json.loads(json.dumps(payload)) == payload


def test_render_model_config_tex_uses_readable_labels_and_escapes_paths():
    rows = [
        ModelConfigRow(
            benchmark="CDI Lines128",
            display_model="SRU-Net",
            row_id="pinn_hybrid_resnet",
            internal_architecture="hybrid_resnet",
            training_objective="PINN",
            input_output_contract="diffraction intensity -> complex object",
            width="32",
            fourier_modes="12",
            encoder_blocks="4",
            bottleneck_blocks="6",
            downsampling="2",
            skip_or_fusion="add",
            unique_trainable_params=9003299,
            raw_recorded_parameter_count=18006600,
            parameter_count_kind="deduped_unique_effective_trainable_params",
            parameter_count_source="runs/pinn_hybrid_resnet/model.pt",
            config_source="runs/pinn_hybrid_resnet/config.json",
            notes="deduped generator/autoencoder aliases",
        ),
        ModelConfigRow(
            benchmark="PDEBench CNS",
            display_model="FFNO",
            row_id="author_ffno_cns_base",
            internal_architecture="author_ffno_cns",
            training_objective="supervised MSE",
            input_output_contract="history_len=5 -> next frame",
            width="not_recorded",
            fourier_modes="not_recorded",
            encoder_blocks="not_recorded",
            bottleneck_blocks="not_applicable",
            downsampling="not_recorded",
            skip_or_fusion="factorized Fourier",
            unique_trainable_params=1074440,
            raw_recorded_parameter_count=1074440,
            parameter_count_kind="unique_effective_trainable_params",
            parameter_count_source="tables/pdebench_cns_matched_condition_metrics.json",
            config_source="tables/pdebench_cns_matched_condition_metrics.json",
            notes="capped row",
        ),
        ModelConfigRow(
            benchmark="BRDT",
            display_model="FFNO",
            row_id="ffno",
            internal_architecture="ffno",
            training_objective="supervised+Born",
            input_output_contract="born_init_image -> q_pred",
            width="16",
            fourier_modes="8",
            encoder_blocks="2",
            bottleneck_blocks="not_applicable",
            downsampling="1",
            skip_or_fusion="factorized Fourier",
            unique_trainable_params=36674,
            raw_recorded_parameter_count=36674,
            parameter_count_kind="unique_effective_trainable_params",
            parameter_count_source="rows/ffno/model_profile.json",
            config_source="rows/ffno/model_profile.json",
            notes="40 epoch",
        ),
    ]

    tex = render_model_config_tex(rows)

    assert "Unique params" in tex
    assert "CDI Lines128" in tex
    assert "PDEBench CNS" in tex
    assert "BRDT" in tex
    assert "pinn\\_hybrid\\_resnet" in tex
    assert "command_wall_time_sec" not in tex


def test_load_cdi_config_rows_uses_corrected_no_refiner_ffno_roots():
    rows = {row.row_id: row for row in load_cdi_config_rows(Path.cwd())}

    assert rows["pinn_ffno"].display_model == "FFNO"
    assert rows["pinn_ffno"].unique_trainable_params == 124966
    assert rows["pinn_ffno"].config_source.endswith(
        "2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/runs/ffno_no_refiner_20260506T223454Z/runs/pinn_ffno/config.json"
    )
    assert rows["supervised_ffno"].display_model == "FFNO"
    assert rows["supervised_ffno"].unique_trainable_params == 124966
    assert rows["supervised_ffno"].config_source.endswith(
        "2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/runs/supervised_ffno_no_refiner_20260506T232535Z/runs/supervised_ffno/config.json"
    )


def test_load_cdi_config_rows_can_switch_to_depth24_pair():
    rows = {
        row.row_id: row
        for row in load_cdi_config_rows(
            Path.cwd(),
            cdi_final_ffno_pair_key="depth24_no_refiner",
        )
    }

    assert rows["pinn_ffno"].config_source.endswith(
        "2026-05-06-cdi-lines128-ffno-depth24-ablation/runs/ffno_depth24_20260507T052301Z/runs/pinn_ffno_depth24/config.json"
    )
    assert rows["pinn_ffno"].encoder_blocks == "24"
    assert rows["pinn_ffno"].notes == "active CDI final paper row; depth24_no_refiner"
    assert rows["supervised_ffno"].config_source.endswith(
        "2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/runs/supervised_ffno_depth24_20260507T192840Z/runs/supervised_ffno_depth24/config.json"
    )
    assert rows["supervised_ffno"].encoder_blocks == "24"


def test_load_brdt_config_rows_uses_sinogram_input_bundle():
    rows = {row.row_id: row for row in load_brdt_config_rows(Path.cwd())}

    assert set(rows) == {"sru_net", "ffno"}
    assert rows["sru_net"].display_model == "SRU-Net"
    assert rows["sru_net"].input_output_contract == (
        "measured complex sinogram -> q_pred"
    )
    assert rows["sru_net"].config_source.endswith(
        "2026-05-07-brdt-sinogram-input-40ep-paper-evidence/rows/sru_net/model_profile.json"
    )
