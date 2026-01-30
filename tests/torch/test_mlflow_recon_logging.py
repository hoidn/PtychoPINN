import torch
from unittest import mock

from ptycho_torch.workflows.recon_logging import PtychoReconLoggingCallback


class DummyLogger:
    def __init__(self):
        self.experiment = mock.Mock()
        self.run_id = "RUN123"


class DummyTrainer:
    def __init__(self):
        self.is_global_zero = True
        self.current_epoch = 4
        self.logger = DummyLogger()


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_config = mock.Mock(mode='Unsupervised')
        self.training_config = mock.Mock()

    def forward(self, x, positions, probe, input_scale_factor, output_scale_factor, experiment_ids=None):
        # pred diffraction, amp, phase
        return x, torch.abs(x), torch.angle(x + 1j * 0)


def test_callback_uses_fixed_indices_and_logs_images(tmp_path):
    cb = PtychoReconLoggingCallback(
        every_n_epochs=5,
        num_patches=4,
        fixed_indices=[0, 1, 2, 3],
        log_stitch=False,
        artifact_root="epoch_05",
    )
    trainer = DummyTrainer()
    module = DummyModule()

    # Minimal fake batch provider
    batch = (
        {
            "images": torch.ones(1, 1, 8, 8),
            "coords_relative": torch.zeros(1, 1, 2, 1),
            "rms_scaling_constant": torch.ones(1, 1, 1, 1),
            "experiment_id": torch.zeros(1, dtype=torch.long),
        },
        torch.ones(1, 8, 8),
        torch.ones(1, 1, 1, 1),
    )
    cb._get_patch_batch = mock.Mock(return_value=batch)

    cb.on_train_epoch_end(trainer, module)

    # Expect MLflow image logging
    assert trainer.logger.experiment.log_image.called
