"""Regression tests for shared Lightning CSV history publication."""

from pathlib import Path


def test_csv_history_path_does_not_create_a_parent_experiment():
    from ptycho_torch.training_history import _metrics_csv_path

    class Logger:
        log_dir = "/tmp/run/lightning_logs/version_0"

        @property
        def experiment(self):
            raise AssertionError("history parsing must not reopen the CSV experiment")

    assert _metrics_csv_path(Path("/tmp/run"), Logger()) == Path(
        "/tmp/run/lightning_logs/version_0/metrics.csv"
    )
