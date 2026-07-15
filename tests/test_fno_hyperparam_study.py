import csv
from pathlib import Path


def test_ensure_dataset_uses_paths_returned_by_generation(monkeypatch, tmp_path):
    from scripts.studies import fno_hyperparam_study as study

    dataset_dir = tmp_path / "datasets/N64/gs1/simulation-abc"
    expected = {
        "train_npz": str(dataset_dir / "train.npz"),
        "test_npz": str(dataset_dir / "test.npz"),
    }
    monkeypatch.setattr(study, "run_grid_lines_workflow", lambda cfg: expected)

    assert study._ensure_dataset(tmp_path, 1, 1, 1) == (
        dataset_dir / "train.npz",
        dataset_dir / "test.npz",
    )


def test_ensure_dataset_reuses_canonical_digest_paths(monkeypatch, tmp_path):
    from scripts.studies import fno_hyperparam_study as study

    cfg = study.GridLinesConfig(
        N=study.DEFAULT_N,
        gridsize=study.DEFAULT_GRIDSIZE,
        output_dir=tmp_path,
        probe_npz=study.DEFAULT_PROBE_NPZ,
        nimgs_train=1,
        nimgs_test=1,
        nepochs=1,
    )
    dataset_dir = study.dataset_out_dir(cfg)
    dataset_dir.mkdir(parents=True)
    train_npz = dataset_dir / "train.npz"
    test_npz = dataset_dir / "test.npz"
    train_npz.write_bytes(b"train")
    test_npz.write_bytes(b"test")
    monkeypatch.setattr(
        study,
        "run_grid_lines_workflow",
        lambda cfg: (_ for _ in ()).throw(AssertionError("must reuse dataset")),
    )

    assert study._ensure_dataset(tmp_path, 1, 1, 1) == (train_npz, test_npz)


def test_sweep_writes_csv(monkeypatch, tmp_path):
    from scripts.studies.fno_hyperparam_study import run_sweep

    def fake_run_torch(cfg):
        return {
            'metrics': {
                'ssim': [0.5, 0.9],
                'psnr': [10.0, 30.0],
                'mae': [0.2, 0.05],
            },
            'model_params': 1234,
            'inference_time_s': 0.12,
        }

    monkeypatch.setattr('scripts.studies.fno_hyperparam_study.run_grid_lines_torch', fake_run_torch)

    out_dir = tmp_path / 'study'
    out_dir.mkdir()
    csv_path = run_sweep(output_dir=out_dir, epochs=1, light=True, ensure_data=False)
    assert csv_path.exists()

    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert 'ssim_phase' in rows[0]
    assert 'model_params' in rows[0]


def test_write_pareto_plot(tmp_path):
    from scripts.studies.fno_hyperparam_study import write_pareto_plot

    results = [
        {"model_params": 1000, "ssim_phase": 0.9, "fno_input_transform": "none"},
        {"model_params": 2000, "ssim_phase": 0.92, "fno_input_transform": "sqrt"},
    ]
    plot_path = write_pareto_plot(results, tmp_path)
    assert plot_path.exists()
