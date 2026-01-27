import csv
from pathlib import Path


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
