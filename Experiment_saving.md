# Experiment Tracking Best Practices

## Directory Structure

Training runs are organized in a two-level hierarchy:

```
output_dir/
  {experiment_name}/
    {run_tag}_run_{timestamp}/
      metadata.json
      configs/
        data_config.json
        model_config.json
        training_config.json
        inference_config.json
      checkpoints/
        best-checkpoint.ckpt
        last.ckpt
      events.out.tfevents.*
```

- **experiment_name** groups related runs (e.g., `"CCNF_Synthetic"`, `"FNO_Ablation"`). Set via `training_config.experiment_name` in your JSON config.
- **run_tag** is an optional prefix for the run directory (e.g., `"fno_encoder"`). Set via `training_config.run_tag`. If empty, the directory is just `run_{timestamp}`.

## Config Fields for Metadata

Set these in your JSON config under `"training_config"`:

| Field | Purpose | Example |
|-------|---------|---------|
| `experiment_name` | Groups runs into experiment directories | `"CCNF_Patterson_Study"` |
| `run_tag` | Prefix for run directory name | `"baseline"`, `"fno_4block"` |
| `notes` | Free-text description of the run | `"Testing FNO encoder with 4 blocks, lr=5e-4"` |
| `model_name` | Model identifier | `"PtychoPINNv2"` |

Example config snippet:

```json
{
  "training_config": {
    "experiment_name": "CCNF_Patterson_Study",
    "run_tag": "fno_4block",
    "notes": "FNO encoder with 4 Fourier blocks, CNN decoder, lr=5e-4",
    "model_name": "PtychoPINNv2"
  }
}
```

## Dataset Provenance

The `--ptycho_dir` argument passed to `train_ccnf.py` is automatically saved as `dataset_dir` (resolved absolute path) in `metadata.json`. No extra config is needed.

## Searching Runs

Use the search interface to filter runs by experiment, dataset, notes, or architecture:

```bash
# List all runs
python -m ptycho_torch.experiment_search /path/to/outputs

# Filter by experiment
python -m ptycho_torch.experiment_search /path/to/outputs -e CCNF_Patterson_Study

# Filter by dataset (substring match)
python -m ptycho_torch.experiment_search /path/to/outputs -d pinn_velo

# Filter by notes (case-insensitive substring)
python -m ptycho_torch.experiment_search /path/to/outputs -n "fno"

# Combine filters, output as JSON
python -m ptycho_torch.experiment_search /path/to/outputs -e CCNF_Patterson_Study -s completed -f json
```

From Python:

```python
from ptycho_torch.experiment_search import search_runs

runs = search_runs(
    "/path/to/outputs",
    experiment_name="CCNF_Patterson_Study",
    dataset_dir="pinn_velo",
    status="completed",
)
for r in runs:
    print(r["run_name"], r["best_val_loss"], r["architecture"])
```

## Tips

- Always set `experiment_name` to something descriptive. The default `"Synthetic_Runs"` will mix unrelated runs together.
- Use `run_tag` to distinguish variants within an experiment (e.g., `"baseline"` vs `"fno_encoder"` vs `"large_batch"`).
- Write `notes` as if you are explaining the run to yourself in a month. Include the hypothesis or what changed from the previous run.
- The `metadata.json` file is updated at training end with `best_val_loss`, `final_epoch`, and `status`, so completed runs are always searchable by outcome.
