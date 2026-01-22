# Inference Pipeline

## Steps

1. Load the model and legacy params via `ptycho/model_manager.py`.
2. Load test data into `RawData` (`ptycho/raw_data.py`).
3. Build grouped data and a `PtychoDataContainer` (`ptycho/loader.py`).
4. Reconstruct object patches and reassemble into a full image (`ptycho/nbutils.py`, `ptycho/tf_helper.py`).
5. Save visualizations and logs (see `scripts/inference/inference.py`).

## CLI

`python scripts/inference/inference.py --model_prefix <path> --test_data <npz>`
