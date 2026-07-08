#!/usr/bin/env python3
"""Evaluate intensity_scaler_inv loss during inference for 3e3 model."""

import numpy as np
import tensorflow as tf
from pathlib import Path
from ptycho.workflows.components import load_inference_bundle
from ptycho.loader import RawData, PtychoDataContainer
from ptycho import loader, params
import logging

logging.basicConfig(level=logging.INFO)

# Load the trained model
model_dir = Path('photon_3e3_custom_study/pinn_run')
print("Loading model...")
model, config = load_inference_bundle(model_dir)

# Load test data
print("\nLoading test data...")
test_data_path = 'photon_3e3_custom_study/data_3e3_test.npz'
data = np.load(test_data_path)

# Create RawData object
raw_data = RawData(
    diff3d=data['diff3d'],
    probeGuess=data['probeGuess'],
    scan_index=data['scan_index'],
    objectGuess=data.get('objectGuess'),
    xcoords=data.get('xcoords'),
    ycoords=data.get('ycoords'),
    xcoords_start=data.get('xcoords_start'),
    ycoords_start=data.get('ycoords_start')
)

# Generate grouped data
print("\nGenerating data container...")
dataset = raw_data.generate_grouped_data(
    config['N'], 
    K=4, 
    nsamples=1952,  # Use all test samples
    dataset_path=test_data_path,
    gridsize=1
)
ptycho_data = loader.load(lambda: dataset, raw_data.probeGuess, which=None, create_split=False)

# Prepare data for model
X = ptycho_data.X
positions = ptycho_data.coords_nominal

print(f"\nData shapes:")
print(f"X (diffraction): {X.shape}")
print(f"positions: {positions.shape}")

# Run inference and get predictions
print("\nRunning inference...")
predictions = model.predict([X, positions], batch_size=32, verbose=1)

# The model returns multiple outputs, including intensity_scaler_inv output
# Extract the predicted intensity (which has been through intensity_scaler_inv)
if isinstance(predictions, list):
    pred_intensity = predictions[-1]  # Usually the last output
else:
    pred_intensity = predictions

print(f"\nPrediction shape: {pred_intensity.shape}")

# Calculate intensity_scaler_inv loss manually
# The loss is the MSE between predicted and actual intensities
Y_intensity = X  # The input diffraction data is already intensity (amplitude squared)

# Calculate MSE loss
mse_loss = np.mean((pred_intensity - Y_intensity) ** 2)
print(f"\nIntensity MSE loss (inference): {mse_loss:.6f}")

# Also calculate the scaled version (similar to training loss reporting)
intensity_scale = config.get('intensity_scale', 1.0)
scaled_mse_loss = mse_loss / (intensity_scale ** 2)
print(f"Scaled MSE loss (normalized): {scaled_mse_loss:.6f}")

# For comparison with training, let's also compute on a subset
subset_size = 512  # Same as training batch
subset_loss = np.mean((pred_intensity[:subset_size] - Y_intensity[:subset_size]) ** 2)
scaled_subset_loss = subset_loss / (intensity_scale ** 2)
print(f"\nSubset ({subset_size} samples) MSE loss: {subset_loss:.6f}")
print(f"Subset scaled loss: {scaled_subset_loss:.6f}")

print(f"\nModel configuration:")
print(f"nphotons: {config.get('nphotons')}")
print(f"intensity_scale: {intensity_scale}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Training: 30 epochs, 1000 training images")
print(f"Training nphotons: {config.get('nphotons')}")
print(f"Inference test set: 1952 images")
print(f"Inference intensity MSE loss: {mse_loss:.6f}")
print(f"Inference scaled loss: {scaled_mse_loss:.6f}")