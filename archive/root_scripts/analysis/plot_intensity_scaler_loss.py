#!/usr/bin/env python3
"""Plot intensity_scaler_inv_loss over training epochs."""

import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the training run with nphotons=3e3
# These are the final values for each epoch
epochs = [1, 2, 3, 4, 5]
train_loss = [2.2915, 0.5014, 0.4970, 0.4953, 0.4940]
val_loss = [0.5176, 0.5019, 0.4986, 0.4935, 0.4915]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'b-o', label='Training', linewidth=2, markersize=8)
plt.plot(epochs, val_loss, 'r-s', label='Validation', linewidth=2, markersize=8)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Intensity Scaler Inverse Loss', fontsize=12)
plt.title('Intensity Scaler Inverse Loss During Training (nphotons=3e3)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# Add value annotations
for i, (e, t, v) in enumerate(zip(epochs, train_loss, val_loss)):
    plt.annotate(f'{t:.3f}', (e, t), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='blue')
    plt.annotate(f'{v:.3f}', (e, v), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='red')

plt.tight_layout()
plt.savefig('intensity_scaler_inv_loss_3e3.png', dpi=150, bbox_inches='tight')
plt.show()

print("Plot saved as intensity_scaler_inv_loss_3e3.png")
print("\nSummary:")
print(f"Training loss decreased from {train_loss[0]:.3f} to {train_loss[-1]:.3f}")
print(f"Validation loss decreased from {val_loss[0]:.3f} to {val_loss[-1]:.3f}")
print(f"Final train/val gap: {abs(train_loss[-1] - val_loss[-1]):.4f}")