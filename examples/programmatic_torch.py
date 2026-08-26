import matplotlib.pyplot as plt
from ptycho_torch.inference import reconstruct
from ptycho_torch.train import train

data = "datasets/Run1084_recon3_postPC_shrunk_3.npz"
model = train(data, "outputs/run1084_cnn", {
    "architecture": "cnn",
    "training_groups": 256,
    "nphotons": 1e9,
    "epochs": 1,
})
result = reconstruct(model, data)
figure, axes = plt.subplots(1, 2)
axes[0].imshow(result.amplitude, cmap="gray")
axes[1].imshow(result.phase, cmap="twilight")
plt.show()
