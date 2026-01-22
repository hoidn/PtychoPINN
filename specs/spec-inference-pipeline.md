# Inference Pipeline Spec

- Load model and params via `ModelManager`.
- Load `RawData` from NPZ.
- Build grouped data (`generate_grouped_data` + loader).
- Reconstruct and reassemble patches.
- Emit visualizations and logs.
