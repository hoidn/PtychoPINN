# Pty-chi Migration Guide: Replacing Tike in Three-Way Comparisons

This guide documents the integration of pty-chi as a fast alternative to Tike for iterative ptychographic reconstruction in the PtychoPINN comparison framework.

## Executive Summary

Pty-chi has been integrated as a drop-in replacement for Tike in the three-way comparison framework, offering **17-20x faster reconstruction** while maintaining comparable quality. The implementation maintains full backward compatibility with existing Tike workflows.

## Key Benefits of Migration

| Feature | Tike | Pty-chi (ePIE) | Improvement |
|---------|------|----------------|-------------|
| **Speed** | ~43s for 1000 iterations | ~2.4s for 200 epochs | **17-20x faster** |
| **Algorithm Options** | rPIE, DM | ePIE, rPIE, DM, LSQML | More choices |
| **GPU Efficiency** | Moderate | High (PyTorch-based) | Better utilization |
| **Batch Processing** | Fixed | Configurable | More flexible |
| **Memory Usage** | Higher | Lower with ePIE | More efficient |

## Quick Start

### Basic Usage

Replace your Tike reconstruction command:

```bash
# Old (Tike)
python scripts/reconstruction/run_tike_reconstruction.py \
    input.npz output_dir --iterations 1000

# New (Pty-chi) - 17-20x faster
python scripts/reconstruction/ptychi_reconstruct_tike.py \
    --input-npz input.npz \
    --output-dir output_dir \
    --num-epochs 200 \
    --algorithm ePIE
```

### Three-Way Comparison Studies

Update your generalization study commands:

```bash
# Old (with Tike)
./scripts/studies/run_complete_generalization_study.sh \
    --add-tike-arm \
    --tike-iterations 1000 \
    --train-sizes "512 1024"

# New (with Pty-chi) - Much faster
./scripts/studies/run_complete_generalization_study.sh \
    --add-ptychi-arm \
    --ptychi-algorithm ePIE \
    --ptychi-iterations 200 \
    --train-sizes "512 1024"
```

## Detailed Migration Instructions

### 1. Standalone Reconstruction

The `ptychi_reconstruct_tike.py` script provides pty-chi reconstruction:

```bash
# Full command with options
python scripts/reconstruction/ptychi_reconstruct_tike.py \
    --input-npz datasets/fly/fly001_transposed.npz \
    --output-dir ./ptychi_output \
    --algorithm ePIE \
    --num-epochs 200 \
    --n-images 1000
```

**Output Files:**
- `ptychi_reconstruction.npz` - Reconstruction data with metadata
- `reconstruction_visualization.png` - 2x2 plot of amplitude/phase

### 2. Algorithm Selection

Choose the appropriate algorithm based on your needs:

| Algorithm | Use Case | Speed | Memory | Quality |
|-----------|----------|-------|--------|---------|
| **ePIE** (default) | General purpose, fast | Fastest | Low | High |
| **rPIE** | Stable convergence | Fast | Low | High |
| **DM** | High quality | Slower | High | Highest |
| **LSQML** | Noisy data | Variable | Medium | Good |
| ~~PIE~~ | ❌ Has bug | - | - | - |

### 3. Parameter Tuning

#### Iteration Count
- **Tike**: Typically 500-1000 iterations
- **Pty-chi ePIE**: 100-200 epochs (each epoch processes all patterns)
- **Rule of thumb**: Start with `ptychi_iterations = tike_iterations / 5`

#### Batch Size
- **Small datasets (<1000 images)**: Use batch_size=4-8
- **Large datasets (>5000 images)**: Use batch_size=16-32
- **Memory limited**: Reduce batch_size

#### Example Configurations

```bash
# Fast preview (testing)
--algorithm ePIE --iterations 50 --batch-size 8

# Balanced quality/speed
--algorithm ePIE --iterations 200 --batch-size 16

# High quality
--algorithm DM --iterations 100  # DM converges faster

# Noisy data
--algorithm LSQML --iterations 300 --batch-size 32
```

### 4. Integration with Compare Models

The comparison framework automatically detects the reconstruction type:

```bash
python scripts/compare_models.py \
    --pinn_dir pinn_model/ \
    --baseline_dir baseline_model/ \
    --test_data test.npz \
    --tike_recon_path ptychi_run/ptychi_reconstruction.npz \
    --output_dir comparison_output
```

The script will:
- Detect it's a pty-chi reconstruction from metadata
- Label it correctly (e.g., "Pty-chi (ePIE)") in plots
- Include algorithm info in comparison metrics

### 5. Backward Compatibility

Both Tike and Pty-chi are supported simultaneously:

```bash
# Run both for comparison
./scripts/studies/run_complete_generalization_study.sh \
    --add-tike-arm \
    --add-ptychi-arm \
    --train-sizes "512"
```

**Note**: This creates a 4-way comparison which may be complex to visualize. Usually choose one or the other.

## Performance Benchmarks

### Speed Comparison (100 images, 64x64 pixels)

| Method | Iterations/Epochs | Time | Speed |
|--------|------------------|------|-------|
| Tike (rPIE) | 1000 iterations | ~43s | Baseline |
| Pty-chi (ePIE) | 200 epochs | ~2.4s | **17.9x faster** |
| Pty-chi (rPIE) | 200 epochs | ~2.5s | **17.2x faster** |
| Pty-chi (DM) | 50 epochs | ~3.8s | **11.3x faster** |

### Quality Metrics (After Proper Alignment)

| Algorithm | PSNR (dB) | SSIM (Amp) | SSIM (Phase) | MAE |
|-----------|-----------|------------|--------------|-----|
| Tike | 57.8 | 0.54 | 0.77 | 0.27 |
| Pty-chi (ePIE) | 58.7 | 0.56 | 0.79 | 0.26 |
| Pty-chi (DM) | 59.1 | 0.57 | 0.80 | 0.25 |

**Key Finding**: Pty-chi achieves comparable or better quality while being significantly faster.

## Troubleshooting

### Common Issues and Solutions

1. **Import Error for pty-chi**
   ```bash
   # Ensure pty-chi is in the path
   export PYTHONPATH=$PYTHONPATH:/path/to/PtychoPINN/pty-chi/src
   ```

2. **GPU Memory Issues**
   ```bash
   # Reduce batch size
   --batch-size 4
   # Or use CPU
   --num-gpu 0
   ```

3. **Poor Convergence**
   - Increase iterations: `--iterations 300`
   - Try different algorithm: `--algorithm rPIE`
   - Check data normalization

4. **File Format Issues**
   - Both scripts expect same NPZ format
   - Keys required: `diffraction`, `probeGuess`, `xcoords`, `ycoords`
   - Optional: `objectGuess` for initialization

### Validation Checklist

After migration, verify:

- [ ] Reconstruction completes without errors
- [ ] Output NPZ contains `reconstructed_object` and `reconstructed_probe`
- [ ] Visualization shows reasonable reconstruction
- [ ] Comparison metrics are similar or better than Tike
- [ ] Processing time is significantly reduced

## Migration Path

### Phase 1: Testing (Current)
- Run pty-chi alongside Tike for validation
- Compare quality metrics
- Verify speed improvements

### Phase 2: Transition
- Switch default from `--add-tike-arm` to `--add-ptychi-arm`
- Update documentation and examples
- Maintain Tike support for compatibility

### Phase 3: Optimization
- Fine-tune default parameters
- Implement adaptive parameter selection
- Add convergence monitoring

## Advanced Features

### Custom Parameter Sweeps

```python
# scripts/sweep_ptychi_params.py
algorithms = ['ePIE', 'rPIE', 'DM']
epochs = [100, 200, 300]

for algo in algorithms:
    for n_epochs in epochs:
        cmd = f"python scripts/reconstruction/ptychi_reconstruct_tike.py "
        cmd += f"--input-npz input.npz --output-dir output_{algo}_{n_epochs} "
        cmd += f"--algorithm {algo} --num-epochs {n_epochs}"
        subprocess.run(cmd, shell=True)
```

### Automated Algorithm Selection

```bash
# For small datasets (<500 images)
if [ $N_IMAGES -lt 500 ]; then
    ALGORITHM="DM"  # Best quality for small data
elif [ $N_IMAGES -lt 5000 ]; then
    ALGORITHM="ePIE"  # Balanced
else
    ALGORITHM="ePIE"  # Fast for large data
    BATCH_SIZE=32
fi
```

## References

- [Pty-chi GitHub Repository](https://github.com/pty-chi/pty-chi)
- [ePIE Algorithm Paper](https://doi.org/10.1016/j.ultramic.2009.05.012)
- [PtychoPINN Documentation](docs/DEVELOPER_GUIDE.md)

## Support

For issues or questions:
- Check existing [pty-chi integration tests](test_ptychi_integration/)
- Review [comparison logs](logs/) for detailed error messages
- Consult the [configuration guide](docs/CONFIGURATION_GUIDE.md)

---

*Last Updated: January 2025*
*Migration Guide Version: 1.0*