# Hybrid ResNet Schematics

This directory documents the reproducible schematic workflow for
`ptycho_torch` `hybrid_resnet`.

## Generate artifacts

```bash
python scripts/studies/render_hybrid_resnet_schematics.py \
  --output-dir .artifacts/hybrid_resnet_schematics/latest \
  --N 128 --gridsize 2 --fno-width 32 --fno-blocks 4 --fno-modes 12
```

## Expected outputs

- `.artifacts/hybrid_resnet_schematics/latest/hybrid_resnet_manifest.json`
- `.artifacts/hybrid_resnet_schematics/latest/hybrid_resnet_high_level.tex`
- `.artifacts/hybrid_resnet_schematics/latest/hybrid_resnet_module_flow.dot`

## Notes

- Keep bulky generated artifacts in `.artifacts/` (not committed).
- The `.tex`/`.dot` sources are deterministic and suitable for diff-based review.
- Optional conversion to PDF/SVG can be done locally with `pdflatex`/`dot` if installed.
