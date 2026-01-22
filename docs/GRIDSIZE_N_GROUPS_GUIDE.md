# GridSize and Grouping Guide

- `gridsize` controls how many neighboring diffraction patterns are grouped together.
- The loader builds grouped patches and offsets in `ptycho/loader.py`.
- `N` controls the base diffraction pattern size.

If you change `gridsize` or `N`, ensure the dataset shapes and offsets are consistent
with the new settings.
