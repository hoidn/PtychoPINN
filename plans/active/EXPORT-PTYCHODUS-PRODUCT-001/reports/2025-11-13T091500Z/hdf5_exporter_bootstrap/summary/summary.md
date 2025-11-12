### Turn Summary
Ran pytest on the Ptychodus product I/O suite (3/3 PASSED) and successfully converted Run1084 NPZ to HDF5 product format.
The generated product conforms to specs/data_contracts.md with all required datasets, position data in meters, and optional raw diffraction bundle.
Verified HDF5 structure via h5py (1087 scan positions, probe 64×64, object 227×226, diffraction in canonical NHW order) and drafted DATA_MANAGEMENT_GUIDE snippet.
Artifacts: plans/active/EXPORT-PTYCHODUS-PRODUCT-001/reports/2025-11-13T091500Z/hdf5_exporter_bootstrap/ (pytest_product_io.log, convert_run1084.log, product_summary.json, data_guide_snippet.md, artifact_inventory.txt)
