# Run1084 Ptychodus Export Evidence (2025-11-13T091500Z)

## Reality Check
- Exporter/importer scaffolds (`ptycho/io/ptychodus_product_io.py`) and the CLI (`scripts/tools/convert_to_ptychodus_product.py`) already exist, but no reports hub contains pytest logs, CLI transcripts, or reader verification proving the mapping matches `specs/data_contracts.md`.
- `tests/io/test_ptychodus_product_io.py` still carries RED comments; we need a green run with the assertions enabled plus a recorded log.
- No Run1084 HDF5 product artifacts or DATA_MANAGEMENT_GUIDE snippets exist, so downstream consumers cannot validate the TensorFlow-side exporter.

## Do Now
1. Guard from `/home/ollie/Documents/PtychoPINN`, export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`, and set `HUB="$PWD/plans/active/EXPORT-PTYCHODUS-PRODUCT-001/reports/2025-11-13T091500Z/hdf5_exporter_bootstrap"`.
2. Run `pytest tests/io/test_ptychodus_product_io.py -vv | tee "$HUB"/green/pytest_product_io.log` (keep the CLI smoke enabled).
3. Convert `datasets/Run1084_recon3_postPC_shrunk_3.npz` into `outputs/ptychodus_products/run1084_product.h5` via the CLI and capture stdout/stderr in `"$HUB"/cli/convert_run1084.log`.
4. Inspect the generated HDF5 with `ptychodus.src.ptychodus.plugins.h5_product_file.H5ProductFileIO`, write the JSON summary to `analysis/product_summary.json`, and log the script output in `analysis/verify_product.log`.
5. Draft the DATA_MANAGEMENT_GUIDE snippet describing the CLI + evidence policy (`analysis/data_guide_snippet.md`) so documentation can be updated immediately after the exporter evidence lands.

## Evidence Requirements
- `green/pytest_product_io.log` with PASS/FAIL counts (no skips).
- `cli/convert_run1084.log` capturing the exact command and conversion result.
- `analysis/product_summary.json` + `analysis/verify_product.log` proving the Ptychodus reader can load the file (name, scan count, probe/object shapes).
- `analysis/data_guide_snippet.md` (draft snippet ready for docs/DATA_MANAGEMENT_GUIDE.md).
- `analysis/artifact_inventory.txt` referencing every artifact above plus the git-ignored output location `outputs/ptychodus_products/run1084_product.h5`.
