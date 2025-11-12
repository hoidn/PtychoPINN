# Run1084 Ptychodus Export Evidence (2025-11-13T091500Z)

## Reality Check
- Exporter/importer scaffolds (`ptycho/io/ptychodus_product_io.py`) and the CLI (`scripts/tools/convert_to_ptychodus_product.py`) already exist, but no reports hub contains pytest logs, CLI transcripts, or reader verification proving the mapping matches `specs/data_contracts.md`.
- `tests/io/test_ptychodus_product_io.py` still carries RED comments; we need a green run with the assertions enabled plus a recorded log.
- No Run1084 HDF5 product artifacts or DATA_MANAGEMENT_GUIDE snippets exist, so downstream consumers cannot validate the TensorFlow-side exporter.

## Do Now
1. Guard from `/home/ollie/Documents/PtychoPINN`, export `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`, set `HUB="$PWD/plans/active/EXPORT-PTYCHODUS-PRODUCT-001/reports/2025-11-13T091500Z/hdf5_exporter_bootstrap"`, and make sure the drafted snippet + exporter evidence are present in this hub.
2. Insert the approved “Ptychodus Product Export” subsection into `docs/DATA_MANAGEMENT_GUIDE.md` (after the NPZ/HDF5 format descriptions). Include the Run1084 CLI example, metadata flag explanations, raw-data inclusion toggle, storage policy reminder, and references to `specs/data_contracts.md` + `ptycho/io/ptychodus_product_io.py`.
3. If useful for discoverability, add a one-line cross-link under the Data Management Guide entry in `docs/index.md` pointing to the new subsection.
4. Refresh `analysis/artifact_inventory.txt`, `summary.md`, and `summary/summary.md` so they cite the documentation insertion (file path + section title). Log blockers in `red/blocked_<timestamp>.md` if anything prevents the doc edit.

## Evidence Requirements
- `docs/DATA_MANAGEMENT_GUIDE.md` diff showing the new subsection (reference it in `analysis/artifact_inventory.txt`).
- Optional `docs/index.md` diff if a new cross-link is added.
- Updated `summary.md` and `summary/summary.md` noting the doc insertion, plus refreshed `analysis/artifact_inventory.txt`.
- Existing exporter evidence (`green/pytest_product_io.log`, `cli/convert_run1084.log`, `analysis/verify_product.log`, `analysis/product_summary.json`) remains part of this hub for traceability.
