Brief:
Work from repo root (`test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"`) with `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md HUB=$PWD/plans/active/EXPORT-PTYCHODUS-PRODUCT-001/reports/2025-11-13T091500Z/hdf5_exporter_bootstrap` and reopen `analysis/data_guide_snippet.md` so you can reuse the approved copy.
Add a full “Ptychodus Product Export” subsection to `docs/DATA_MANAGEMENT_GUIDE.md` (after the existing NPZ/HDF5 guidance) that includes the `scripts/tools/convert_to_ptychodus_product.py --input-npz datasets/Run1084_recon3_postPC_shrunk_3.npz --output-product outputs/ptychodus_products/run1084_product.h5 --name … --include-diffraction/--no-include-diffraction` example, enumerates the metadata flags, explains the raw-data toggle, reiterates the `outputs/ptychodus_products/` storage policy, and links to `specs/data_contracts.md` plus `ptycho/io/ptychodus_product_io.py`.
If discoverability would benefit, add a one-line cross-link under the Data Management Guide entry in `docs/index.md` that points directly to the new subsection.
Update `$HUB/analysis/artifact_inventory.txt`, `$HUB/summary.md`, and `$HUB/summary/summary.md` with the doc path and section heading, logging `$HUB/red/blocked_<timestamp>.md` if any blocker surfaces.
No pytest selector is required for this doc-only pass.

Summary: plans/active/EXPORT-PTYCHODUS-PRODUCT-001/summary.md
Plan: plans/active/EXPORT-PTYCHODUS-PRODUCT-001/implementation_plan.md
Selector: none — evidence-only
