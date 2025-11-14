Brief:
Export `AUTHORITATIVE_CMDS_DOC`, `TF_XLA_FLAGS="--tf_xla_auto_jit=0"`, and `USE_XLA_TRANSLATE=0`, then create the GS1 folders under `$HUB/scaling_alignment/phase_c1_gs1/` and `$HUB/tf_baseline/phase_c1_gs1/` so both backends log the env capture before any commands run.
Rerun the PyTorch short baseline + inference with `--gridsize 1` (same dataset/files as Phase B3) writing logs/artifacts into the new GS1 folder, copy the patch stats/debug bundle, and record the torch bundle digest.
Run the TensorFlow integration selector, then execute the TF training and inference CLIs with `--gridsize 1` (logs + debug dump under `tf_baseline/phase_c1_gs1/`), capture bundle digests + a stats delta text file comparing the GS1 PyTorch vs TF `stats.json`, and update `$HUB/analysis/artifact_inventory.txt` / `$HUB/summary.md` with a “GS1 fallback” section plus the Dataset note.
Log `$HUB/scaling_alignment/phase_c1_gs1/red/blocked_*.md` or `$TF_BASE_GS1/red/blocked_*.md` immediately if any GS1 command fails; otherwise leave the hub ready for Phase C2 consumers.

Summary: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md
Plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md
Selector: tests/test_integration_workflow.py::TestFullWorkflow::test_train_save_load_infer_cycle
