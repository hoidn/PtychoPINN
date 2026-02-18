# Hybrid ResNet Trash Recon Debug Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Determine why recent external fly001 N=128 runs produce poor `pinn_hybrid_resnet` reconstructions, while `outputs/grid_lines_external_fly001_n128_top_train_full_test_e20_seed3_cnn_hybrid_resnet` was previously good.

**Architecture:** Follow strict root-cause-first debugging. First rule out stochastic seed effects with controlled 5-epoch sweeps. Then isolate regression domain (model training vs data prep vs stitching/reassembly/visualization) by replaying known-good artifacts through current codepaths. If needed, bisect code history to identify the first bad commit and map it to a concrete behavior change.

**Tech Stack:** Python 3.11, `scripts/studies/grid_lines_torch_runner.py`, `scripts/studies/grid_lines_compare_wrapper.py`, `ptycho/workflows/grid_lines_workflow.py`, git, pytest.

---

### Task 1: Freeze Baseline and Failure Signatures

**Files:**
- Read: `outputs/grid_lines_external_fly001_n128_top_train_full_test_e20_seed3_cnn_hybrid_resnet/metrics_by_model.json`
- Read: latest failed run output dirs under `outputs/grid_lines_external_fly001_n128_top_train_full_test_*`
- Create: `tmp/debug/hybrid_resnet_trash_recon/manifest.json`

**Step 1: Capture known-good reference metrics and hashes**
Run:
```bash
mkdir -p tmp/debug/hybrid_resnet_trash_recon
python - <<'PY'
import hashlib, json
from pathlib import Path

good = Path('outputs/grid_lines_external_fly001_n128_top_train_full_test_e20_seed3_cnn_hybrid_resnet')
recon = good/'recons/pinn_hybrid_resnet/recon.npz'
metrics = good/'metrics_by_model.json'

def sha256(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

payload = {
  'good_output_dir': str(good),
  'good_recon_sha256': sha256(recon) if recon.exists() else None,
  'good_metrics_sha256': sha256(metrics) if metrics.exists() else None,
}
Path('tmp/debug/hybrid_resnet_trash_recon/manifest.json').write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
PY
```
Expected: manifest written with non-null hashes.

**Step 2: Record current failing-run artifact set**
Run:
```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('outputs')
cands = sorted(root.glob('grid_lines_external_fly001_n128_top_train_full_test_*'))
print('\n'.join(str(p) for p in cands[-12:]))
PY
```
Expected: list of recent run dirs to inspect.

---

### Task 2: Rule Out Random Seed Hypothesis (Controlled 5-Epoch Sweep)

**Files:**
- Create: `tmp/debug/hybrid_resnet_trash_recon/seed_sweep_5e.sh`
- Create: `tmp/debug/hybrid_resnet_trash_recon/seed_sweep_results.json`

**Step 1: Build deterministic sweep launcher (same dataset + params, only seed changes)**
Use three seeds (`3,4,5`), hybrid_resnet only, `epochs=5`, shared train/test NPZ copied from known-good run.

**Step 2: Execute sweep and collect metrics**
Run:
```bash
bash tmp/debug/hybrid_resnet_trash_recon/seed_sweep_5e.sh
```
Expected: three output dirs and summary JSON with per-seed MAE/SSIM and recon paths.

**Step 3: Decision gate**
- If all (or almost all) seeds are bad: reject pure-randomness hypothesis.
- If only some seeds are bad: keep seed sensitivity as contributor, continue to isolate deterministic regressions.

---

### Task 3: Isolate Domain: Model vs Data vs Stitch/Reassembly

**Files:**
- Read: `scripts/studies/grid_lines_torch_runner.py`
- Read: `ptycho/workflows/grid_lines_workflow.py`
- Create: `tmp/debug/hybrid_resnet_trash_recon/isolation_report.md`

**Step 1: Re-evaluate known-good recon with current metrics/render pipeline**
Run current `evaluate_selected_models + _finalize_compare_outputs` on the known-good recon NPZ (without retraining).

**Step 2: If this re-render now looks bad, bug is in evaluation/render/stitching path.**
If it still looks good, bug is likely in training/data path.

**Step 3: Validate dataset invariants across good vs bad runs**
Compare train/test NPZ metadata, coords ranges, object/probe stats, and hashes. Confirm whether “same params” truly implies same data payload.

---

### Task 4: Regression Identification (Git History)

**Files:**
- Read: `git log -- scripts/studies/grid_lines_torch_runner.py ptycho/workflows/grid_lines_workflow.py ptycho_torch/model.py ptycho_torch/helper.py`
- Create: `tmp/debug/hybrid_resnet_trash_recon/bisect_log.txt`

**Step 1: Define pass/fail oracle**
Use automated metric threshold from known-good run (e.g., hybrid phase SSIM/MAE band) against 5-epoch smoke to classify GOOD/BAD.

**Step 2: Run `git bisect` only if Tasks 2–3 indicate deterministic code regression**
Use narrowed pathspec and scripted oracle command.

**Step 3: Identify first bad commit and exact behavior delta**
Write concise causal chain (what changed, where, why it creates trash recons).

---

### Task 5: Root-Cause-Backed Fix + Verification

**Files:**
- Modify: only root-cause files
- Test: targeted tests for touched modules
- Create: `tmp/debug/hybrid_resnet_trash_recon/fix_verification.md`

**Step 1: Implement minimal fix for identified root cause**
No speculative changes.

**Step 2: Verify on controlled run**
Run one 5-epoch hybrid_resnet check and confirm metrics/visual no longer “trash” relative to oracle.

**Step 3: Report outcome**
Summarize: root cause, evidence, fix, and whether seed hypothesis was supported/rejected.
