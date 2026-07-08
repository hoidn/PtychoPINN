# TF–Torch Discrepancy Debug Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Identify and resolve the persistent behavioral discrepancy between TensorFlow and PyTorch reconstruction outputs by tracing data, normalization, offsets, and reassembly through both pipelines with controlled parity checks.

**Architecture:** Establish a reproducible parity harness (same dataset, subsample, and config), add boundary‑level probes (inputs/outputs) to each stage, then isolate deltas via minimal tests (offsets, probe normalization, intensity scaling, forward model, and stitching). Fix only after a single root cause is confirmed.

**Tech Stack:** Python, TensorFlow, PyTorch, NumPy, pytest

---

## Task 1: Build Parity Baseline (No Code Changes)

**Files:**
- Create: `tmp/parity_baseline/summary.md`

**Step 1: Choose canonical dataset + subsample**
- Use the same NPZ as the torch long integration (canonical dataset).
- Fix `subsample_seed` and `n_images` in both TF and Torch to the same values.

**Step 2: Run TF and Torch inference with identical inputs**
- Use identical `n_images`, `gridsize`, and `nphotons`.
- Capture stdout logs to `tmp/parity_baseline/` for both runs.

**Step 3: Record outputs**
- Save amplitude/phase PNGs for both.
- Compute bbox + basic stats (min/max/mean/std) for both.
- Summarize in `tmp/parity_baseline/summary.md`.

---

## Task 2: Add Stage‑Boundary Probes (Evidence Only)

**Files:**
- Modify: `ptycho_torch/inference.py`
- Modify: `ptycho_torch/helper.py`
- Modify: `ptycho/inference.py` (TF) or the TF reassembly call site used by the integration test
- Create: `ptycho/debug_parity.py`

**Step 1: Torch probes**
- Log: raw diffraction stats (min/max/mean/std), probe stats, offsets stats (shape, unique count, std), and reassembly output shape.

**Step 2: TF probes**
- Log the same data points at the equivalent boundaries.

**Step 3: Run with parity probe env var**
- Ensure probes are gated by env var (no default behavior change).
- Capture logs for comparison.

---

## Task 3: Minimal Parity Tests (RED → GREEN)

**Files:**
- Create: `tests/torch/test_parity_offsets.py`
- Create: `tests/torch/test_parity_probe_normalization.py`
- Create: `tests/torch/test_parity_intensity_scale.py`
- Create: `tests/torch/test_parity_reassembly.py`

**Step 1: Offsets parity**
- Build a deterministic set of offsets and compare TF vs Torch translation/reassembly outputs.

**Step 2: Probe normalization parity**
- Compare TF probe normalization (set_probe_guess) vs Torch normalization helper.

**Step 3: Intensity scaling parity**
- Ensure derived scale matches given identical diffraction stats + nphotons.

**Step 4: Reassembly parity**
- Verify identical stitched output support/mask for a small multi‑patch synthetic case.

---

## Task 4: Identify Root Cause

**Files:**
- Create: `tmp/parity_baseline/analysis.md`

**Step 1: Compare stage outputs**
- Identify the first boundary where TF and Torch diverge.

**Step 2: State a single root cause hypothesis**
- Record in `analysis.md` with evidence links.

---

## Task 5: Fix (TDD)

**Files:**
- Modify: minimal file at the divergence boundary
- Update: the failing parity test from Task 3

**Step 1: Write/adjust a failing test**
- Fails for the identified root cause.

**Step 2: Implement minimal fix**
- One change only.

**Step 3: Verify**
- The parity test passes and baseline re-runs match.

---

## Task 6: Re‑run Full Parity Baseline

**Step 1:** Re-run TF and Torch end‑to‑end.
**Step 2:** Update `tmp/parity_baseline/summary.md` and `analysis.md`.

---

## Notes
- Keep probes and logs gated by env vars.
- Do not modify core TF physics unless root cause is proven.
- If 3+ fixes fail, pause for architecture review.
