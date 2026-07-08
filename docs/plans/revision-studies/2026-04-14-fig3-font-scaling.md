# Fig. 3 Font Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Fig. 3 text readable at manuscript scale without changing the underlying probe-mischaracterization data or selected conditions.

**Architecture:** Keep the existing accepted-run figure generator and composite layout. Add explicit font-size constants for probe titles, axis labels, tick labels, and legends, then regenerate the accepted-run and paper PNG from the existing metrics/probe artifacts.

**Tech Stack:** Python, Matplotlib, pytest, pdflatex.

---

### Task 1: Scale Fig. 3 Fonts

**Files:**
- Modify: `scripts/studies/probe_mischaracterization_stress_test.py`
- Modify: `tests/studies/test_probe_mischaracterization_stress_test.py`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/figures/probe_mischaracterization_stress.png`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/data/README.md`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/changelog.txt`

- [x] Add a regression test that requires readable Fig. 3 font constants.
- [x] Run the focused test and confirm it fails before implementation.
- [x] Add named font-size constants and apply them in probe titles, axis labels, tick labels, and legends.
- [x] Run the focused test and full probe-mischaracterization test module.
- [x] Regenerate the accepted-run figure and copy it to the paper repo.
- [x] Update provenance/checklist/changelog notes with the new figure hash.
- [x] Rebuild `ptychopinn_2025.pdf` and inspect page 8.

### Task 2: Simplify And Move Fig. 3 Legend

**Files:**
- Modify: `scripts/studies/probe_mischaracterization_stress_test.py`
- Modify: `tests/studies/test_probe_mischaracterization_stress_test.py`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/figures/probe_mischaracterization_stress.png`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/data/README.md`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/changelog.txt`

- [x] Add regression tests requiring compact legend labels and no per-axis legends.
- [x] Run the focused tests and confirm they fail before implementation.
- [x] Shorten legend labels and render one shared legend outside the curve axes.
- [x] Run the focused tests and full probe-mischaracterization test module.
- [x] Regenerate the accepted-run figure and copy it to the paper repo.
- [x] Update provenance/checklist/changelog notes with the new figure hash.
- [x] Rebuild `ptychopinn_2025.pdf` and inspect page 8.
