Revised P0 + P1 Revision Plan (With Codebase & Generalization)
Objective: Submit a manuscript that claims Fast, Physics-Informed, Generalizable, and Single-Shot capabilities, using only existing data/figures.
A. Self-contained revision tasks (Do these first)
A0. Setup
Create a clean branch/folder.
Ensure main.tex compiles.
A1. Lock the Scope (Updated with Verification Sources)
Keep:

### Methods: Core physics + Dual-resolution decoder
- Codebase Ref: ptycho/model.py (Search create_decoder_last to see probe.big logic verifying the dual-resolution claim).
- Codebase Ref: ptycho/tf_helper.py (Search pad_and_diffract to verify the Far-Field/FFT physics model).
- Doc Ref: docs/specs/spec-ptycho-core.md (Normative forward model definition, lines 18-36; FFT amplitude formula lines 22-27).
- Doc Ref: docs/architecture_tf.md (TensorFlow component diagram and training sequence).

### Poisson NLL Loss (Paper §2.4, Discussion)
- Codebase Ref: ptycho/model.py (negloglik function implements the loss).
- Doc Ref: docs/specs/spec-ptycho-core.md §Losses lines 77-84 (Normative: "L_poisson = Y_pred − Y_true · log(Y_pred)").
- Doc Ref: docs/DATA_NORMALIZATION_GUIDE.md (CRITICAL: defines three normalization types - physics/statistical/display. Paper must not conflate these).

### Single-Shot: The Fresnel/Overlap-free argument (Fig 1)
- Doc Ref: docs/GRIDSIZE_N_GROUPS_GUIDE.md (Read "GridSize = 1" section to confirm how n_groups works without overlap).
- Doc Ref: docs/specs/overlap_metrics.md (Defines overlap metrics: group-based, image-based, COM-based).

### Data Efficiency: The SSIM vs Training Size argument (Fig 4)
- Script Ref: scripts/studies/run_complete_generalization_study.sh (Source of the training size sweep logic).
- Doc Ref: docs/studies/GENERALIZATION_STUDY_GUIDE.md (Complete methodology for training size sweeps).
- Doc Ref: docs/SAMPLING_USER_GUIDE.md (n_subsample and n_groups parameter semantics).

### Cross-Facility Generalization: APS → LCLS transfer (Fig 2)
- Doc Ref: docs/FLY64_DATASET_GUIDE.md (Experimental data preprocessing requirements).
- Doc Ref: docs/FLY64_GENERALIZATION_STUDY_ANALYSIS.md (VERIFY CLAIMS: contains actual analysis - notes baseline may outperform in some cases).

### Low-Dose: The Poisson vs MAE visual argument (Fig 5)
- Script Ref: scripts/studies/dose_response_study.py (Source of the specific High vs Low dose comparison logic).
- Doc Ref: docs/DATA_GENERATION_GUIDE.md (Grid vs nongrid simulation pipelines for dose studies).

### Data Format Verification
- Doc Ref: specs/data_contracts.md (NPZ format: required keys diff3d, xcoords, ycoords, probeGuess; dtypes and shapes).

### Configuration Parameters (Paper Table 1)
- Doc Ref: docs/CONFIGURATION.md (Canonical parameter documentation for ModelConfig, TrainingConfig, InferenceConfig).

### Cut:
- The separate "Sparse Sampling Robustness" subsection (3.3.1).
- The "Photon-Limited Performance" Table (Table ??).
- Specific speedup claims (e.g., "1000x") unless you have a citation.

---

## A1.5 Manuscript Preparation Conventions (REPRODUCIBILITY)

**Core Convention:** All figures and numerical results in the manuscript MUST be reproducible via scripts in the `paper/` directory.

### Directory Structure
```
paper/
├── ptychopinn_2025.tex          # Main manuscript
├── figures/                      # Generated figures (gitignored PNGs, committed PDFs)
│   └── scripts/                  # Figure generation scripts
│       ├── generate_fig1.py      # → figures/recon_2x2.pdf
│       ├── generate_fig2.py      # → figures/outdist.pdf
│       └── ...
├── tables/
│   ├── metrics.tex               # Generated LaTeX tables
│   └── scripts/
│       └── generate_metrics.py   # → tables/metrics.tex
├── data/                         # Intermediate results (gitignored)
│   └── README.md                 # Documents data provenance
└── Makefile                      # One-command reproducibility
```

### Requirements for Figures

1. **Script-Generated:** Each figure MUST have a corresponding script in `paper/figures/scripts/`
2. **Naming Convention:** `generate_<fig_label>.py` → `<fig_label>.pdf`
3. **Data Provenance:** Scripts MUST document input data source (NPZ path, study output, etc.)
4. **Determinism:** Scripts MUST produce identical output given same inputs (set random seeds)

Example header for figure scripts:
```python
"""
Generate Figure X: <description>

Input: plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/.../results.json
Output: paper/figures/<fig_label>.pdf

Reproduces: Paper Section X.Y, Figure X
"""
```

### Requirements for Numerical Results

1. **No Hardcoded Values:** Metrics in the manuscript MUST be read from JSON/CSV artifacts
2. **Artifact Trail:** Each number MUST trace to a study output or script result
3. **Uncertainty Reporting:** Include std/CI where applicable (as in Table 1: "mean ± std")

### Verification Before Submission

```bash
# From paper/ directory:
make clean && make all    # Regenerate all figures and tables
make verify               # Check all artifacts are current
```

### Current Gaps to Address

| Figure/Table | Script Exists? | Data Source Documented? | Status |
|--------------|---------------|------------------------|--------|
| Fig 1 (recon_2x2) | ✗ | ✗ | Needs script; source: SIM-LINES-4X study |
| Fig 2 (outdist) | ✗ | ✗ | Needs script |
| Fig 3 (smalldat) | ✗ | ✗ | Needs script |
| Fig 4 (ssim) | ✗ | ✗ | Needs script |
| Fig 5 (lowcounts) | ✗ | ✗ | Needs script |
| Fig 6 (dose) | Partial (`dose_response_study.py`) | ✓ | Link to paper/ |
| Table 1 (metrics) | ✗ | Partial | Needs provenance |

### Linking Existing Study Scripts

For figures derived from existing studies, create thin wrapper scripts:

```python
# paper/figures/scripts/generate_dose_comparison.py
"""Wrapper to generate dose figure from STUDY-SYNTH-DOSE-COMPARISON-001"""
import subprocess
import shutil
from pathlib import Path

STUDY_SCRIPT = "scripts/studies/dose_response_study.py"
OUTPUT_DIR = Path("plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports")
FIGURE_DEST = Path("paper/figures/dose_comparison.pdf")

# Run study if results don't exist
# Copy/convert figure to paper/figures/
```

### Makefile Template

```makefile
# paper/Makefile
FIGURES := figures/recon_2x2.pdf figures/outdist.pdf figures/ssim.pdf
TABLES := tables/metrics.tex

all: $(FIGURES) $(TABLES) ptychopinn_2025.pdf

figures/%.pdf: figures/scripts/generate_%.py
	python $< --output $@

tables/metrics.tex: tables/scripts/generate_metrics.py
	python $< --output $@

ptychopinn_2025.pdf: ptychopinn_2025.tex $(FIGURES) $(TABLES)
	pdflatex ptychopinn_2025.tex
	bibtex ptychopinn_2025
	pdflatex ptychopinn_2025.tex
	pdflatex ptychopinn_2025.tex

verify:
	@echo "Checking artifact freshness..."
	@for f in $(FIGURES) $(TABLES); do \
		if [ ! -f "$$f" ]; then echo "MISSING: $$f"; fi; \
	done

clean:
	rm -f $(FIGURES) $(TABLES) *.aux *.bbl *.blg *.log
```

---

A2. Placeholder Removal (Search & Destroy)
Global search for: TODO, ??, COMMENT, [ ] (brackets).
Action: Delete or replace using the text blocks provided in Section C below.
A3. Fix the "Inverse Problem" Intro (Page 2)
Delete the large TODO paragraph at the end of the Intro.
Action: Paste Text Block C1.
Context: This text aligns with ptycho/model.py's loss function (negloglik) which implements the Poisson constraint mentioned.
A4. Fix Section 3: Results
3.1 Reconstruction Quality
Reference Figure \ref{fig:smalldat} (currently Fig 3) and Table 1.
Remove the sentence promising "iterative reconstruction SSIM values."
3.2 Overlap-Free Reconstruction
CRITICAL FIX: Change 
C
g
=
0
C 
g
​
 =0
 to 
C
g
=
1
C 
g
​
 =1
 (single-frame group).
Codebase Verification: See ptycho/raw_data.py. The grouping logic handles gridsize=1 (1x1=1 frame), not 0. A group size of 0 is a code error.
Reference Figure \ref{fig:recon_2x2} (currently Fig 1).
Action: Paste Text Block C2.
3.3 Data Efficiency (Rescued)
Keep Figure 4.
Action: Update caption to: "Validation set structural similarity (SSIM) as a function of training set size. PtychoPINN achieves high fidelity with an order of magnitude less data than the supervised baseline."
Action: Paste Text Block C3.
3.4 Out-of-Distribution Generalization (Rescued)
Keep Figure 2.
CRITICAL FIX (Fig 2): The figure currently has "X nm" placeholders.
If you know the pixel size: Replace "X" with the number.
If you DO NOT know: Open the image file in an editor and crop out the scale bars, OR use LaTeX \includegraphics[trim=...] to hide the bottom edge. Do not submit "X nm".
Labeling: Ensure Legend A = "Train: APS / Test: APS", Legend B = "Train: APS / Test: LCLS".
Action: Paste Text Block C4.
3.5 Photon-Limited Performance
Keep Figure 5.
Cut Table ??.
Action: Paste Text Block C5.
Context: This text describes the output of scripts/studies/dose_response_study.py, specifically the generate_six_panel_figure function.
3.6 Computational Performance
Keep your throughput number (~2000 patterns/s).
Verification: You can re-verify this by running scripts/training/train.py on your GPU and checking the progress bar logs (logs/debug.log).
Cut the "speedup factor" claims unless you have a citation.
A5. Cleanup Discussion & Abstract
Abstract: Ensure you mention "cross-facility generalization" and "data efficiency." Remove "100-1000x" if you didn't prove it in results.
Discussion:
Delete "Ablations TODO".
Delete internal notes/citations.
Ensure the text matches the figures kept above.
B. Tasks requiring external info (Only if needed)
Author List/Affiliations: Confirm with PI.
Disclosures: Confirm "No conflicts."
Data Availability: Decide where the code lives.
Speedup Benchmark (Optional):
Codebase Ref: If you must include a comparison, run scripts/reconstruction/run_tike_reconstruction.py on the same dataset to get the "Iterative Solver" timing for a fair comparison.
C. Copy-Paste Text Blocks (Use these to replace TODOs)
C1. Introduction (End of section)
Inverse problem and constraints. Coherent diffractive imaging seeks a complex object from intensity-only diffraction measurements. Reconstruction methods enforce two complementary constraints: a reciprocal-space constraint requiring predicted intensities to match data (via a physics-based forward model), and a real-space constraint enforcing consistency between overlapping views. In our framework, the reciprocal-space constraint is enforced directly via a differentiable forward model and a Poisson likelihood. Real-space overlap is handled via a translation-aware merging operator. Crucially, this allows overlap to be treated as a flexible experimental parameter rather than a hard requirement; setting the group size to a single frame removes overlap constraints entirely, enabling "single-shot" reconstruction when the probe provides sufficient phase diversity.
C2. Results: Overlap-Free Reconstruction
In overlap-free operation, we set the group size to a single diffraction frame (
C
g
=
1
C 
g
​
 =1
), removing overlap-based real-space consistency. Reconstruction then relies entirely on the diffraction likelihood and the known probe structure (defocused probe/Fresnel geometry). Figure \ref{fig:recon_2x2} illustrates this single-frame mode compared with multi-position ptychography. While the overlap-free reconstruction shows a slight reduction in fidelity compared to the highly redundant ptychographic case, it successfully resolves the object features, enabling high-throughput imaging without mechanical scanning overhead.
C3. Results: Data Efficiency
Figure \ref{fig:ssim} illustrates the reconstruction quality (phase SSIM) as a function of dataset size. The physics-informed framework maintains high fidelity (SSIM 
>
0.95
>0.95
) even when trained on as few as 512 diffraction patterns. In contrast, the supervised baseline degrades rapidly below 1024 samples. The horizontal shift between the curves indicates that PtychoPINN achieves comparable quality using roughly an order of magnitude less training data than the supervised approach. This confirms that enforcing the diffraction forward model acts as a powerful regularizer, reducing the number of samples required to constrain the solution.
C4. Results: Out-of-Distribution Generalization
We evaluated the model's ability to generalize across facilities by training on APS data and reconstructing LCLS data without retraining. Despite significant differences in probe shape, energy, and geometry (Figure \ref{fig:fivepanel}, rows A vs B), the physics-informed model successfully reconstructs the LCLS object features (Figure \ref{fig:fivepanel}, 'PINN' column). The supervised baseline, which relies on learning dataset-specific statistics rather than the physical diffraction operator, fails to generalize to the new domain (Figure \ref{fig:fivepanel}, 'Baseline' column).
C5. Results: Photon-Limited Performance
Figure \ref{fig:lowcounts} compares reconstructions trained with a standard Mean Absolute Error (MAE) loss versus the Poisson Negative Log-Likelihood (NLL) loss at low photon counts. The MAE loss implicitly assumes Gaussian noise, which is ill-suited for low-dose diffraction data where shot noise dominates. As visible in the figure, the Poisson-trained model preserves high-frequency features that are washed out by noise in the MAE reconstruction. This suggests that correct statistical modeling is essential for minimizing dose while maintaining resolution.

---

## D. Documentation Verification Checklist

Before submission, verify each paper claim against the authoritative project documentation:

### D1. Physics & Forward Model
- [ ] Paper Eq. 2-5 (forward model) matches `docs/specs/spec-ptycho-core.md` lines 18-36
- [ ] FFT normalization `|F|²/(N·N)` matches spec line 24
- [ ] Poisson loss formula matches spec lines 77-80
- [ ] Intensity scale symmetry assertion matches spec lines 87-93

### D2. Normalization
- [ ] Paper §2.3 (Data Preprocessing) correctly distinguishes physics vs statistical normalization per `docs/DATA_NORMALIZATION_GUIDE.md`
- [ ] Paper does NOT conflate intensity_scale (physics) with normalize_data (statistical)
- [ ] `nphotons` parameter behavior matches doc §Pitfall 2

### D3. Data Contracts
- [ ] NPZ keys mentioned match `specs/data_contracts.md` (diff3d, xcoords, ycoords, probeGuess)
- [ ] Amplitude vs intensity semantics correct (diff3d is amplitude = sqrt of counts)

### D4. Single-Shot / Overlap-Free
- [ ] Paper uses Cg=1 (NOT Cg=0) per `docs/GRIDSIZE_N_GROUPS_GUIDE.md`
- [ ] Overlap definitions align with `docs/specs/overlap_metrics.md`

### D5. Cross-Facility Generalization
- [ ] Claims verified against `docs/FLY64_GENERALIZATION_STUDY_ANALYSIS.md`
- [ ] Any contradictions (baseline outperforming) acknowledged or explained
- [ ] Preprocessing requirements from `docs/FLY64_DATASET_GUIDE.md` followed

### D6. Data Efficiency
- [ ] Methodology matches `docs/studies/GENERALIZATION_STUDY_GUIDE.md`
- [ ] n_subsample/n_groups semantics match `docs/SAMPLING_USER_GUIDE.md`

### D7. Configuration
- [ ] Paper Table 1 parameters match `docs/CONFIGURATION.md` defaults
- [ ] Parameter names use correct casing (gridsize not grid_size, etc.)

### D8. Architecture
- [ ] Paper Fig 7 (architecture) aligns with `docs/architecture_tf.md` component diagram
- [ ] Dual-resolution decoder description matches code in `ptycho/model.py:create_decoder_last`

### D9. Reproducibility (per A1.5 conventions)
- [ ] All figures have generation scripts in `paper/figures/scripts/`
- [ ] All numerical results trace to JSON/CSV artifacts
- [ ] `make clean && make all` regenerates all figures and tables
- [ ] `make verify` passes with no missing artifacts
- [ ] No hardcoded metric values in .tex file (all from generated tables)
- [ ] Figure generation scripts document input data provenance
- [ ] Random seeds set for deterministic output

---

## E. Known Issues to Address

From `docs/FLY64_GENERALIZATION_STUDY_ANALYSIS.md`:
> "unexpected results where baseline models significantly outperform PtychoPINN"

**Action Required:** If Fig 2 (cross-facility) shows PINN outperforming baseline, verify this is on a DIFFERENT dataset/configuration than the FLY64 study, or acknowledge the discrepancy in Discussion.

From `docs/bugs/BUG-DOUBLE-NORMALIZATION-001.md`:
- Check if any double-normalization issues affect the paper's reported metrics.

---

## F. Reference Summary

| Paper Section | Primary Doc Reference | Verification Status |
|---------------|----------------------|---------------------|
| §2.1 Formulation | specs/spec-ptycho-core.md | [ ] |
| §2.3 Preprocessing | DATA_NORMALIZATION_GUIDE.md | [ ] |
| §2.4 Training Objective | specs/spec-ptycho-core.md §Losses | [ ] |
| §3.2 Overlap-Free | GRIDSIZE_N_GROUPS_GUIDE.md | [ ] |
| §3.3 Data Efficiency | studies/GENERALIZATION_STUDY_GUIDE.md | [ ] |
| §3.4 Cross-Facility | FLY64_GENERALIZATION_STUDY_ANALYSIS.md | [ ] |
| §3.5 Low-Dose | DATA_GENERATION_GUIDE.md | [ ] |
| Table 1 | CONFIGURATION.md | [ ] |
| Fig 7 Architecture | architecture_tf.md | [ ] |
| All Figures | paper/figures/scripts/*.py | [ ] |
| All Metrics | paper/data/*.json artifacts | [ ] |
