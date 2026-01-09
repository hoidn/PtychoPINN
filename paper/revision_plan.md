Revised P0 + P1 Revision Plan (With Codebase & Generalization)
Objective: Submit a manuscript that claims Fast, Physics-Informed, Generalizable, and Single-Shot capabilities, using only existing data/figures.
A. Self-contained revision tasks (Do these first)
A0. Setup
Create a clean branch/folder.
Ensure main.tex compiles.
A1. Lock the Scope (Updated with Verification Sources)
Keep:
Methods: Core physics + Dual-resolution decoder.
Codebase Ref: ptycho/model.py (Search create_decoder_last to see probe.big logic verifying the dual-resolution claim).
Codebase Ref: ptycho/tf_helper.py (Search pad_and_diffract to verify the Far-Field/FFT physics model).
Single-Shot: The Fresnel/Overlap-free argument (Fig 1).
Codebase Ref: docs/GRIDSIZE_N_GROUPS_GUIDE.md (Read "GridSize = 1" section to confirm how n_groups works without overlap).
Data Efficiency: The SSIM vs Training Size argument (Fig 4).
Codebase Ref: scripts/studies/run_complete_generalization_study.sh (Source of the training size sweep logic).
Cross-Facility Generalization: APS 
→
→
 LCLS transfer (Fig 2).
Low-Dose: The Poisson vs MAE visual argument (Fig 5).
Codebase Ref: scripts/studies/dose_response_study.py (Source of the specific High vs Low dose comparison logic).
Cut:
The separate "Sparse Sampling Robustness" subsection (3.3.1).
The "Photon-Limited Performance" Table (Table ??).
Specific speedup claims (e.g., "1000x") unless you have a citation.
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
