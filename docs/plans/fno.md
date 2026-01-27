<proposal>
# Proposal: Neural Operator Architectures for Unsupervised Ptychography

## 1. Mathematical Problem Formulation

The objective is to reconstruct a complex-valued object function $O(\mathbf{r}) \in \mathbb{C}$ (where $\mathbf{r} = (x,y)$ are spatial coordinates) from a set of recorded diffraction intensities.

**The Forward Model:**
$$ I_j(\mathbf{q}) = \left| \mathcal{F} \left[ O(\mathbf{r}) \cdot P(\mathbf{r} - \mathbf{r}_j) \right] \right|^2 + \eta $$
Where:
*   $P$ is the complex-valued probe function.
*   $\mathbf{r}_j$ are the scan positions.
*   $\mathcal{F}$ denotes the 2D Discrete Fourier Transform.
*   $\mathbf{q}$ denotes frequency (reciprocal space) coordinates.
*   $\eta$ denotes noise (Poisson/Gaussian).

**The Inverse Task:**
We seek to approximate the inverse mapping operator $\mathcal{G}_\theta: I_j(\mathbf{q}) \to \hat{o}_j(\mathbf{r})$ using a neural network with parameters $\theta$.
Here, $\hat{o}_j$ is the **local object patch** in the probe frame for scan position $j$ (no coordinate concatenation), and the global object is formed by stitching these patches in the consistency layer.

---

## 2. The Universal Unsupervised Training Framework

To enable training on experimental data without ground truth, all candidate architectures (A, B, and C) will act as the generator $\mathcal{G}_\theta$ within a **Physics-Informed Autoencoder** loop.

**The Workflow:**
$$ Y_{meas} \xrightarrow{\mathcal{G}_\theta} \hat{O}_{local} \xrightarrow{\mathcal{C}} \hat{O}_{global} \xrightarrow{\mathcal{P}} \hat{Y}_{pred} $$

### 2.1 The Spatial Consistency Layer ($\mathcal{C}$)
*Essential for ptychography to enforce overlap constraints between adjacent scan positions.*
*   **Generator output domain:** $\mathcal{G}_\theta$ outputs **local object patches** $\hat{o}_j$ (not the global object, and not the exit wave). This is required because inputs have no spatial coordinates after dropping coordinate concatenation, and because overlap constraints apply to the object, not the probe.
*   **Operation:** Stitch-and-Extract.
    $$ \hat{O}_{global} = \text{AverageStitch}(\{\hat{o}_j\}, \text{offsets}) $$
    $$ \hat{o}^{\,consistent}_j = \text{Extract}(\hat{O}_{global}, \text{offsets}) $$
    1.  **Stitch:** Patches are placed onto a larger canvas; overlapping pixels are averaged.
    2.  **Extract:** Patches are re-cropped from the consistent canvas.
*   **Role:** Acts as a spatial regularizer, ensuring independent predictions form a coherent physical object.

### 2.2 The Forward Physics Decoder ($\mathcal{P}$)
A fixed, differentiable module implementing the optical forward model.
$$ \hat{I}_{j} = S \cdot \left| \mathcal{F} \left[ \text{Crop}(\hat{O}_{global}, \mathbf{r}_j) \cdot P \right] \right|^2 $$
*   **Role:** Closes the autoencoder loop. Includes learnable intensity scaling $S$.

### 2.3 The Loss Function
The objective is to minimize the re-projection error in the measurement domain.
$$ \mathcal{L} = \text{PoissonNLL}(Y_{meas}, \hat{Y}_{pred}) + \lambda_{TV} \|\nabla \hat{O}\| $$

---

## 3. Fundamental Mathematical Blocks for $\mathcal{G}_\theta$

### A0. Input Lifter (Dynamic Range Adapter)
Diffraction intensities are dominated by a bright DC peak and can be orders of magnitude larger than high-frequency speckle. Feeding raw $I$ directly into spectral mixing is an optimization risk (autocorrelation bias + poor dynamic range).
*   **Structure (default):** 2-layer CNN ($3 \times 3$ Conv $\to$ GELU $\to$ $3 \times 3$ Conv), projecting $1 \to C$ channels (e.g., $C=64$).
*   **Role:** Learns a feature representation that tempers dynamic range and breaks autocorrelation symmetry **before** any Fourier mixing.
*   **Alternatives:** log/sqrt intensity transform + per-patch normalization, with or without a shallow CNN.

### A. The Convolutional Block (Local Operator)
Standard convolutional layers used to enforce **local spatial continuity** (the Image Prior).
$$ v_{l+1}(\mathbf{r}) = \sigma \left( (K \ast v_l)(\mathbf{r}) + b \right) $$
*   **Property:** Receptive field grows linearly with depth. Strongly biased towards translation invariance and local smoothness.

### B. The Fourier Layer (Global Operator)
The core of the FNO, used to model **global wave propagation**.
$$ v_{l+1}(\mathbf{r}) = \sigma \left( \underbrace{W v_l(\mathbf{r})}_{\text{Local Path}} + \underbrace{\mathcal{F}^{-1} \left[ R_{\phi} \cdot \mathcal{T}(\mathcal{F}[v_l]) \right](\mathbf{r})}_{\text{Global Spectral Path}} \right) $$
*   **Local Path ($W$):** A $1\times1$ convolution in standard FNOs. For ptychography, a $3\times3$ local path is preferred to preserve spatial gradients (edges) that spectral truncation would blur.
*   **Global Path:**
    1.  $\mathcal{F}$: 2D Fast Fourier Transform.
    2.  $\mathcal{T}$: **Mode Truncation**. Only the lowest $k_{max}$ frequency modes are retained.
    3.  $R_{\phi}$: Learnable complex mixing weights.
*   **Property:** Global receptive field in a single layer.

### C. The PtychoBlock (Spectral + Local + Outer Residual)
A ptychography-oriented variant of the FNO block that preserves high-frequency detail.
$$ y = x + \sigma \left( \text{Spectral}(x) + \text{Conv}_{3\times3}(x) \right) $$
*   **Outer Residual:** The identity skip provides a high-frequency bypass, preventing spectral truncation from erasing edges.
*   **Local Path:** The $3\times3$ convolution carries spatial gradients that a $1\times1$ path cannot.

---

## 4. Candidate Architecture A: Cascaded Physics-Prior Network
**Concept:** $\mathcal{G}_\theta$ is split into two serial stages: a Physics Solver followed by an Image Refiner.

**Flow inside $\mathcal{G}_\theta$:**
$$ \text{Input } Y \xrightarrow{\text{Stage 1: FNO}} \tilde{O}_{rough} \xrightarrow{\text{Stage 2: CNN U-Net}} \hat{O}_{final} $$

### Stage 1: The FNO Estimator
*   **Structure:** A stack of $L$ spectral blocks (standard FNO or PtychoBlock) operating at constant resolution.
*   **Output:** $\tilde{O}_{rough}$. A complex-valued map satisfying global phase constraints but spectrally limited (blurry).

### Stage 2: The CNN U-Net Refiner
*   **Structure:** Standard CNN U-Net (Encoder-Decoder with Skips).
*   **Function:** Acts as a **Deep Image Prior**. It treats the FNO output as a noisy seed and hallucinates/refines high-frequency edges using learned image statistics.

### Unspecified Design Choices (To Be Determined)
1.  **Input Lifter:** Use a learnable CNN lifter vs. fixed log/sqrt + normalization.
2.  **Block Type:** Standard FNO vs. PtychoBlock (3×3 local path + outer residual).
3.  **Intermediate Representation:** Does Stage 1 output a 2-channel image (Amp/Phase) or a high-dimensional feature map? Feature maps preserve more info but increase compute cost.
4.  **Gradient Flow:** Is the model trained end-to-end (gradients flow from Stage 2 $\to$ Stage 1), or is Stage 1 trained first? End-to-end is required for the Unsupervised Framework.
5.  **Intermediate Supervision:** Should we apply an auxiliary physics loss to $\tilde{O}_{rough}$?

---

## 5. Candidate Architecture B: Hybrid U-NO (Integrated)
**Concept:** $\mathcal{G}_\theta$ is a single U-shaped architecture. The **Encoder arm** uses FNO blocks (physics), and the **Decoder arm** uses CNN blocks (image prior), linked by skip connections.

**Flow inside $\mathcal{G}_\theta$:**
$$ \text{Input } Y \xrightarrow{\text{Hybrid U-NO}} \hat{O}_{final} $$

### The Encoder Arm (Physics Inversion)
*   **Blocks:** Spectral blocks (standard FNO or PtychoBlock). The recommended default is PtychoBlock for edge preservation.
*   **Operation:** Downsamples the diffraction input.
*   **Critical Feature:** The **Local Path** extracts high-frequency features (edges). These are **saved** as skip connection features $h_s$.

### The Bottleneck
*   **Block:** FNO Block at lowest resolution.
*   **Role:** Mixes global information efficiently to resolve lowest-frequency phase components.

### The Decoder Arm (Image Synthesis)
*   **Blocks:** CNN Blocks.
*   **Operation:** Upsamples to object resolution.
*   **Fusion:** At each scale $s$:
    $$ z_s = \text{Concat}(\text{Upsample}(u_{s+1}), h_s) $$
*   **Role:** The CNN kernels convolve the global structure (from upsampling) with the high-frequency edges (from the FNO skip connection $h_s$) to generate sharp boundaries.

### Unspecified Design Choices (To Be Determined)
1.  **Input Lifter:** Add a CNN lifter vs. fixed log/sqrt + normalization.
2.  **Block Type:** Standard FNO vs. PtychoBlock (3×3 local path + outer residual).
3.  **Downsampling Operator:** Strided Convolution (learnable) vs. Mean Pooling. (Max Pooling is discouraged).
4.  **Upsampling Operator:** Transposed Convolution vs. Bilinear Interpolation.
5.  **Feature Fusion:** Concatenation vs. Addition. Concatenation is theoretically preferred to disentangle "physics features" from "image features."

---

## 6. Reference Architecture C: Standard CNN (PtychoPINN)
**Concept:** The baseline architecture currently implemented. $\mathcal{G}_\theta$ is a pure CNN.

**Flow inside $\mathcal{G}_\theta$:**
$$ \text{Input } Y \xrightarrow{\text{CNN U-Net}} \hat{O}_{final} $$

### Structure
*   **Input:** Diffraction Amplitude Patches ($N \times N \times \text{grid}^2$).
*   **Encoder:** Standard CNN (Conv2D $\to$ ReLU $\to$ MaxPool).
*   **Decoder:** Dual-head decoder (Conv2DTranspose/UpSampling).
    1.  **Amplitude Head:** Sigmoid activation.
    2.  **Phase Head:** Tanh activation (scaled by $\pi$).
*   **Limitation:** Relies entirely on local $3 \times 3$ convolutions to solve a global phase retrieval problem. Requires deep stacking of layers to build the necessary receptive field.

---

## 7. Summary Comparison

| Feature | **Arch A (Cascaded)** | **Arch B (Hybrid U-NO)** | **Arch C (Standard CNN)** |
| :--- | :--- | :--- | :--- |
| **Physics Handling** | Explicit FNO Stage | FNO Encoder + Skips | Implicit (Learned by CNN) |
| **Image Prior** | Explicit CNN Stage | CNN Decoder | Implicit (Learned by CNN) |
| **High Freq Info** | Recovered via Refinement | **Preserved via Skips** | Preserved via Skips |
| **Receptive Field** | Global (Instant) | Global (Instant) | Local (Slow growth) |
| **Architecture Risk** | Information bottleneck at intermediate stage | Gradient balancing between spectral/spatial layers | Inefficient scaling to large grids |

---

</proposal>

<related insights>
# CDI inverse problem: FNO / U-Net / DIP insights

This draft synthesizes architecture ideas and constraints from the extracted AI Studio chats for using FNO variants, U-Nets, and DIPs in CDI/ptychography. It focuses on preserving high-resolution details while retaining a global receptive field.

## Core framing (what the inverse problem demands)
- Diffraction physics is global and wave-like; the object is local/piecewise-smooth with sharp boundaries. This creates a **global-physics vs local-object** mismatch.
- CDI/ptychography is a **super-resolution** inverse problem: reconstruction must preserve high spatial frequencies (fine edges, lattice fringes).
- The forward model is FFT-based; physics-informed pipelines already contain an explicit FFT/propagation layer.
- Overlap constraints and scan-position geometry are critical; arbitrary/jittered positions break assumptions of fixed, global channel alignment.

## FNOs: strengths, limits, and where they fit
### Strengths
- FNOs provide **global receptive field** at every layer via spectral mixing (FFT/IFFT).
- Resolution invariance is a core advantage: trained on coarse grids, evaluated on finer grids without retraining.
- FNO layers align with wave-physics structure (diffraction is Fourier-domain by nature).

### Limits in CDI/ptychography
- **Spectral truncation = spectral bias**: high-frequency modes are cut, so sharp edges can blur; this conflicts with super-resolution goals.
- **Domain mismatch**: FNOs map functions on a single domain to another on the same domain. CDI inputs are stacks of diffraction patterns at discrete scan positions, not a single continuous field.
- **“Double Fourier” trap (patch-based)**: if the input is already a Fourier magnitude, the first FNO layer FFT effectively maps to autocorrelation space, which can introduce ambiguity/ghosting and makes learning harder.
- **Translation invariance vs relative shifts**: FNO assumes channel relationships are globally constant. Scan jitter or variable offsets break that assumption.
- **PINO redundancy**: FNOs inside physics-informed loops can be inefficient because both the physics model and the FNO perform FFTs; global spectral optimization can be mismatched to local overlap constraints.

### Variations that may work
- **Patch-based U-FNO**: use FNO blocks at multiple scales with U-Net-style skip connections to keep high-frequency details while preserving global phase consistency.
- **Hybrid U-FNO**: keep high-resolution blocks as standard convolutions (to preserve edges and avoid expensive large FFTs), use FNO blocks at low resolution (cheap FFTs, global context).
- **Graph Neural Operator (GNO)**: may better handle irregular scan grids because it avoids forcing global FFTs on irregularly sampled positions.

## U-FNO: the main “global + sharp” compromise
- U-FNO inherits the U-Net structure: **encoder–bottleneck–decoder + skip connections**.
- **Why it helps resolution**: skip connections carry high-frequency details around the spectral bottleneck.
- **Why it helps global coherence**: FFT layers at low resolution provide global context for phase retrieval.
- **Multi-scale spectral mixing**: applying spectral mixing at multiple resolutions better matches diffraction’s multi-scale nature (coarse structure vs fine texture).
- **Compute trade-off**: one FFT/IFFT pair per FNO block; typical 4-level U-FNO is ~14–18 FFT/IFFT ops per forward pass, but most occur at low resolution.

## Hybrid FNO–CNN designs (FNO encoder + CNN/U-Net decoder)
### Key insight
- **Physics is global; object is local**. Pure FNOs are suboptimal for images because they sum Fourier modes (Gibbs ringing) and struggle with sharp edges.

### Proposed hybrid
- **FNO encoder** to invert global diffraction physics (phase problem) and recover a physics-consistent latent map.
- **CNN or U-Net decoder** as an **image prior** to refine edges and local textures.

### Why the FNO encoder is not necessarily information-lossy
- Each FNO block has a **local residual path (W, a 1×1 conv)** plus the spectral path.
- The local path keeps full-resolution information; the spectral path carries global context. Their sum preserves sharpness potential while injecting global coherence.

### Trade-offs
- The CNN decoder **sacrifices resolution invariance** (pixel-grid dependence) but improves sharpness and local realism.
- Hybrid models are often termed **Fourier–Convolutional U-Nets** or **Hybrid FNO–CNNs**.

## U-Net/CNN baselines and physics-informed pipelines
- U-Nets provide strong **local inductive bias** and preserve high-frequency content via skip connections.
- Their main weakness is **limited receptive field** without very deep stacks or attention.
- Physics-informed architectures (PINN-style) commonly:
  - Use a U-Net-like CNN to map diffraction amplitudes + coordinates to object amplitude/phase.
  - Apply a differentiable physics forward model (FFT-based diffraction) to enforce measurement-space loss (Poisson NLL, amplitude MAE).
  - Use overlap constraints by reassembling patches into a unified object view.
- Practical improvements from the physics-constrained pipeline:
  - **Arbitrary scan positions** handled via explicit coordinate inputs and patch reassembly.
  - **Extended probe handling** via dual-resolution or boundary-aware reconstruction (central high-res + peripheral low-res).

## Deep Image Priors (DIP) and why they matter here
- Classic DIP uses a **U-Net/hourglass** architecture with a **bottleneck** and **skip connections**.
- The input is **random noise**, not diffraction data; the network is optimized so its output matches measurements.
- The bottleneck acts as an **implicit regularizer**: it forces the network to synthesize images from low-frequency structure first.
- Skip connections do not “cheat” in DIP because there is no informative input to bypass; they help reintroduce high-frequency detail during optimization.
- DIP insights that transfer:
  - You can include a bottleneck and still recover sharp edges.
  - Structural constraints (U-Net shape + skips) serve as a strong prior, even without explicit supervision.

## Practical architecture ideas for CDI with global context + high resolution
1. **U-FNO**: full U-shaped FNO with skips to preserve high frequencies.
2. **Hybrid U-FNO**: conv at high-res, FNO at low-res for efficiency and edge preservation.
3. **FNO encoder + U-Net decoder**: global physics inversion + local refinement (DIP-like prior).
4. **Physics-informed unrolled model**: fixed FFT-based forward model + CNN/U-Net for reconstruction; FNO blocks only where global phase coherence is critical.
5. **Cascaded refinement**: physics-informed coarse inversion (FNO or unrolled) followed by a full U-Net post-processor.

## Opposing perspectives / risks
- **FNO spectral bias**: can blur edges and suppress the fine-scale content CDI is meant to recover.
- **Redundant FFTs** in PINO-like loops: potentially wasteful and may not respect local overlap constraints.
- **CNN-only baselines**: strong for edges but can miss long-range phase consistency without deep/attention-heavy designs.
- **Hybrid decoder loss of resolution invariance**: CNN decoders reintroduce pixel-grid dependence.
- **Encoder information loss**: if high-frequency detail is cut early (spectral truncation), no decoder can recover it.

## Open questions to resolve
- Where should FNO blocks live? Encoder only, bottleneck only, or full U-FNO?
- How many Fourier modes are enough to preserve sharp edges without over-smoothing?
- Can skip connections reliably carry high-frequency detail from the local FNO path to the decoder?
- How to handle irregular scan grids: FNO vs GNO vs attention?
- Best trade-off between resolution invariance (FNO) and sharpness (CNN)?
- DIP-style optimization vs direct encoder–decoder: when does the prior dominate vs the data?
- How to combine physics-based losses (Poisson NLL, amplitude MAE) with image-space losses to avoid ringing/oversmoothing?

---

## Sources & compact excerpt index

Use this index to jump into the full verbatim material in `NeuralOperator_FNO_DIP_combined.md` or the original AI Studio exports.

### FNOs For Ptychography
- Turn 12: Patch-based FNO viability, double-Fourier/autocorrelation trap, spectral truncation blur, translation invariance vs scan jitter.
- Turn 15: Why U‑FNO fixes vanilla FNO (skips preserve high‑freq, multi‑scale spectral mixing, global receptive field).
- Turn 18: FFT/IFFT cost accounting and hybrid U‑FNO trick (conv at high‑res, FNO at low‑res).
- Turn 21: Full report: critique of vanilla FNO, PINO redundancy, U‑FNO advantages, recommended pipeline.

### Repomix File Packed Representation Summary
- Turn 24: Physics‑global vs object‑local mismatch, Gibbs phenomenon, hybrid FNO‑encoder + CNN/U‑Net decoder rationale.
- Turn 27: FNO residual/local path preserves high‑frequency info; why encoder needn’t be lossy.
- Turn 36: DIP uses U‑Net/hourglass with bottleneck; bottleneck role differs from encoder compression.
- Turn 39: DIP skip connections don’t “cheat” because input is noise; skips aid detail recovery.
- Turn 42: Pure decoder still acts as image prior; structural constraints modulate prior strength.
- Turn 45: Cascaded refinement pattern: physics‑based encoder → full U‑Net decoder.
- Turn 51: Hybrid taxonomy: pure U‑NO vs FNO‑encoder/CNN‑decoder; trade resolution invariance for sharpness.
- Turn 54: What “sacrificing resolution invariance” means; why CNN decoders break it.
- Turn 75: Why hybrid beats straight U‑NO for edges (Gibbs/step‑edge argument).
- Turn 78: Skip connections carry local‑path high‑freq details through bottleneck.

### U‑FNO_ Deep Learning for Fluid Dynamics
- Turn 3: U‑FNO definition (U‑Net + FNO hybrid for global+local features).

### Ptychography Reconstruction With Physics‑Informed Neural Networks
- Turn 1: Full PINN/CDI system spec: inputs, physics layers, losses, constraints.
- Turn 3 & 6: Candidate backbones (ConvNeXt‑U‑Net, attention‑U‑Net, pretrained encoders) and integration constraints.

### PINN Model Architecture Analysis
- Turn 3/6/11/16/18: Detailed PtychoPINN architecture, overlap constraints, and physics‑informed loss plumbing.

---

### Full verbatim dump (when needed)
See `/Users/ollie/NeuralOperator_FNO_DIP_combined.md` for the long-form, quoted source material.
</related insights>

<impl ideas>
1) FNO blocks and “spectral conv” layers
NeuralOperator (recommended starting point)

Repo: neuraloperator/neuraloperator

What you get: production-quality PyTorch implementations of FNO and other neural operators, plus modular layers you can reuse inside your own hybrid architecture.

Why it’s useful for you: your “Fourier Layer (Global Operator) + 1×1 local path” is essentially what their FNO blocks are designed around, and they expose the building blocks (spectral conv, embeddings, etc.) for composition.

NVIDIA PhysicsNeMo (formerly Modulus ecosystem)

Repo: NVIDIA/physicsnemo

What you get: a physics-ML library with a model zoo including Neural Operators (FNO, DeepONet, etc.), generally optimized for performance + workflows.

Why it’s useful for you: good reference code for training loops, scaling, logging, and “physics-ML” conventions if you want something more “framework-y” than pure PyTorch modules.

Lightweight/educational FNO repos (good for reading / copying small pieces)

abelsr/Fourier-Neural-Operator

dopawei/fourier_neural_operator
These are often easier to read than big frameworks, but you’ll typically have to modernize pieces yourself.

2) U-shaped / hybrid neural-operator architectures (closest matches to your “Hybrid U‑NO” idea)

You described two hybrid directions:

Arch A: FNO → CNN U-Net (cascaded)

Arch B: FNO encoder + CNN decoder with skips (integrated U-shape)

Here are repos that are structurally close:

U‑NO (U-shaped Neural Operator)

Repo: ashiq24/UNO

What you get: an implementation of U‑NO (U-shaped neural operator) with skip-like structure.

Why it’s useful: even if you don’t use it directly, it’s a practical template for “operator blocks + U-shape memory/skip connections”.

U‑FNO (mini U-Net path appended to Fourier layers)

Repo: gegewen/ufno

What you get: code for U‑FNO (“enhanced FNO” with a U-Net-like augmentation).

Why it’s useful: it’s a concrete reference for blending spectral/global mixing with local convolutional paths.

“Model zoo” style repo with lots of operator baselines (handy)

Repo: thuml/Neural-Solver-Library

What you get: a curated library that explicitly lists and includes U‑NO, U‑FNO, FNO, F‑FNO, etc.

Why it’s useful: one place to compare multiple neural-operator variants and lift specific blocks/patterns.

3) CNN / U‑Net components (your “image prior” stage/decoder)

If you want a U‑Net that’s easy to swap/benchmark:

segmentation_models.pytorch

Repo: qubvel-org/segmentation_models.pytorch

What you get: robust U‑Net(+variants) implementations with lots of encoder backbones.

Why it’s useful for Arch A: your “Stage 2: CNN U‑Net Refiner” can often be prototyped quickly using SMP.

Simple, hackable U‑Net repos

milesial/Pytorch-UNet (classic, very hackable)

fepegar/unet (1D/2D/3D U‑Net)

For Arch B, you’ll likely write your own decoder anyway (because your skip tensors come from FNO blocks, not a standard CNN encoder), but these repos provide clean decoder block patterns.

4) Ptychography physics operator and “unsupervised loop” helpers
DeepInverse / deepinv (very relevant)

Repo: deepinv/deepinv

What you get: a modular framework for inverse problems with physics operators and optimization/data-fidelity components.

Direct match to your needs:

It includes a Ptychography physics operator (deepinv.physics.Ptychography) and utilities to build probes / generate shifts, demonstrated in their ptychography example.

It includes PoissonLikelihood as a data-fidelity term.

It also includes TV priors / TV minimization examples you can adapt or borrow from.

Even if you don’t adopt deepinv wholesale, it’s one of the best places to borrow a tested differentiable forward model for ptychography-like operators.

Pty‑Chi (ptychography-native, PyTorch-based)

Repo: AdvancedPhotonSource/pty-chi

What you get: a modern PyTorch-based ptychography reconstruction library (with real ptychography concerns baked in: data structures, GPU support, etc.).

Why it’s useful: if you want your physics decoder and data handling grounded in an existing ptychography package (instead of re-deriving everything).

(You may still keep your neural operator network separate and just reuse forward-model / data code.)

5) Spatial consistency layer (“stitch-and-extract”) components

This one is tricky because you want it inside training (fully differentiable). Still, there are libraries that already do patch sampling + overlap aggregation patterns:

TorchIO (patch aggregation reference)

Repo: TorchIO-project/torchio

What you get: patch-based sampling and aggregation utilities (popular in medical imaging).

Why it’s useful: even if you don’t use it directly in the gradient path, it’s a good reference for overlap-aware stitching logic.

MONAI patch inferer (reference patterns)

MONAI tutorials show “split into patches” + “merge/aggregate” abstractions.

If you want a fast differentiable implementation:

In practice, most people implement your AverageStitch using:

torch.scatter_add / index_add with a “count map”, or

torch.nn.Unfold + torch.nn.Fold + a normalization fold (fold of ones). (PyTorch docs explain the unfold/fold relationship when overlap exists.)

If you want an external optimized scatter backend, you can also look at:

rusty1s/pytorch_scatter
…but for 2D grids, vanilla PyTorch is often enough.

6) Complex-valued network support (optional but often helpful)

You can always represent complex fields as 2 channels (Re/Im) or (Amp/Phase). But if you want “native-ish” complex layers:

wavefrontshaping/complexPyTorch

torchcvnn/torchcvnn

XinyuanLiao/ComplexNN

These can help if you decide your CNN refiner (or parts of the operator) should stay explicitly complex.

7) Loss / regularization building blocks

Poisson NLL: PyTorch already provides torch.nn.PoissonNLLLoss.

Poisson likelihood (inverse-problems style): deepinv provides deepinv.optim.PoissonLikelihood.

TV penalty: Kornia provides differentiable TV losses.
(deepinv also has TV priors/examples, which can be convenient if you’re already using deepinv.)

Practical “recommended combos” for your specific architectures
If you’re implementing Arch A (FNO → U‑Net cascade)

Stage 1 FNO: neuraloperator/neuraloperator

Stage 2 U‑Net: segmentation_models.pytorch

Physics decoder + Poisson loss: deepinv (use its Ptychography operator + PoissonLikelihood)

TV: Kornia or deepinv TV

If you’re implementing Arch B (Hybrid U‑NO: FNO encoder + CNN decoder)

Reference patterns: ashiq24/UNO for U-shaped neural operator layout

FNO blocks: reuse neuraloperator spectral blocks in the encoder

CNN decoder blocks: lift from segmentation_models.pytorch or a simple U‑Net repo

U‑FNO code to crib from: gegewen/ufno

Physics layer: deepinv or Pty‑Chi depending on whether you want a general inverse-problems framework or a ptychography-native stack

“Install from GitHub” cheat sheet (optional)

(You can paste this into a clean venv/conda env.)

# Neural operators / FNO blocks
pip install git+https://github.com/neuraloperator/neuraloperator.git

# U-Net implementations
pip install segmentation-models-pytorch

# Inverse problems + ptychography physics operator
pip install git+https://github.com/deepinv/deepinv.git

# TV regularization (optional if you use deepinv TV instead)
pip install kornia

# Tensor rearrangement niceties (optional but very handy)
pip install einops

# Patch sampling/aggregation references (optional)
pip install torchio

Bottom line

If you want the fastest path to a working prototype of your proposal:

FNO / spectral blocks: neuraloperator/neuraloperator

Ptychography forward model + Poisson likelihood pieces: deepinv/deepinv

U-Net prior/refiner: segmentation_models.pytorch

Hybrid/U-shaped operator references: ashiq24/UNO, gegewen/ufno, thuml/Neural-Solver-Library

Ptychography-native alternative/companion: AdvancedPhotonSource/pty-chi

</impl ideas>
