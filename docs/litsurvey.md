# Literature Survey: Local/Spectral FNO Blocks and Hybrid FNO/CNN Decoders

## Scope
This note captures literature matches for two internal design questions:
1. **Local + spectral mixing** (e.g., `PtychoBlock` = spectral conv + local 3×3 conv + residual).
2. **Hybrid FNO encoder + CNN-style decoder** (e.g., `hybrid_resnet`).

## Findings

### Local + Spectral Mixing Inside FNO Layers
- **Standard FNO layers already include a local term.** The FNO operator is defined as an integral kernel plus a local linear transform \(W v(x)\), which is added to the Fourier-convolution output in real space. This is the canonical “spectral + local” structure, though the local term is usually pointwise (1×1).  
  Source: NeuralOperator theory guide (FNO).  
  https://neuraloperator.github.io/dev/theory_guide/fno.html

- **LocalNO / LocalFNO explicitly add localized kernels in parallel to Fourier layers.** The LocalNO architecture replaces Fourier convolution layers with blocks that place **differential** and **local integral** kernels in parallel to the Fourier layers.  
  Source: NeuralOperator LocalNO docs.  
  https://neuraloperator.github.io/dev/modules/generated/neuralop.models.LocalNO.html

- **Localized Integral/Differential Kernels (ICML 2024)** provide the theoretical and empirical motivation for adding local kernels to FNOs to capture local details and reduce over-smoothing.  
  Source: PMLR (ICML 2024).  
  https://proceedings.mlr.press/v235/liu-schiaffini24a.html

**Interpretation:** `PtychoBlock`’s spectral + **3×3** local conv + residual is a stronger local inductive bias than the standard FNO’s pointwise local term, but conceptually aligns with published work that augments FNOs with localized kernels.

### Hybrid FNO + CNN / U‑Net Structures
- **U‑FNO** introduces U‑Net structure inside the Fourier layer (a “U‑Fourier layer”), explicitly combining spectral operators with local CNN blocks.  
  Source: Scientific Reports (diagram + description of U‑FNO architecture).  
  https://www.nature.com/articles/s41598-024-72393-0

- **HUFNO** (Hybrid U‑Net + FNO) explicitly combines U‑Net and FNO components to handle mixed periodic/non‑periodic boundary conditions.  
  Source: arXiv 2504.13126.  
  https://arxiv.org/abs/2504.13126

- **FourierUnet (PDEArena)** uses Fourier blocks in the **downsampling path** of a U‑Net, i.e., a spectral encoder + CNN decoder structure with skips.  
  Source: PDEArena architecture listing.  
  https://pdearena.github.io/pdearena/architectures/

**Interpretation:** These are conceptual matches to “FNO encoder + CNN decoder” designs, but none are exact matches to `hybrid_resnet` (FNO encoder → ResNet‑6 bottleneck → CycleGAN upsamplers). The surveyed references suggest the hybridization pattern is common, even if the specific ResNet‑6 + CycleGAN decoder is bespoke.

## Conclusion
We did **not** find a direct prior implementation identical to `hybrid_resnet`. However, there is strong precedent for:
- augmenting FNO layers with **local kernels** (LocalNO / LocalFNO),
- hybridizing FNO components with **U‑Net / CNN** structures (U‑FNO, HUFNO, FourierUnet).

This supports the conceptual validity of the local‑plus‑spectral design and the broader FNO‑encoder + CNN‑decoder pattern, while leaving the specific `hybrid_resnet` decoder as a novel combination.
