# Intensity Scaling in `RectangularScaledDiffraction`

**Where the output scale constant comes from, and why it forces the learned object to $|O|\approx 0.069$**

Target: `ptycho_torch/model.py:1642`

```python
output_scale = torch.sqrt(1/(probe_scaling**2 * physics_scale + 1e-9))
```

---

## 0. Summary

**Established.**

1. The measured data imposes exactly **one** scalar constraint on the forward model,
   $c\,\|s\odot O\|_{\rm rms} = \sqrt{\bar N_{ph}}$ (§2). Everything else is degenerate.
2. Line 1642 sets $c = \sqrt{A\,\bar N_{ph}}$, where $A \equiv \sum|P_{\rm raw}|^2$ is the
   stored probe's total intensity. The physically-motivated value is $c = \sqrt{\bar N_{ph}}$.
   The factor $\sqrt{A} = 1/\sigma$ is applied **twice** (§3).
3. On both datasets checked the over-scale is $1/\sigma = 14.5$, driving the equilibrium object
   modulus to $|O|_{\rm rms} = \sigma = 0.069$ (§4). This is the reported symptom.
4. The $10^{-9}$ guard term is **not** negligible: $\sigma^2 s_\phi \sim 2\times10^{-9}$, so the
   guard suppresses $c$ by a further 8.8–18.4%, dataset-dependently (§5.1).
5. The polar branch uses no probe factor and lands $|O|$ at $O(1)$, corroborating that
   $\sqrt{\bar N_{ph}}$ is the intended magnitude (§3.3).
6. Inference re-solves the photon constants via VarPro, which is why the final reconstruction
   modulus looks correct despite the network's object being 14.5× small (§5.3).

**Not established.** Whether the split of the residual factor between $|O|$ and $(s_1,s_2)$ is
dominated by the object — argued from optimizer dynamics in §5.2, not measured. Logging
$s_1,s_2$ during training would settle it.

**Open decision.** Whether `probeGuess` should be the calibration anchor (§6). The repository's
own synthetic generator saves an absolutely-calibrated probe; the 2ID data does not. They
disagree by $\sqrt{\bar N_{ph}/A}\approx 106$.

---

## 1. Unit audit

Nothing below works unless the units are pinned down, so these are verified rather than assumed.

### 1.1 `images` is intensity (photon counts), not amplitude

Three independent confirmations:

| Evidence | Location |
|---|---|
| `.round()` applied at load — "Round for non-photon detectors" | `dataloader.py:747` |
| `torch.poisson` applied to the saved stack in datagen | `datagen/datagen.py:672`, `helper.py:827` |
| `PoissonLoss(pred_is_amplitude=False)` in rectangular mode; $I_{\rm pred}$ is compared directly to `x` as the Poisson rate $\lambda$ | `model.py:1545`, `model.py:667` |

So the loss target is $I_{\rm meas}$ in absolute detector counts, and the forward model must
produce counts.

### 1.2 The three constants

Let $I$ denote measured intensity, $N$ the pattern size, and $\langle\cdot\rangle$ the mean over
the normalization group (batch / group / each, per `data_config.normalize`).

**RMS scale** (`helper.py:754`):

$$s_{\rm rms} \;=\; \sqrt{\frac{N^2}{\big\langle \sum_{hw} I^2 \big\rangle}}$$

This normalizes the *encoder input* to unit RMS: $\frac{1}{N^2}\sum (I\,s_{\rm rms})^2 = 1$.
It is applied at `model.py:1232` (`self.scaler.scale(x, input_scale_factor)`) and **nowhere
else** in rectangular mode. It is a preconditioner; it does not participate in the physics.

**Physics scale** (`helper.py:794`):

$$s_\phi \;=\; \frac{1}{\big\langle \sum_{hw} I \big\rangle} \;=\; \frac{1}{\bar N_{ph}}
\qquad [\text{photons}^{-1}]$$

This is the reciprocal mean photon budget per pattern. The docstring's claim — "scales the total
intensity of the image to 1" — is correct.

**Probe scale** (`helper.py:683`, stored at `dataloader.py:718`):

$$\sigma \;=\; \frac{1}{\sqrt{\sum_{hw}|P_{\rm raw}|^2}} \;\equiv\; A^{-1/2},
\qquad A \equiv \sum_{hw}|P_{\rm raw}|^2$$

`normalize_probe` divides the probe in place and returns $\sigma$, so the probe **stored in the
memory map** is

$$P_n \;=\; \sigma\, P_{\rm raw}, \qquad \sum_{hw}|P_n|^2 = 1 .$$

This is the crucial point: $\sigma$ is the *absolute* normalization constant that was consumed
at load time, not a dimensionless residual.

One further multiplier exists — `probe/self.probe_scale` at `model.py:1244`, with
`data_config.probe_scale = 1.0` by default (`config_params.py:29`), so it is currently a no-op.
It is a third knob on the same degeneracy and should be left at unity.

---

## 2. The amplitude budget

`RectangularScaledDiffraction.forward` (`model.py:729`, autograd branch) builds

$$\psi \;=\; c\,\big(s_1\, P_n O_r \;+\; i\, s_2\, P_n O_i\big),
\qquad c \equiv \texttt{output\_scale}$$

then $I_{\rm pred} = \sum_p \big|\mathcal{F}_{\rm ortho}\{\psi_p\}\big|^2$ with `norm='ortho'`.

Parseval's theorem for the orthonormal DFT gives
$\sum_{hw}|\mathcal{F}_{\rm ortho}\{\psi\}|^2 = \sum_{hw}|\psi|^2$, hence

$$\boxed{\;\sum_{hw} I_{\rm pred}
\;=\; c^2 \sum_{hw} |P_n|^2\,\big|s_1 O_r + i s_2 O_i\big|^2
\;=\; c^2 \,\big\langle |s_1 O_r + i s_2 O_i|^2 \big\rangle_{|P_n|^2}\;}$$

where $\langle\cdot\rangle_{|P_n|^2}$ is a probe-intensity-weighted mean whose weights sum to
exactly $1$ (that is what `normalize_probe` bought us).

Matching the data, $\sum_{hw} I_{\rm pred} = \bar N_{ph} = 1/s_\phi$:

$$\boxed{\;c \cdot \big\|s \odot O\big\|_{\rm rms} \;=\; \sqrt{1/s_\phi} \;=\; \sqrt{\bar N_{ph}}\;}
\tag{2.1}$$

**This is the entire physical content.** The data fixes only the *product* of the hard-coded
scale, the learned coefficients, and the object modulus. Whatever $c$ you write down, training
drives $\|s\odot O\|_{\rm rms}$ to the reciprocal.

Two structural notes:

- The reassembly step preserves magnitude. `reassemble_patches_position_real_probe`
  (`helper.py:148`) forms $\big(\sum_c O_c W_c\big)\big/\big(\sum_c W_c\big)$ — a weighted
  *average*, not a sum — and `extract_channels_from_region` (`helper.py:246`) only translates and
  crops. Neither changes the scale of $O$.
- Choosing $c = \sqrt{\bar N_{ph}}$ is the statement "$|O|\approx 1$ and $s\approx 1$", i.e. a
  physically transparent object with the learned coefficients acting as small corrections. That
  is the only reason to prefer one $c$ over another; (2.1) does not care.

---

## 3. Where line 1642 lands

### 3.1 The value

$$c_{\rm code} \;=\; \sqrt{\frac{1}{\sigma^2 s_\phi}}
\;=\; \frac{1}{\sigma}\sqrt{\frac{1}{s_\phi}}
\;=\; \sqrt{A}\cdot\sqrt{\bar N_{ph}}$$

Compare with the requirement from (2.1) at $|O|\approx 1,\ s\approx 1$:

$$\frac{c_{\rm code}}{c_{\rm required}} \;=\; \frac{\sqrt{A\,\bar N_{ph}}}{\sqrt{\bar N_{ph}}}
\;=\; \sqrt{A} \;=\; \frac{1}{\sigma}$$

and therefore the equilibrium the optimizer is being pushed towards is

$$\big\|s\odot O\big\|_{\rm rms} \;=\; \frac{\sqrt{\bar N_{ph}}}{c_{\rm code}} \;=\; \sigma
\;=\; \frac{1}{\sqrt{A}} .
\tag{3.1}$$

Note this ratio is **independent of the photon count** — it depends only on the probe's stored
norm. It is a pure bookkeeping factor, not a physics mismatch.

### 3.2 Why the probe factor cancels

The intuition behind wanting both terms is sound: the probe was renormalized at load time, so
something must undo it; and the detector counts must be honoured, so something must supply them.
The correct decomposition makes this explicit:

$$c \;=\; \underbrace{\sigma^{-1}}_{\text{undo load-time normalization}}
\;\times\;
\underbrace{\sqrt{\frac{\sigma^2}{s_\phi}}}_{\text{probe}\,\to\,\text{detector calibration mismatch}}
\;=\; \sqrt{\frac{1}{s_\phi}}
\tag{3.2}$$

**The $\sigma^{-1}$ cancels.** The second factor is the ratio between the photon budget the
detector actually recorded and the photon budget the stored probe implies,
$\sqrt{\bar N_{ph}/A}$ — and it already contains one power of $\sigma^{-1}$. Line 1642 keeps the
first factor and then uses the *full* $\sqrt{1/s_\phi}$ (which is the product of both) as the
second, so $\sigma^{-1}$ enters twice.

Restated: $\sigma^{-1}$ and $\sqrt{1/s_\phi}$ are **two estimates of the same quantity** whenever
the probe is absolutely calibrated — both equal $\sqrt{\bar N_{ph}}$. The code multiplies them
where it should have selected one.

The product would be correct only if `probe_scaling` held a dimensionless residual ratio. It
does not; `helper.py:683` returns the absolute constant.

### 3.3 Corroboration from the polar branch

`PolarForwardModel` (`model.py:911`) uses `output_scale = rms_scale` and divides
(`inv_scale`, `model.py:1202`), giving an effective $c = 1/s_{\rm rms}$ with **no probe factor
at all**:

| dataset | $1/s_{\rm rms}$ | $\sqrt{\bar N_{ph}}$ | $c_{\rm code}$ (rectangular) |
|---|---|---|---|
| 2id_S54 | $1.52\times10^3$ | $1.54\times10^3$ | $2.24\times10^4$ |
| velo_gold_tp_2 | $3.26\times10^2$ | $9.82\times10^2$ | $1.42\times10^4$ |

The polar scale is $O(\sqrt{\bar N_{ph}})$ — it lands $|O|$ within a factor of a few of unity.
(The two columns are not identical: $1/s_{\rm rms} = \sqrt{\langle I^2\rangle}$ and
$\sqrt{\bar N_{ph}} = \sqrt{N^2\langle I\rangle}$ agree only accidentally, as the second row
shows.) The rectangular scale is an order of magnitude above both.

---

## 4. Measured values

Computed from the repository's own data (Appendix A). $A$ and $\bar N_{ph}$ from the first 200
patterns of each file.

| quantity | 2id_S54 | velo_gold_tp_2 |
|---|---|---|
| $A=\sum\|P_{\rm raw}\|^2$ | $2.119\times10^{2}$ | $2.096\times10^{2}$ |
| $\bar N_{ph} = \langle\sum I\rangle$ | $2.368\times10^{6}$ | $9.638\times10^{5}$ |
| $\sigma^2 = 1/A$ | $4.720\times10^{-3}$ | $4.771\times10^{-3}$ |
| $s_\phi = 1/\bar N_{ph}$ | $4.222\times10^{-7}$ | $1.038\times10^{-6}$ |
| $\sigma^2 s_\phi$ | $1.993\times10^{-9}$ | $4.951\times10^{-9}$ |
| required $c = \sqrt{\bar N_{ph}}$ | $1.539\times10^{3}$ | $9.817\times10^{2}$ |
| $c_{\rm code}$, no $\epsilon$ | $2.240\times10^{4}$ | $1.421\times10^{4}$ |
| $c_{\rm code}$, with $\epsilon=10^{-9}$ | $1.828\times10^{4}$ | $1.296\times10^{4}$ |
| over-scale $1/\sigma = \sqrt{A}$ | **14.56** | **14.48** |
| equilibrium $\|s\odot O\|_{\rm rms}$, eq. (3.1) | **0.0687** | **0.0691** |

Both datasets land at $|O|\approx 0.069$ — the reported symptom, to within the accuracy of the
$|O|\approx 1$ idealization.

**Calibration state of the stored probes.** $A \approx 212$ against
$\bar N_{ph}\approx 2.4\times10^6$: the 2ID `probeGuess` is short of absolute units by
$\sqrt{\bar N_{ph}/A} \approx 106$. By contrast the repository's synthetic generator saves
`probeGuess = probe * sqrt(intensity_scale_factor)` (`helper.py:662`, returned through
`datagen/datagen.py:700`), which **is** absolutely calibrated: for those files
$A \approx \bar N_{ph}$ and hence $\sigma^{-1}\approx\sqrt{\bar N_{ph}}$ exactly as in §3.2.
The two data sources therefore disagree about what `probeGuess` means, which is most likely how
the double-count stayed hidden.

---

## 5. Consequences

### 5.1 The $10^{-9}$ guard is a live term

From the table, $\sigma^2 s_\phi = 1.99\times10^{-9}$ and $4.95\times10^{-9}$ — the *same order*
as the guard. The suppression factor is

$$\frac{c_\epsilon}{c_{\epsilon=0}} = \sqrt{\frac{\sigma^2 s_\phi}{\sigma^2 s_\phi + 10^{-9}}}
= \begin{cases} 0.816 & \text{(2id\_S54, }-18.4\%\text{)}\\[2pt]
0.912 & \text{(velo\_gold, }-8.8\%\text{)}\end{cases}$$

This is dataset-dependent, so in multi-dataset training each experiment gets a *different*
silent rescale of its object. Any fix must shrink the guard (e.g. $10^{-30}$) or restructure the
expression so the guard sits on a quantity of order unity.

### 5.2 Where the residual factor goes: $|O|$ vs. $(s_1, s_2)$

$s_1, s_2$ are `nn.Parameter(torch.ones(num_datasets))` (`model.py:731-732`), i.e. two scalars
per dataset initialized at 1. Equation (2.1) makes them exactly degenerate with the object's
real and imaginary magnitudes, *separately*. Nothing in the training objective breaks the
degeneracy; only initialization and optimizer dynamics decide the split.

Argument for the object absorbing nearly all of it (not yet measured): under Adam every
parameter moves at $\approx \eta$ per step regardless of gradient magnitude, so $s_1$ takes
$O(1/\eta)$ steps to travel from $1$ to $0.07$, while the decoder's final layer can rescale its
entire output within a few steps by adjusting a weight matrix and bias jointly. The million-
parameter path is simply faster in function space.

**This asymmetric scaling is not a gauge freedom.** A global rescale $O \to \lambda O$ leaves
$\arg O = \arctan(O_i/O_r)$ invariant. An anisotropic one ($s_1 \neq s_2$) does not — it changes
modulus *and* phase, and there is no term in the loss penalizing it. Recommend logging
$s_1, s_2$ per epoch alongside $\|O\|_{\rm rms}$.

### 5.3 Dynamic range at the decoder heads

In rectangular mode with `use_shared_decoder=False`, `x_real` comes from `Decoder_amp` →
`Amplitude_activation` (SiLU by default, `model.py:77`, attached at `model.py:484`) and `x_imag`
from `Decoder_phase` → $\pi\tanh$ (`model.py:38`, attached at `model.py:465`).

- **Real head.** Forced into $O_r \in [0,\ 0.07]$, SiLU operates in its near-origin regime where
  $\mathrm{SiLU}(x) \approx x/2$; producing $0.07$ needs a pre-activation of $\approx 0.14$.
  There is no `1 +` offset anywhere in the rectangular path, so the network must learn the DC
  transmission level itself, at $1/14.5$ of its natural scale.
- **Imaginary head.** $|O_i| \lesssim 0.07$ occupies ~2% of the $(-\pi,\pi)$ range.
- **Phase specifically** is scale-invariant under a *global* shrink, so phase resolution is not
  directly destroyed by $c$ — it is destroyed only if $s_1$ and $s_2$ drift apart (§5.2).

### 5.4 Why reconstructions still look right

`forward_predict` (`model.py:1249`) returns the raw complex object with **no output scaling**, and
`VarProScaler` re-solves the photon constants over the dataset at assembly time. The inference
path therefore re-absorbs the 14.5× and the final canvas modulus is correct. The defect is
visible only in the network's internal object — exactly the regime where the reported symptom
lives.

The same is true of the non-autograd branch of `RectangularScaledDiffraction`
(`model.py:756-775`), which solves $s_1,s_2$ per pattern by variable projection and is immune to
whatever $c$ is passed in.

---

## 6. Fix options

The two candidate anchors disagree by $\sqrt{\bar N_{ph}/A}\approx 106$ on the 2ID data, so this
is a decision, not a lookup.

**Option 1 — data-derived anchor (recommended).**

```python
output_scale = torch.sqrt(1/(physics_scale + 1e-30))
```

Equivalently, keeping (3.2) explicit: `(1/probe_scaling) * torch.sqrt(probe_scaling**2/physics_scale)`.
Trusts the measured photon budget. Correct for both real and synthetic data as currently stored.
Puts $|O|$ at $O(1)$.

**Option 2 — probe-derived anchor.**

```python
output_scale = 1/probe_scaling
```

Trusts `probeGuess` as absolutely calibrated. Correct for `datagen`-produced files; wrong by
$106\times$ for the 2ID data unless its preprocessing is changed to rescale `probeGuess` to
absolute units.

**Independent of the choice:** the $10^{-9}$ guard must shrink (§5.1), and $(s_1, s_2)$ should be
logged (§5.2). Whether to add a fixed `1 +` offset on the real head is a separate question, worth
revisiting only after $c$ is corrected — the offset would be treating the symptom.

### Verification

Log $R \equiv \sum I_{\rm pred} / \sum I_{\rm meas}$ at step 0 with an untrained decoder
($|O|\sim 0.5$):

$$R_{\rm before} \;=\; \frac{c_{\rm code}^2 |O|^2}{\bar N_{ph}} \;=\; A|O|^2 \;\approx\; 53,
\qquad
R_{\rm after} \;=\; |O|^2 \;\approx\; 0.25$$

A two-order-of-magnitude drop in $R$ at initialization is the signature of the fix. Then track
$\|O\|_{\rm rms}$ over training: it should settle near $1$ rather than near $\sigma$.

---

## Appendix A — reproduction

```python
import numpy as np

for f in ["data/2id_S54_singlemode/2id_S54.npz",
          "data/pinn_velo_gold_tp_2/nxs_data_gold_tp_2_64_train.npz"]:
    d = np.load(f)
    P = d['probeGuess']
    I = d['diff3d'][:200].astype(np.float64)

    A      = np.sum(np.abs(P)**2)                 # probe total intensity
    Nph    = np.mean(np.sum(I, axis=(-2, -1)))    # mean photons per pattern
    sigma2 = 1/A                                  # probe_scaling**2
    s_phi  = 1/Nph                                # physics_scaling_constant
    s_rms  = np.sqrt(I.shape[-1]**2 / np.mean(np.sum(I**2, axis=(-2, -1))))

    c_code = np.sqrt(1/(sigma2*s_phi + 1e-9))     # model.py:1642
    c_bare = np.sqrt(1/(sigma2*s_phi))            # same, guard removed
    c_req  = np.sqrt(Nph)                         # eq. (2.1) at |O| ~ 1

    print(f"{f}\n  A={A:.4e}  Nph={Nph:.4e}  sigma^2*s_phi={sigma2*s_phi:.4e}")
    print(f"  c_code={c_code:.4e}  c_bare={c_bare:.4e}  c_required={c_req:.4e}")
    print(f"  over-scale = {c_bare/c_req:.3f} (= 1/sigma = {np.sqrt(A):.3f})")
    print(f"  guard effect = {c_code/c_bare:.3f}")
    print(f"  equilibrium |O|_rms = {c_req/c_bare:.4f}")
    print(f"  polar 1/s_rms = {1/s_rms:.4e}\n")
```

## Appendix B — symbol table

| symbol | meaning | code |
|---|---|---|
| $I_{\rm meas}$ | measured intensity, detector counts | `batch[0]['images']` |
| $\bar N_{ph}$ | mean photons per pattern, $\langle\sum I\rangle$ | — |
| $s_\phi$ | physics scale, $1/\bar N_{ph}$ | `batch[0]['physics_scaling_constant']`, `helper.py:794` |
| $s_{\rm rms}$ | RMS scale, encoder-input preconditioner | `batch[0]['rms_scaling_constant']`, `helper.py:754` |
| $A$ | stored probe total intensity, $\sum\|P_{\rm raw}\|^2$ | — |
| $\sigma$ | probe scale, $A^{-1/2}$ | `batch[2]`, `helper.py:683` |
| $P_n$ | normalized probe, $\sigma P_{\rm raw}$, $\sum\|P_n\|^2=1$ | `batch[1]` |
| $c$ | output scale | `output_scale`, `model.py:1642` |
| $s_1, s_2$ | learned real/imag coefficients | `model.py:731-732` |
| $O_r, O_i$ | network object, real and imaginary parts | `x_real`, `x_imag` |
