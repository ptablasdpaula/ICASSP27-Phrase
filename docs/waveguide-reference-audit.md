# Waveguide section: reference and implementation audit

Reviewed 16 September 2026 against the Overleaf revision `c110f8f`.
The newly rewritten section is numbered III in `paper/main.tex`.
The audit covers all seven references cited in that section, its red notes,
and the associated implementation. Other sections' claims were not revised.

| Reference | Checked material | Finding and manuscript action |
| --- | --- | --- |
| Smith (2010), *Physical Audio Signal Processing* | [Thiran allpass interpolators](https://www.dsprelated.com/freebooks/pasp/Thiran_Allpass_Interpolators.html), including coefficient formula and group-delay discussion | Supports fractional-delay tuning and maximally flat group delay at DC. Unity magnitude follows from the all-pass structure, rather than the maximally flat approximation. Corrected this distinction. |
| Tablas de Paula et al. (2026), *Four Decades of Digital Waveguides* | [Full manuscript](https://arxiv.org/pdf/2604.12878), historical tuning discussion, waveguide theory, and plucked-string applications | Supports DWG use for plucked strings and the need to account for attenuation-filter phase when tuning. The simple pitch formula is an ideal-delay relation, not an exact formula for the implemented interpolated, filtered loop. Qualified it and documented phase compensation. |
| Tablas de Paula et al. (2026), *Sound Matching with a Differentiable Karplus–Strong Algorithm* | [Full paper](https://www.dafx.de/paper-archive/2026/papers/DAFx26_paper_38.pdf), loss-filter discussion, Fig. 3 and Eq. (16) | Supports coupled pitch, damping and decay time. Fig. 3 concerns a related KSA and does not validate physical realism for the present DWG. Replaced “physically realistic” with the supported pitch-dependent decay statement. |
| Dal Santo et al. (2025), FLAMO | [Full paper](https://arxiv.org/html/2409.08723v2), Sections II and III-A; [Delay documentation](https://gdalsanto.github.io/flamo/processor/dsp.html) | Supports differentiable complex-exponential delays and the time-aliasing issue. Padding and Hermitian symmetry are separate operations. Changed “avoid” to “reduce” wrap-around; documented real DC/Nyquist projection separately. The padding ratio is our implementation choice, not a result of the reference. |
| Laakso et al. (1996), *Splitting the Unit Delay* | Local full PDF, printed p. 49, “Maximally Flat Group Delay Design of Allpass Filters,” Eq. (86), and the following design guide | Supports the Thiran all-pass construction, unity magnitude, and group-delay flatness at zero frequency. Used the existing citation at the red placeholder. Its discussion only states stability for sufficiently large positive delay; the explicit standard condition is corroborated by the primary paper linked below and enforced by our code. |
| Yu and Fazekas (2025), PhilTorch | [Full paper](https://arxiv.org/html/2511.14390v1), Section II, Eqs. (1)–(3), and implementation discussion | Supports direct-form evaluation and accelerated differentiation. DF2 is a choice used here, not a universal requirement of the library. Removed “as required.” |
| Yu et al. (2024), TorchLPC | [Full paper](https://dafx.de/paper-archive/2024/papers/DAFx24_paper_75.pdf), time-varying all-pole filtering and gradient derivation | Supports differentiable recursive filtering. The specific PhilTorch/TorchLPC combination is confirmed by `src/waveguide.py`, not established by the older paper alone. |

The local Laakso copy is
`/data/home/acw794/ICASSP2027-GtrArticulations/refs/Splitting_the_Unit_Delay.pdf`.
For the explicit Thiran stability condition, see also the Section 2 of
[Hacıhabiboğlu, Günel and Kondoz, “Interpolated Allpass Fractional-Delay Filters using Root Displacement,” ICASSP 2006](https://users.metu.edu.tr/hhuseyin/Conferences/hhogunkon_icassp06.pdf),
which states the admissible delay as D > N − 1.

## Implementation checks

- `src/config.py`: sample rate 4000 Hz, signal length 8000 samples,
  Fourier FFT length 262144. Thus the FFT is **32.768 times the full signal
  length**, described as approximately 32.8 times in the paper.
- `src/exciter.py`: FLAMO supplies fractional-delay phase responses; event
  spectra are summed, projected, inverse-transformed once, and cropped to the
  first 8000 samples. This matches the revised synthesis description.
- `src/_fractional.py`: `hermitian_rfft_projection` makes DC and Nyquist
  real. Increasing FFT length alone does not enforce this property. Finite
  padding also does not make the ideal fractional-delay kernel finite.
- `src/waveguide.py`: each travelling rail contains the long and short
  segments; the round trip traverses both twice. The loss filter contributes
  phase. Twelve iterations compensate interpolation and loop-filter phase.
- Evaluating the existing phase solver at 80 and 320 Hz gives bridge-side
  delays of **5.7212866 and 1.4174841 samples**, respectively; nut-side delays
  are **19.1538725 and 4.7454903 samples**. The short high-pitch path cannot
  accommodate an order-3 Thiran section requiring delay greater than 2.
  This explains the order-1 choice; unity magnitude itself does not select
  the order. The integer part and all-pass section are distinct.
- The half-raised-cosine formula has support `0 <= t < T_e`; therefore `A_e`
  is an amplitude scale / limiting peak, not an attained sampled peak.
- The recurrence uses `philtorch.lpv.lfilter` with `backend="torchlpc"`.
  Reset boundaries are `floor(onset.detach() * sample_rate)`. The onset
  excitation remains differentiable, but the reset decision is detached.
- Smoother multi-event landscapes under reset are an empirical diagnostic
  finding, not a theorem or a consequence established by the cited papers.
  Interference with residual vibration remains a hypothesis. Removed the
  general claim that reset is necessary and retained the diagnostic scope.

## Figure placement

The waveguide float requests bottom placement. The landscape float is moved
in the source to the experiments section so it cannot block the waveguide
in LaTeX's figure queue. Figure numbering follows this new order; references
use labels. No landscape caption or experimental prose is changed.

## Follow-up: Hermitian symmetry and FFT padding

Smith's *Mathematics of the Discrete Fourier Transform*, second edition
(2007), Chapter 7, supplies a single reference for the two distinct claims:

- [Symmetry](https://www.dsprelated.com/freebooks/mdft/Symmetry.html): a real
  sequence has a conjugate-symmetric spectrum. At the self-conjugate DC and
  Nyquist bins of an even-length DFT, this requires real values.
- [Convolution theorem](https://www.dsprelated.com/freebooks/mdft/Convolution_Theorem.html)
  and [zero-padding applications](https://www.dsprelated.com/freebooks/mdft/Zero_Padding_Applications.html):
  spectral multiplication implements circular convolution, and zero padding
  guards against wrap-around. The familiar finite-convolution bound depends
  on both operands' support; it does not prescribe a universal padding ratio.
- [Windowed sinc interpolation](https://www.dsprelated.com/freebooks/pasp/Windowed_Sinc_Interpolation.html)
  in Smith's *Physical Audio Signal Processing* describes ideal bandlimited
  interpolation. The ideal fractional-delay sinc kernel has infinite support,
  so finite FFT padding cannot make the operation exactly nonperiodic.

The new `smith2007dft` citation is placed at both Fourier claims. Laakso is
now cited directly at the unity-magnitude/group-delay statement and again at
the stability/order discussion. “Unity magnitude” is the all-pass property;
the section's delay need not be one sample.

### Is 32.8 times excessive?

For this **onset excitation**, it is conservative and probably larger than
necessary. This is a numerical assessment, not a ratio recommended by Smith.
The reproducible diagnostic `scripts/audit_onset_padding.py` compares the
actual sampled 10-ms HRC shape against nonperiodic sinc interpolation,
including its analytic derivative with respect to onset. It evaluates 41
onsets spanning 0.2–1.8 seconds and fractional-sample offsets in float64,
retaining the same 8000 output samples as the synthesiser. The analytic
reference avoids assuming that some larger FFT is already converged.
Full results are in `docs/onset-padding-audit.json`.

| FFT length | Ratio to full signal | Worst relative waveform L2 error | Worst relative onset-derivative L2 error |
| ---: | ---: | ---: | ---: |
| 16,384 | 2.048× | 0.0195% | 0.200% |
| 32,768 | 4.096× | 0.00457% | 0.0468% |
| 65,536 | 8.192× | 0.00113% | 0.0115% |
| 262,144 | 32.768× | 0.0000701% | 0.000717% |

Each column reports its own maximum over the sampled onsets. Derivative
errors are relative vector norms, not per-sample relative errors or loss-gradient errors. Integer onsets can reproduce the excitation exactly while
still having an inaccurate derivative, which is why both quantities matter.

A length of 32,768–65,536 is a plausible efficiency choice for a future
configuration. The current experiments retain their original 262,144-point
setting: this excitation-only diagnostic does not establish equivalence of
waveguide outputs, loss gradients or converged phrase fits. No synthesis
settings or experimental results were changed.

As an implementation cross-check, at a 4000.25-sample onset with a
16,384-point FFT, the diagnostic Fourier computation agrees with the actual
FLAMO-backed `Exciter`: maximum absolute waveform difference was
3.67e-13, and the onset derivative agreed with PyTorch's automatic differentiation JVP within 2.46e-9 (derivatives measured per second).
