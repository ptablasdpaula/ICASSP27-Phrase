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
