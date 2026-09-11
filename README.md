# ICASSP27-Phrase

Code and manuscript for **Gradient Descent Optimization of Plucked-String
Musical Phrases via Cumulative Energy Loss**.

The repository contains one compact synthesis path, the six losses compared in
the paper, all 150 frozen LHS target coordinate sets per event cardinality, and
an interactive Marimo app that reruns any individual fit on demand. It does not
store rendered target audio, optimisation trajectories, spectrogram caches,
checkpoints, W&B caches, or cluster-specific campaign output.

## Run in your browser

[Open the optimisation notebook in Molab](https://molab.marimo.io/github/ptablasdpaula/ICASSP27-Phrase/blob/main/notebooks/optimize_phrase.py).
Like an "Open in Colab" link, this opens the notebook from GitHub in Marimo's
hosted workspace. Fork it, start the free CPU runtime, and press **Run
optimisation**; no clone, local Python installation, GPU, or Slurm account is
required. The notebook carries its pinned environment as inline metadata.

## Renderer architecture

`PhraseSynth` is the composition of three differentiated modules:

1. `Exciter` constructs the half-raised-cosine source. Its defaults are the
   paper's amplitude 0.8 and duration 10 ms. Onsets can be placed by direct
   sampling (`naive`), fifth-order Lagrange FIR, FLAMO frequency sampling
   (`fourier`), or first-order Thiran all-pass.
2. `Waveguide` implements the pickup-free two-rail string. Linear,
   Lagrange-5/1, and Thiran-3/1 propagation are available through the
   PhilTorch/TorchLPC DF2 recurrence; the same time-domain sections can instead
   retain literal right/left-going rail histories. Frequency-sampled
   propagation uses FLAMO's `frequency` realization and hard resets.
3. `PhraseSynth` sorts the active regimes by detached onset, sums the event
   excitations, and routes the source through the selected waveguide.

The default is the exact recovery configuration used in the paper. PhilTorch
dispatches the DF2 recurrence through TorchLPC on CPU and CUDA:

```python
import torch
from icassp27_phrase import PhraseSynth, load_target

synth = PhraseSynth().to("cuda")  # Fourier exciter + Thiran DF2 waveguide
_, phrase = load_target(4, 1, device="cuda")
audio = synth.render(phrase)
```

Controls and audio are float64 at 4 kHz for two seconds. Candidate pitches and
onsets use independent bounded logits; there is no ordering/spacing transform
or post-update clamp.

## Installation

The base reproducibility environment targets Python 3.12, PyTorch 2.7.1,
FLAMO 0.2.18, and the registered PhilTorch/TorchLPC commits. To run the CPU
notebook outside Molab:

```bash
pixi install
pixi run install-cpu-backends
pixi run notebook
```

The paper campaigns additionally use CUDA 12.6 and the exact
PhilTorch/TorchLPC commits registered in the study. A CUDA toolkit compatible
with the PyTorch wheel is needed to build and qualify that accelerated
recurrence:

```bash
pixi install
pixi run install-backends
pixi run test
pixi run gpu-check
```

The backend installer is deliberately separate: it builds the two native
projects against the PyTorch/CUDA installation on the machine where the paper
campaign will run. Run it on a GPU compute node after making a CUDA 12.6
toolkit (`nvcc`) available; it builds the paper-qualified `sm_70` and `sm_80`
TorchLPC kernels by default. Set `TORCH_CUDA_ARCH_LIST` explicitly to add a
different local architecture.

## Rerun one optimisation

Launch the editable notebook or its read-only app:

```bash
pixi run notebook
# or
pixi run app
```

Choose `number_of_events` from 1, 2, 4, 6, or 8; choose `loss_type` from
`L_1`, `L_2`, `MSS`, `SOT`, `TFW_2`, or `BiCuL`; and choose `target` from 1 to
150. The app renders and plays the target before fitting. During the synchronous
fit it reports the evaluation, patience, best loss, and learning rate. Once it
finishes, it provides target/best audio players and a play/pause/scrubbable
animation containing every evaluated pitch/onset iterate. All artefacts stay in
memory. The notebook deliberately runs on CPU so it can be hosted without a
GPU, while retaining the PhilTorch/TorchLPC DF2 path used by the paper.
Larger fits will naturally be slower than the qualified CUDA campaign.

## Paper and confirmatory study

The self-contained [`paper/`](paper/) directory can be linked directly to
Overleaf. Build it locally with:

```bash
pixi run paper
```

The preregistered confirmatory study has 150 targets at each cardinality and 15
paired BiCuL-versus-TFW2 primary tests. Its original ten-target prefix and the
140 appended targets are frozen in `src/icassp27_phrase/data/targets.json`.
The active cluster campaign remains in its immutable execution repository;
milestone tables and final aggregate results can be copied into `paper/` when
complete without committing any checkpoints or machine-specific paths here.

## Repository scope

Only methods and assets used by the manuscript are retained. Historical
campaign code, experimental estimators, private scratch paths, launch logs,
generated audio, and superseded figures are intentionally absent.
