# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "marimo==0.23.11",
# ]
# ///

"""Interactive reproduction of any single optimisation reported in the paper."""

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full", app_title="ICASSP27 Phrase Optimisation")


@app.cell
def _():
    import importlib
    import os
    import shutil
    import subprocess
    import sys
    from importlib import metadata, util

    import marimo as mo

    return importlib, metadata, mo, os, shutil, subprocess, sys, util


@app.cell(hide_code=True)
def _(importlib, metadata, mo, os, shutil, subprocess, sys, util):
    def _installed_version(distribution):
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return None

    def _environment_is_ready():
        torch_version = _installed_version("torch")
        return all(
            (
                _installed_version("icassp27-phrase") == "0.1.0",
                _installed_version("flamo") == "0.2.18",
                _installed_version("torchlpc") is not None,
                _installed_version("philtorch") is not None,
                torch_version is not None,
                torch_version.partition("+")[0] == "2.7.1",
                util.find_spec("icassp27_phrase") is not None,
            )
        )

    if not _environment_is_ready():
        _uv = shutil.which("uv")
        if _uv is None:
            raise RuntimeError(
                "This notebook needs uv to install its reproducibility environment. "
                "Molab supplies uv automatically; locally, install uv or use `pixi run notebook`."
            )

        _common = [_uv, "pip", "install", "--python", sys.executable]
        _install_environment = os.environ.copy()
        _install_environment["CUDA_VISIBLE_DEVICES"] = ""
        _install_environment["MAX_JOBS"] = "2"
        _commands = (
            (
                "Installing the PyTorch 2.7.1 CPU wheels",
                [
                    *_common,
                    "--index",
                    "https://download.pytorch.org/whl/cpu",
                    "torch==2.7.1+cpu",
                    "torchaudio==2.7.1+cpu",
                ],
            ),
            (
                "Installing the paper and notebook dependencies",
                [
                    *_common,
                    "flamo==0.2.18",
                    "numba>=0.61,<0.67",
                    "numpy>=2,<3",
                    "plotly>=6.3,<7",
                    "scipy>=1.11,<1.17",
                    "ninja>=1.11,<2",
                    "setuptools>=77",
                    "setuptools-git-versioning==2.1.0",
                    "wheel>=0.45,<1",
                ],
            ),
            (
                "Building the pinned TorchLPC CPU recurrence",
                [
                    *_common,
                    "--no-build-isolation",
                    "--no-deps",
                    "https://github.com/DiffAPF/torchlpc/archive/"
                    "1bfde4a457f87b1dd0fc22a6548206be3a26647c.tar.gz",
                ],
            ),
            (
                "Building the pinned PhilTorch frontend",
                [
                    *_common,
                    "--no-build-isolation",
                    "--no-deps",
                    "https://github.com/yoyolicoris/philtorch/archive/"
                    "710946142b6149b486a37f4a3be87ddbf9e2cda3.tar.gz",
                ],
            ),
            (
                "Installing ICASSP27-Phrase from GitHub",
                [
                    *_common,
                    "--no-deps",
                    "git+https://github.com/ptablasdpaula/ICASSP27-Phrase.git@main",
                ],
            ),
        )

        with mo.status.spinner(
            title="Preparing the exact CPU environment",
            subtitle="This one-time setup can take several minutes.",
        ) as _installer:
            for _label, _command in _commands:
                _installer.update(subtitle=_label)
                _completed = subprocess.run(
                    _command,
                    check=False,
                    env=_install_environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                if _completed.returncode != 0:
                    _log_tail = "\n".join(_completed.stdout.splitlines()[-60:])
                    raise RuntimeError(f"{_label} failed:\n{_log_tail}")

        importlib.invalidate_caches()

    if not _environment_is_ready():
        raise RuntimeError("The CPU environment installer completed but did not qualify.")

    environment_ready = True
    _environment_notice = mo.callout(
        "The pinned CPU environment is ready: PyTorch 2.7.1, FLAMO 0.2.18, "
        "and PhilTorch dispatched through the compiled TorchLPC recurrence.",
        kind="success",
    )
    _environment_notice
    return (environment_ready,)


@app.cell
def _(environment_ready, importlib):
    if environment_ready is not True:
        raise RuntimeError("The CPU environment has not been prepared.")

    torch = importlib.import_module("torch")
    _phrase = importlib.import_module("icassp27_phrase")
    _visualization = importlib.import_module("icassp27_phrase.visualization")

    LOSS_LABELS = _phrase.LOSS_LABELS
    PhraseSynth = _phrase.PhraseSynth
    fit = _phrase.fit
    load_target = _phrase.load_target
    trajectory_figure = _visualization.trajectory_figure
    wav_bytes = _visualization.wav_bytes

    _phrase.configure_reproducibility()
    _phrase.require_df2_backend(torch.device("cpu"))
    return (
        LOSS_LABELS,
        PhraseSynth,
        fit,
        load_target,
        mo,
        torch,
        trajectory_figure,
        wav_bytes,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Gradient-descent phrase recovery

    Select one of the paper's event counts, losses, and 150 frozen LHS
    targets. The target is rendered immediately. The fit starts only when
    you press **Run optimisation**; no audio, spectrogram, or trajectory is
    written to disk.
    """)
    return


@app.cell(hide_code=True)
def _(LOSS_LABELS, mo):
    number_of_events = mo.ui.dropdown(
        options={str(value): value for value in (1, 2, 4, 6, 8)},
        value="1",
        label="Number of events",
    )
    loss_type = mo.ui.dropdown(
        options=list(LOSS_LABELS), value="BiCuL", label="Loss"
    )
    target_number = mo.ui.number(
        start=1, stop=150, step=1, value=1, label="Target"
    )
    mo.hstack(
        [number_of_events, loss_type, target_number],
        widths="equal",
        justify="start",
    )
    return loss_type, number_of_events, target_number


@app.cell(hide_code=True)
def _(mo, torch):
    device = torch.device("cpu")
    device_notice = mo.callout(
        "This notebook runs on **CPU**. It uses the same Fourier onset, "
        "Thiran waveguide coefficients, hard-reset event regimes, losses, and "
        "optimizer as the paper. PhilTorch's DF2 filter is dispatched through "
        "TorchLPC's compiled CPU recurrence; higher-cardinality fits may take "
        "a while without a GPU.",
        kind="info",
    )
    device_notice  # noqa: B018 - final expression is the rendered cell output
    return (device,)


@app.cell(hide_code=True)
def _(
    PhraseSynth,
    device,
    load_target,
    mo,
    number_of_events,
    target_number,
    torch,
    wav_bytes,
):
    selected_events = int(number_of_events.value)
    selected_target = int(target_number.value)
    synth = PhraseSynth().to(device)
    target_metadata, target_phrase = load_target(
        selected_events, selected_target, device=device
    )
    with torch.no_grad():
        target_audio = synth.render(target_phrase).detach()
    event_rows = [
        {
            "Event": index + 1,
            "$f_0$ (Hz)": round(float(f0), 4),
            "$t$ (s)": round(float(onset), 6),
        }
        for index, (f0, onset) in enumerate(
            zip(target_metadata.f0_hz, target_metadata.onset_seconds, strict=True)
        )
    ]
    target_panel = mo.vstack(
        [
            mo.md(f"## Target {selected_target}: `{target_metadata.target_id}`"),
            mo.audio(wav_bytes(target_audio, synth.sample_rate)),
            mo.ui.table(event_rows),
        ]
    )
    target_panel  # noqa: B018 - final expression is the rendered cell output
    return synth, target_audio, target_metadata, target_phrase


@app.cell(hide_code=True)
def _(mo):
    run_fit = mo.ui.run_button(
        label="Run optimisation",
        tooltip="Fit this target from the registered equal-cell initialisation",
        kind="primary",
    )
    mo.vstack(
        [
            mo.md(
                "The progress display reports the current evaluation and the "
                "number of evaluations since a meaningful loss improvement."
            ),
            run_fit,
        ]
    )
    return (run_fit,)


@app.cell(hide_code=True)
def _(fit, loss_type, mo, run_fit, synth, target_audio, target_metadata):
    mo.stop(not run_fit.value)
    with mo.status.spinner(
        title="Running optimisation",
        subtitle="Evaluation 0 · patience 0",
        remove_on_exit=False,
    ) as progress_indicator:

        def show_progress(snapshot):
            progress_indicator.update(
                subtitle=(
                    f"evaluation {snapshot.evaluation} · "
                    f"patience {snapshot.patience} / 250 · "
                    f"best loss {snapshot.best_loss:.5g} · "
                    f"learning rate {snapshot.learning_rate:.3g}"
                ),
            )

        fit_result = fit(
            target_audio,
            cardinality=target_metadata.cardinality,
            loss_name=loss_type.value,
            synth=synth,
            progress=show_progress,
        )
        progress_indicator.update(
            title="Optimisation complete",
            subtitle=(
                f"{fit_result.evaluations} evaluations · "
                f"best loss {fit_result.best_loss:.5g}"
            ),
        )
    return (fit_result,)


@app.cell(hide_code=True)
def _(
    fit_result,
    mo,
    synth,
    target_audio,
    target_phrase,
    torch,
    trajectory_figure,
    wav_bytes,
):
    with torch.no_grad():
        best_audio = synth.render(fit_result.best_phrase).detach()
    summary = mo.md(
        f"""
        ## Strict-best result

        - evaluations: **{fit_result.evaluations}**
        - updates: **{fit_result.updates}**
        - stop: **{fit_result.stopped_by}**
        - plateau rollbacks: **{fit_result.plateau_events}**
        - initial loss: **{fit_result.initial_loss:.6g}**
        - best loss: **{fit_result.best_loss:.6g}**
        - wall time: **{fit_result.wall_seconds:.1f} s**
        """
    )
    players = mo.hstack(
        [
            mo.vstack([mo.md("**Target**"), mo.audio(wav_bytes(target_audio, synth.sample_rate))]),
            mo.vstack(
                [
                    mo.md("**Best candidate**"),
                    mo.audio(wav_bytes(best_audio, synth.sample_rate)),
                ]
            ),
        ],
        widths="equal",
    )
    animation = mo.ui.plotly(trajectory_figure(fit_result, target_phrase))
    mo.vstack([summary, players, mo.md("## Every evaluated iterate"), animation])
    return


if __name__ == "__main__":
    app.run()
