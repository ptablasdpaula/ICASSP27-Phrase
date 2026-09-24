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
                _installed_version("icassp27-phrase") == "0.2.2",
                _installed_version("flamo") == "0.2.18",
                _installed_version("torchlpc") is not None,
                _installed_version("philtorch") is not None,
                _installed_version("cels-audio") == "0.1.1",
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
                    "matplotlib>=3.11,<3.12",
                    "python-dotenv>=1,<2",
                    "cels-audio==0.1.1",
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
                    "git+https://github.com/ptablasdpaula/ICASSP27-Phrase.git@paper-reproduction-v0.2.2",
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
        # A reused cloud kernel may still hold the previous package in memory.
        for _module_name in list(sys.modules):
            if _module_name.split(".")[0] in {"icassp27_phrase", "cels"}:
                del sys.modules[_module_name]

    if not _environment_is_ready():
        raise RuntimeError("The CPU environment installer completed but did not qualify.")

    environment_ready = True
    _environment_notice = mo.callout(
        "The pinned CPU environment is ready: PyTorch 2.7.1, FLAMO 0.2.18, "
        "and PhilTorch dispatched through the compiled TorchLPC recurrence.",
        kind="success",
    )
    mo.output.replace(_environment_notice)
    return (environment_ready,)


@app.cell
def _(environment_ready, importlib, mo):
    if environment_ready is not True:
        raise RuntimeError("The CPU environment has not been prepared.")

    torch = importlib.import_module("torch")
    _phrase = importlib.import_module("icassp27_phrase")
    _visualization = importlib.import_module("icassp27_phrase.visualization")

    LOSS_LABELS = _phrase.LOSS_LABELS
    PhraseSynth = _phrase.PhraseSynth
    fit = _phrase.fit
    load_target = _phrase.load_target
    spectrogram_figure = _visualization.spectrogram_figure
    wav_bytes = _visualization.wav_bytes

    _phrase.configure_reproducibility()
    _phrase.require_df2_backend(torch.device("cpu"))
    return (
        LOSS_LABELS,
        PhraseSynth,
        fit,
        load_target,
        mo,
        spectrogram_figure,
        torch,
        wav_bytes,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Gradient-descent phrase recovery

    Select one of the paper's event counts, losses, and 150 frozen LHS
    targets. The target is rendered immediately. The fit starts only when
    you press **Run optimisation**. The current candidate spectrogram is
    refreshed initially and every 10 Adam updates. The iteration counter updates
    every step; nothing is written
    to disk.
    """)
    return


@app.cell(hide_code=True)
def _(LOSS_LABELS, mo):
    number_of_events = mo.ui.dropdown(
        options={str(value): value for value in (1, 2, 4, 6, 8)},
        value="1",
        label="Number of events",
    )
    loss_type = mo.ui.dropdown(options=list(LOSS_LABELS), value="CeL", label="Loss")
    target_number = mo.ui.number(start=1, stop=150, step=1, value=1, label="Target")
    mo.hstack(
        [number_of_events, loss_type, target_number],
        widths="equal",
        justify="start",
    )
    return loss_type, number_of_events, target_number


@app.cell(hide_code=True)
def _(
    PhraseSynth,
    load_target,
    mo,
    number_of_events,
    spectrogram_figure,
    target_number,
    torch,
    wav_bytes,
):
    selected_events = int(number_of_events.value)
    selected_target = int(target_number.value)
    synth = PhraseSynth().to("cpu")
    target_metadata, _target_phrase = load_target(selected_events, selected_target, device="cpu")
    with torch.no_grad():
        target_audio = synth.render(_target_phrase).detach()
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
            mo.ui.plotly(
                spectrogram_figure(
                    target_audio,
                    synth.sample_rate,
                    title="Target magnitude spectrogram",
                )
            ),
            mo.ui.table(event_rows),
        ]
    )
    target_panel  # noqa: B018 - final expression is the rendered cell output
    return synth, target_audio, target_metadata


@app.cell(hide_code=True)
def _(
    fit,
    loss_type,
    mo,
    spectrogram_figure,
    synth,
    target_audio,
    target_metadata,
    torch,
    wav_bytes,
):
    def _run_optimisation(pressed):
        if not pressed:
            return
        _loss_name = loss_type.value
        _identity = (
            f"Target {target_metadata.target_id} · {target_metadata.cardinality} events · "
            f"{_loss_name}"
        )
        # UI callbacks receive a fresh output context, so initialise all slots.
        mo.output.replace(optimisation_controls)
        mo.output.append(mo.md(f"## Starting optimisation\n{_identity}"))
        mo.output.append(mo.md("Preparing initial candidate…"))
        mo.output.append(mo.md(""))

        def _show_progress(snapshot, current_audio):
            _status = mo.md(
                f"""
                ## Optimising — iteration {snapshot.update:,} / 20,000

                {_identity}

                **{snapshot.update:,} completed Adam updates ·
                evaluation {snapshot.evaluation:,}**

                - current loss: **{snapshot.raw_loss:.6g}**
                - lowest loss: **{snapshot.best_loss:.6g}**
                - patience: **{snapshot.patience} / 1000**
                - learning rate: **{snapshot.learning_rate:.4g}**
                - learning-rate reductions: **{snapshot.plateau_events}**
                """
            )
            mo.output.replace_at_index(_status, 1)
            if snapshot.update % 10 == 0:
                _figure = spectrogram_figure(
                    current_audio,
                    synth.sample_rate,
                    title=f"Current candidate — after {snapshot.update:,} Adam updates",
                )
                mo.output.replace_at_index(_figure, 2)

        try:
            _result = fit(
                target_audio,
                cardinality=target_metadata.cardinality,
                loss_name=_loss_name,
                synth=synth,
                progress=_show_progress,
            )
            with torch.no_grad():
                _best_audio = synth.render(_result.best_phrase).detach()
            mo.output.replace_at_index(
                mo.md(
                    f"## Lowest-loss candidate ready\n\n{_identity}\n\n"
                    f"**{_result.updates:,} Adam updates · {_result.evaluations:,} evaluations · "
                    f"best loss {_result.best_loss:.6g} · stopped by {_result.stopped_by}.**"
                ),
                1,
            )
            mo.output.replace_at_index(
                spectrogram_figure(
                    _best_audio, synth.sample_rate, title="Lowest-loss candidate spectrogram"
                ),
                2,
            )
            _event_rows = [
                {"Event": i + 1, "$f_0$ (Hz)": round(float(f0), 4), "$t$ (s)": round(float(t), 6)}
                for i, (f0, t) in enumerate(
                    zip(_result.best_phrase.f0_hz, _result.best_phrase.onset_seconds, strict=True)
                )
            ]
            _players = mo.hstack(
                [
                    mo.vstack(
                        [mo.md("**Target**"), mo.audio(wav_bytes(target_audio, synth.sample_rate))]
                    ),
                    mo.vstack(
                        [
                            mo.md("**Best candidate**"),
                            mo.audio(wav_bytes(_best_audio, synth.sample_rate)),
                        ]
                    ),
                ],
                widths="equal",
            )
            mo.output.replace_at_index(
                mo.vstack(
                    [
                        mo.md(
                            f"Initial loss: **{_result.initial_loss:.6g}** · "
                            f"learning-rate reductions: **{_result.plateau_events}** · "
                            f"wall time: **{_result.wall_seconds:.1f} s**"
                        ),
                        _players,
                        mo.md("## Recovered events"),
                        mo.ui.table(_event_rows),
                    ]
                ),
                3,
            )
        except Exception:
            mo.output.replace_at_index(
                mo.callout(
                    "Optimisation failed; see the error below. You can retry.", kind="danger"
                ),
                1,
            )
            raise

    # Execute inside the click handler, including in notebooks using lazy execution.
    # Progress and results stay in this cell; no downstream cell needs a manual run.
    run_fit = mo.ui.run_button(
        label="Run optimisation",
        tooltip="Immediately fit this target from the registered equal-cell initialisation",
        kind="success",
        on_change=_run_optimisation,
    )
    # Keep this binding public so delayed callbacks can resolve it in Marimo.
    optimisation_controls = mo.vstack(
        [
            mo.md(
                "The counter updates every iteration; the spectrogram refreshes every 10 "
                "Adam updates. Each fit starts at learning rate 0.05, halves it after "
                "200 plateau evaluations, and stops after 1000 evaluations without "
                "0.01% relative improvement or 20,000 updates. Adam state is retained; "
                "the lowest-loss candidate is reported."
            ),
            run_fit,
        ]
    )
    mo.output.replace(optimisation_controls)
    mo.output.append(mo.md("Press **Run optimisation** to start."))
    mo.output.append(mo.md(""))
    mo.output.append(mo.md(""))
    return optimisation_controls, run_fit


if __name__ == "__main__":
    app.run()
