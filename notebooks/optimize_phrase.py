# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "icassp27-phrase[accelerated,notebook] @ git+https://github.com/ptablasdpaula/ICASSP27-Phrase.git@main",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.extra-build-dependencies]
# torchlpc = [{ requirement = "torch", match-runtime = true }]
# philtorch = [{ requirement = "torch", match-runtime = true }]
# ///

"""Interactive reproduction of any single optimisation reported in the paper."""

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full", app_title="ICASSP27 Phrase Optimisation")


@app.cell
def _():
    import marimo as mo
    import torch

    from icassp27_phrase import (
        LOSS_LABELS,
        PhraseSynth,
        configure_reproducibility,
        fit,
        load_target,
    )
    from icassp27_phrase.visualization import trajectory_figure, wav_bytes

    configure_reproducibility()
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
        value=1,
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
