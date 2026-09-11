"""In-memory visualisation helpers for the interactive reproduction app."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from scipy.io import wavfile
from torch import Tensor

from .config import EventPhrase
from .optimization import FitResult


def wav_bytes(audio: Tensor, sample_rate: int = 4_000) -> BytesIO:
    """Encode a finite mono tensor as browser-compatible 16-bit WAV bytes."""
    value = audio.detach().cpu().numpy().astype(np.float64, copy=False)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("audio must be one finite sample vector")
    peak = float(np.max(np.abs(value), initial=0.0))
    scale = 1.0 if peak <= 1.0 else peak
    pcm = np.round(np.clip(value / scale, -1.0, 1.0) * 32767.0).astype(np.int16)
    stream = BytesIO()
    wavfile.write(stream, sample_rate, pcm)
    stream.seek(0)
    return stream


def trajectory_figure(result: FitResult, target: EventPhrase):
    """Create a Plotly animation with play, pause, and iteration scrubbing."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:  # pragma: no cover - optional notebook dependency
        raise RuntimeError("install the `notebook` extra to animate a fit") from error

    snapshots = result.trajectory
    if not snapshots:
        raise ValueError("cannot animate an empty trajectory")
    evaluations = [item.evaluation for item in snapshots]
    losses = [item.raw_loss for item in snapshots]
    first = snapshots[0]
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Event recovery", "Optimising loss"),
        horizontal_spacing=0.14,
    )
    figure.add_trace(
        go.Scatter(
            x=target.onset_seconds.detach().cpu().numpy(),
            y=target.f0_hz.detach().cpu().numpy(),
            mode="markers",
            marker={"symbol": "x", "size": 13, "color": "#d62728", "line": {"width": 2}},
            name="Target",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=first.onset_seconds,
            y=first.f0_hz,
            mode="markers",
            marker={"size": 11, "color": "#6f2dbd"},
            name="Candidate",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=evaluations,
            y=losses,
            mode="lines",
            line={"color": "#555555", "width": 1.5},
            name="Loss",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=[first.evaluation],
            y=[first.raw_loss],
            mode="markers",
            marker={"size": 10, "color": "#6f2dbd"},
            name="Current",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    frames = []
    for item in snapshots:
        frames.append(
            go.Frame(
                name=str(item.evaluation),
                data=[
                    go.Scatter(x=item.onset_seconds, y=item.f0_hz),
                    go.Scatter(x=[item.evaluation], y=[item.raw_loss]),
                ],
                traces=[1, 3],
                layout=go.Layout(
                    title={
                        "text": (
                            f"Evaluation {item.evaluation} · patience {item.patience} · "
                            f"best loss {item.best_loss:.4g}"
                        )
                    }
                ),
            )
        )
    figure.frames = frames
    slider_steps = [
        {
            "args": [
                [str(item.evaluation)],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": (
                str(item.evaluation)
                if item.evaluation in {1, len(snapshots)} or item.evaluation % 100 == 0
                else ""
            ),
            "method": "animate",
        }
        for item in snapshots
    ]
    figure.update_layout(
        title=(
            f"Evaluation 1 · patience {first.patience} · "
            f"best loss {first.best_loss:.4g}"
        ),
        height=470,
        margin={"l": 55, "r": 20, "t": 80, "b": 70},
        template="plotly_white",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": -0.15,
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 70, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Evaluation "},
                "pad": {"t": 45},
                "steps": slider_steps,
            }
        ],
    )
    figure.update_xaxes(title_text="Onset (s)", range=[0.2, 1.8], row=1, col=1)
    figure.update_yaxes(
        title_text="$f_0$ (Hz)",
        type="log",
        range=[np.log10(80), np.log10(320)],
        row=1,
        col=1,
    )
    figure.update_xaxes(title_text="Evaluation", row=1, col=2)
    figure.update_yaxes(title_text="Raw loss", type="log", row=1, col=2)
    return figure


__all__ = ["trajectory_figure", "wav_bytes"]
