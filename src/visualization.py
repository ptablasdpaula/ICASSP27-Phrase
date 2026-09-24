"""In-memory visualisation helpers for the interactive reproduction app."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from scipy.io import wavfile
from torch import Tensor


def wav_bytes(audio: Tensor, sample_rate: int = 16_000) -> BytesIO:
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


def spectrogram_figure(
    audio: Tensor,
    sample_rate: int = 16_000,
    *,
    title: str = "Magnitude spectrogram",
):
    """Create a lightweight, peak-normalised STFT magnitude spectrogram."""
    try:
        import plotly.graph_objects as go
    except ImportError as error:  # pragma: no cover - optional notebook dependency
        raise RuntimeError("install the `notebook` extra to plot a spectrogram") from error

    value = audio.detach().cpu().numpy().astype(np.float64, copy=False)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("audio must be one finite sample vector")
    if sample_rate < 1:
        raise ValueError("sample_rate must be positive")

    window_length = 256
    hop_length = 64
    padded = np.pad(value, (window_length // 2, window_length // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, window_length)[::hop_length]
    window = np.hanning(window_length)
    magnitude = np.abs(np.fft.rfft(frames * window[None, :], axis=1))
    peak = float(np.max(magnitude, initial=0.0))
    floor = np.finfo(np.float64).tiny
    relative = magnitude / max(peak, floor)
    magnitude_db = 20.0 * np.log10(np.maximum(relative, 1e-4))
    times = np.arange(frames.shape[0], dtype=np.float64) * hop_length / sample_rate
    frequencies = np.fft.rfftfreq(window_length, d=1.0 / sample_rate)

    figure = go.Figure(
        data=go.Heatmap(
            x=times,
            y=frequencies,
            z=magnitude_db.T,
            colorscale="Magma",
            zmin=-80.0,
            zmax=0.0,
            zsmooth=False,
            colorbar={"title": {"text": "dB"}},
            hovertemplate="%{x:.3f} s<br>%{y:.1f} Hz<br>%{z:.1f} dB<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        height=340,
        margin={"l": 65, "r": 25, "t": 55, "b": 55},
        template="plotly_white",
    )
    figure.update_xaxes(title_text="Time (s)", range=[0.0, len(value) / sample_rate])
    figure.update_yaxes(title_text="Frequency (Hz)", range=[0.0, sample_rate / 2.0])
    return figure


__all__ = ["spectrogram_figure", "wav_bytes"]
