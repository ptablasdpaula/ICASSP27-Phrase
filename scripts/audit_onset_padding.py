"""Compare onset FFT padding with exact sinc interpolation of the HRC samples.

Run with the repository Python environment. Writes a diagnostic JSON to stdout;
no synthesis or campaign settings are changed. Errors concern excitation and
its onset derivative, not final waveguide output or optimisation outcomes.
"""

import json

import numpy as np


def main():
    samples = 8000
    prototype = 0.4 * (1 - np.cos(np.pi * np.arange(40) / 40))
    lengths = [8192, 16384, 32768, 65536, 131072, 262144]
    spectra = {length: np.fft.rfft(prototype, n=length) for length in lengths}
    frequencies = {length: 2 * np.pi * np.fft.rfftfreq(length) for length in lengths}
    records = {length: [] for length in lengths}
    # Cover the permitted onset interval, including its boundaries and
    # fractional offsets. Integer offsets alone would hide interpolation error.
    delays = sorted(
        {
            min(7200.0, base + fraction)
            for base in np.linspace(800, 7200, 9)
            for fraction in [0.0, 0.125, 0.25, 0.5, 0.875]
        }
    )
    for delay in delays:
        offset = np.arange(samples)[:, None] - delay - np.arange(40)[None, :]
        reference = np.sinc(offset) @ prototype
        # d sinc(x) / dx; its removable singularity at zero has value zero.
        derivative = np.zeros_like(offset)
        mask = offset != 0
        x = offset[mask]
        derivative[mask] = (np.pi * x * np.cos(np.pi * x) - np.sin(np.pi * x)) / (np.pi * x * x)
        reference_grad = -derivative @ prototype
        for length in lengths:
            omega = frequencies[length]
            shifted = spectra[length] * np.exp(-1j * omega * delay)
            gradient = shifted * (-1j * omega)
            # irfft imposes the same real DC/Nyquist projection as the code.
            audio = np.fft.irfft(shifted, n=length)[:samples]
            grad = np.fft.irfft(gradient, n=length)[:samples]
            records[length].append(
                {
                    "relative_l2": float(
                        np.linalg.norm(audio - reference) / np.linalg.norm(reference)
                    ),
                    "max_abs": float(np.max(np.abs(audio - reference))),
                    "derivative_relative_l2": float(
                        np.linalg.norm(grad - reference_grad) / np.linalg.norm(reference_grad)
                    ),
                }
            )
    print(
        json.dumps(
            {
                "reference": ("Exact nonperiodic sinc interpolation of the 40 HRC support samples"),
                "onset_delays_samples": delays,
                "sample_rate": 4000,
                "retained_samples": samples,
                "dtype": "float64",
                "results": [
                    {
                        "fft_length": length,
                        "padding_ratio": length / samples,
                        **{
                            f"worst_{key}": max(row[key] for row in records[length])
                            for key in records[length][0]
                        },
                    }
                    for length in lengths
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
