"""Full frozen-registry orthogonal/clockwise recovery study, CPU float64."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from _recovery_study_fit import decode_coordinates, encode_coordinates, fit
from icassp27_phrase.config import CARDINALITIES, PAPER_OPTIMIZER, EventPhrase
from icassp27_phrase.losses import CumulativeEnergyDistance
from icassp27_phrase.runtime import require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from screen_eight_directions import AxisLoss
from test_amplitude_recovery import metrics
from test_clockwise_eight import weights

ROOT = Path("results/direction-recovery-150")
VARIANTS = ("orthogonal", "orthogonal_lw", "clockwise", "clockwise_lw")


def signature():
    digest = hashlib.sha256()
    paths = [
        Path(__file__),
        Path("scripts/_recovery_study_fit.py"),
        Path("scripts/screen_eight_directions.py"),
        Path("scripts/test_clockwise_eight.py"),
        Path("scripts/test_amplitude_recovery.py"),
    ]
    paths += list(sorted(Path("src").glob("*.py"))) + [Path("src/data/targets.json")]
    for path in paths:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Objective:
    def __init__(self, target, family, weighted):
        self.diagonal = CumulativeEnergyDistance(target)
        self.axis = AxisLoss(self.diagonal)
        self.family, self.weighted = family, weighted

    def __call__(self, audio):
        if self.family == "orthogonal":
            return self.axis.terms(audio[None])[0, 4 * self.weighted : 4 * self.weighted + 4].mean()
        return self.diagonal.directional_distances(audio)[int(self.weighted)].mean()

    def all_terms(self, audio):
        d = self.diagonal.directional_distances(audio)[int(self.weighted)]
        a = self.axis.terms(audio[None])[0, 4 * self.weighted : 4 * self.weighted + 4]
        return torch.cat((d, a))


def serialise_fit(result):
    out = asdict(result)
    out["best_phrase"] = {k: v.tolist() for k, v in out["best_phrase"].items()}
    return out


def log_spectral_distance(candidate, target):
    """Registered metric: centered STFT, common target-relative amplitude floor."""
    window = torch.hann_window(256, periodic=True, dtype=target.dtype)

    def magnitude(audio):
        return torch.stft(
            audio[None],
            n_fft=256,
            hop_length=64,
            win_length=256,
            window=window,
            center=True,
            return_complex=True,
        ).abs()

    c, t = magnitude(candidate), magnitude(target)
    floor = (t.amax(dim=(-2, -1), keepdim=True) * 1e-5).clamp_min(torch.finfo(target.dtype).tiny)
    difference = 20 * torch.log10(torch.maximum(c, floor)) - 20 * torch.log10(
        torch.maximum(t, floor)
    )
    return float(difference.square().mean(dim=(-2, -1)).sqrt()[0])


def describe(phrase, synth, target_audio, metadata):
    result = metrics(phrase.f0_hz.numpy(), phrase.onset_seconds.numpy(), metadata)
    assignment = result["assignment"]
    dp = np.log2(phrase.f0_hz.numpy() / np.array(metadata.f0_hz)[assignment]) / 2
    dt = (phrase.onset_seconds.numpy() - np.array(metadata.onset_seconds)[assignment]) / 1.6
    result["joint_event_error"] = float(np.sqrt((dp * dp + dt * dt).mean()))
    with torch.no_grad():
        audio = synth.render(phrase)
        result["log_spectral_distance_db"] = log_spectral_distance(audio, target_audio)
        result["relative_waveform_l2"] = float((audio - target_audio).norm() / target_audio.norm())
    return result


def escape(initial, baseline_loss, objective, synth):
    raw = encode_coordinates(initial.f0_hz, initial.onset_seconds).requires_grad_()
    best, best_raw, best_update = baseline_loss, raw.detach().clone(), 0
    opt = torch.optim.Adam([raw], lr=0.05)
    trace = []
    for step in range(2401):
        if step == 1600:
            opt = torch.optim.Adam([raw], lr=0.05)
        if step >= 1600:
            opt.param_groups[0]["lr"] = 0.05 * 0.3 ** min((step - 1600) // 200, 3)
        f, t = decode_coordinates(raw)
        audio = synth.render_batch(f[None], t[None])[0]
        if step < 1600:
            terms = objective.all_terms(audio)
            canonical = terms[:4].mean()
            loss = (terms * weights(step, "clockwise")).sum()
        else:
            canonical = objective(audio)
            loss = canonical
        value = float(canonical.detach())
        if value < best:
            best, best_raw, best_update = value, raw.detach().clone(), step
        trace.append(
            dict(
                update=step,
                raw_loss=value,
                objective_loss=float(loss.detach()),
                best_loss=best,
                learning_rate=opt.param_groups[0]["lr"],
                f0_hz=f.detach().tolist(),
                onset_seconds=t.detach().tolist(),
            )
        )
        if step == 2400:
            break
        opt.zero_grad()
        (loss / baseline_loss).backward()
        assert bool(torch.isfinite(raw.grad).all())
        opt.step()
    f, t = decode_coordinates(best_raw)
    return EventPhrase(f.detach(), t.detach()), best, best_update, trace


def run(index, root=ROOT):
    variant = VARIANTS[index // 750]
    events = CARDINALITIES[(index % 750) // 150]
    number = index % 150 + 1
    metadata, target = load_target(events, number)
    path = root / "raw" / variant / f"{metadata.target_id}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    sig = signature()
    if path.exists():
        with gzip.open(path, "rt") as f:
            assert json.load(f)["signature"] == sig
        print("EXISTS", index, path, flush=True)
        return
    start = time.perf_counter()
    synth = PhraseSynth()
    with torch.no_grad():
        audio = synth.render(target)
    family = "orthogonal" if variant.startswith("orthogonal") else "diagonal"
    objective = Objective(audio, family, variant.endswith("_lw"))
    result = fit(audio, events, variant, synth=synth, objective=objective)
    out = dict(
        index=index,
        variant=variant,
        target=asdict(metadata),
        signature=sig,
        renderer=synth.provenance(),
        optimizer=asdict(PAPER_OPTIMIZER),
        torch_version=torch.__version__,
        baseline=serialise_fit(result),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        target_registry_sha256=hashlib.sha256(
            Path("src/data/targets.json").read_bytes()
        ).hexdigest(),
    )
    final, loss = result.best_phrase, result.best_loss
    out["baseline_metrics"] = describe(final, synth, audio, metadata)
    out["rotation_applied"] = family == "diagonal" and result.stopped_by == "patience"
    out["updates"] = result.updates
    if out["rotation_applied"]:
        final, loss, best_update, trace = escape(final, loss, objective, synth)
        out.update(rotation_trajectory=trace, best_rotation_update=best_update)
        out["updates"] += 2400
    out["best_loss"] = loss
    out["best_phrase"] = dict(
        f0_hz=final.f0_hz.tolist(), onset_seconds=final.onset_seconds.tolist()
    )
    out["metrics"] = describe(final, synth, audio, metadata)
    out["wall_seconds"] = time.perf_counter() - start
    with torch.no_grad():
        assert abs(float(objective(synth.render(final))) - loss) < 1e-10
    temporary = path.with_suffix(".tmp.gz")
    with gzip.open(temporary, "wt") as f:
        json.dump(out, f, separators=(",", ":"))
    temporary.replace(path)
    print(
        "COMPLETE",
        index,
        variant,
        metadata.target_id,
        out["updates"],
        round(out["wall_seconds"], 2),
        out["metrics"],
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    assert 0 <= args.index < 3000
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    require_df2_backend("cpu")
    qualification = json.loads(Path("docs/direction-recovery-150/qualification.json").read_text())
    assert qualification["passed"] and qualification["signature"] == signature()
    run(args.index, args.root)


if __name__ == "__main__":
    main()
