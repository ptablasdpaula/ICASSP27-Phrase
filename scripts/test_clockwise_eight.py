"""Clockwise eight-direction descent on the frozen plateau cases, with matched controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from _amplitude_fit import decode_coordinates, encode_coordinates
from _takeover_validation import Engine, initial_state
from icassp27_phrase.config import EventPhrase
from icassp27_phrase.losses import CumulativeEnergyDistance
from icassp27_phrase.runtime import require_df2_backend
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from screen_eight_directions import DIRECTIONS, AxisLoss

ROOT = Path("results/clockwise-eight/raw")
ORDER = (0, 6, 1, 5, 3, 7, 2, 4)  # ↗ → ↘ ↓ ↙ ← ↖ ↑
EXPLORATION = 1600
POLISH = 800
PERIOD = 800
RAMP = 100
VARIANTS = ("clockwise", "uniform_eight", "diagonal_control")


def weights(step, variant):
    baseline = torch.tensor([0.25] * 4 + [0.0] * 4, dtype=torch.float64)
    if variant == "diagonal_control":
        return baseline
    if variant == "uniform_eight":
        return torch.full((8,), 0.125, dtype=torch.float64)
    position = (step % PERIOD) / (PERIOD / 8)
    sector = int(position)
    fraction = position - sector
    rotated = torch.zeros(8, dtype=torch.float64)
    rotated[ORDER[sector]] = 1 - fraction
    rotated[ORDER[(sector + 1) % 8]] = fraction
    strength = min(step / RAMP, 1.0)
    return (1 - strength) * baseline + strength * rotated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=int, required=True, help="0=development; 1..6=frozen failures"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    require_df2_backend("cpu")
    case = dict(target_id="C04-T0005", events=4, target=6)
    base_path = Path("docs/amplitude-pilot/baseline.json")
    if args.index:
        case = json.loads(Path("docs/takeover-validation/cases.json").read_text())[args.index - 1]
        base_path = Path("results/takeover-validation/raw") / case["target_id"] / "baseline.json"
    base = json.loads(base_path.read_text())
    metadata, target = load_target(case["events"], case["target"])
    initial = EventPhrase(
        *(
            torch.tensor(base["best_phrase"][k], dtype=torch.float64)
            for k in ("f0_hz", "onset_seconds")
        )
    )
    synth = PhraseSynth()
    with torch.no_grad():
        audio = synth.render(target)
    diagonal = CumulativeEnergyDistance(audio)
    orthogonal = AxisLoss(diagonal)
    scale = base["best_loss"]
    engine = Engine(synth, audio, metadata, scale)
    permutation = torch.arange(case["events"])
    initial_description = engine.describe(initial_state(initial), permutation)
    assert abs(initial_description["loss"] - scale) < 1e-12
    signature = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    root = ROOT / case["target_id"]
    root.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        path = root / f"{variant}.json"
        if path.exists():
            assert json.loads(path.read_text())["script_sha256"] == signature
            continue
        start = time.perf_counter()
        raw = encode_coordinates(initial.f0_hz, initial.onset_seconds).requires_grad_()
        opt = torch.optim.Adam([raw], lr=0.05)
        best_loss, best_raw, best_update = scale, raw.detach().clone(), 0
        trajectory = []
        endpoint = None
        for step in range(EXPLORATION + POLISH + 1):
            if step == EXPLORATION:
                # Always refine the actual last rotated state, even if its diagonal loss is worse.
                opt = torch.optim.Adam([raw], lr=0.05)
            if step >= EXPLORATION:
                rate = 0.05 * 0.3 ** min((step - EXPLORATION) // 200, 3)
                opt.param_groups[0]["lr"] = rate
            f, t = decode_coordinates(raw)
            candidate = synth.render_batch(f[None], t[None])[0]
            d = diagonal.directional_distances(candidate)[0]
            canonical = d.mean()
            w = weights(step, variant) if step < EXPLORATION else weights(0, "diagonal_control")
            if step < EXPLORATION and variant != "diagonal_control":
                terms = torch.cat((d, orthogonal.terms(candidate[None])[0, :4]))
                loss = (terms * w).sum()
            else:
                terms, loss = d, canonical
            value = float(canonical.detach())
            if value < best_loss:
                best_loss, best_raw, best_update = value, raw.detach().clone(), step
            row = dict(
                update=step,
                canonical_loss=value,
                objective_loss=float(loss.detach()),
                best_canonical_loss=best_loss,
                weights=w.tolist(),
                directional_losses=terms.detach().tolist(),
                f0_hz=f.detach().tolist(),
                onset_seconds=t.detach().tolist(),
                lr=opt.param_groups[0]["lr"],
            )
            if step in (0, EXPLORATION, EXPLORATION + POLISH) or step % 100 == 0:
                description = engine.describe(
                    initial_state(EventPhrase(f.detach(), t.detach())), permutation
                )
                row["metrics"] = description["metrics"]
                print(case["target_id"], variant, step, best_loss, flush=True)
                if step == EXPLORATION:
                    endpoint = description
            trajectory.append(row)
            if step == EXPLORATION + POLISH:
                break
            opt.zero_grad()
            (loss / scale).backward()
            assert bool(torch.isfinite(raw.grad).all())
            opt.step()
        bf, bt = decode_coordinates(best_raw)
        final = engine.describe(initial_state(EventPhrase(bf.detach(), bt.detach())), permutation)
        assert abs(final["loss"] - best_loss) < 1e-12
        result = dict(
            case=case,
            variant=variant,
            baseline=initial_description,
            final=final,
            exploration_endpoint=endpoint,
            last=description,
            best_update=best_update,
            trajectory=trajectory,
            updates=EXPLORATION + POLISH,
            wall_seconds=time.perf_counter() - start,
            script_sha256=signature,
            baseline_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
            renderer=synth.provenance(),
            torch_version=torch.__version__,
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            directions=DIRECTIONS,
            clockwise_order=ORDER,
            period=PERIOD,
            ramp=RAMP,
            objective_scale=scale,
            amplitudes="fixed .8",
            log_weighing=False,
            selection="strict best original diagonal loss, including baseline",
            polish="actual exploration endpoint; fresh Adam; 800 updates",
        )
        temporary = path.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(path)
        print("COMPLETE", case["target_id"], variant, final, flush=True)


if __name__ == "__main__":
    main()
