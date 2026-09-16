"""Paired patient swap trials with complete optimiser-state takeover."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
from _amplitude_fit import decode_coordinates, encode_coordinates
from icassp27_phrase.losses import build_loss
from icassp27_phrase.synth import PhraseSynth
from icassp27_phrase.targets import load_target
from test_amplitude_recovery import metrics

ROOT = Path("docs/counterfactual-takeover")


@dataclass
class State:
    raw: torch.Tensor
    first: torch.Tensor
    second: torch.Tensor
    step: int = 0
    lr: float = 0.05

    def clone(self):
        return State(self.raw.clone(), self.first.clone(), self.second.clone(), self.step, self.lr)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "result.json"
    if path.exists():
        raise FileExistsError(path)
    baseline = json.loads(Path("docs/amplitude-pilot/baseline.json").read_text())
    f, t = [
        torch.tensor(baseline["best_phrase"][k], dtype=torch.float64)
        for k in ["f0_hz", "onset_seconds"]
    ]
    metadata, target = load_target(4, 6)
    synth = PhraseSynth()
    with torch.no_grad():
        audio = synth.render(target)
    objective = build_loss("bidirectional_cumulative_energy", audio)
    scale = baseline["best_loss"]

    def evaluate(state, perm):
        with torch.no_grad():
            f, t = decode_coordinates(state.raw)
            value = float(objective(synth.render_batch(f[perm][None], t[None])[0]))
        assert math_isfinite(value)
        return value

    def advance(state, perm):
        raw = state.raw.detach().clone().requires_grad_()
        f, t = decode_coordinates(raw)
        loss = objective(synth.render_batch(f[perm][None], t[None])[0])
        (g,) = torch.autograd.grad(loss / scale, raw)
        assert bool(torch.isfinite(g).all())
        m = 0.9 * state.first + 0.1 * g
        v = 0.999 * state.second + 0.001 * g.square()
        n = state.step + 1
        updated = raw - state.lr * (m / (1 - 0.9**n)) / ((v / (1 - 0.999**n)).sqrt() + 1e-8)
        return State(updated.detach(), m.detach(), v.detach(), n, state.lr)

    def describe(state, perm):
        f, t = decode_coordinates(state.raw)
        return dict(
            loss=evaluate(state, perm),
            f0_hz=f[perm].tolist(),
            onset_seconds=t.tolist(),
            permutation=perm.tolist(),
            adam_step=state.step,
            learning_rate=state.lr,
            metrics=metrics(f[perm].numpy(), t.numpy(), metadata),
        )

    raw = encode_coordinates(f, t)
    live = State(raw, torch.zeros_like(raw), torch.zeros_like(raw))
    perm = torch.arange(4)
    initial = describe(live, perm)
    trials = []
    # One complete adjacent-pair sweep; freeze live during each paired trial.
    for i, j in [(0, 1), (1, 2), (2, 3)]:
        frozen = live.clone()
        before = evaluate(live, perm)
        alternative = perm.clone()
        alternative[i], alternative[j] = perm[j], perm[i]
        permutations = [perm.clone(), alternative]
        states = [live.clone(), live.clone()]
        bests = [s.clone() for s in states]
        values = [evaluate(s, p) for s, p in zip(states, permutations, strict=True)]
        references = values.copy()
        patiences = [0, 0]
        trace = []
        for depth in range(1, 301):
            for k in range(2):
                states[k] = advance(states[k], permutations[k])
                value = evaluate(states[k], permutations[k])
                patiences[k] += 1
                if value < values[k]:
                    values[k] = value
                    bests[k] = states[k].clone()
                if value < references[k] * (1 - 1e-4):
                    references[k] = value
                    patiences[k] = 0
            trace.append(
                dict(
                    depth=depth,
                    best_keep=values[0],
                    best_swap=values[1],
                    keep_patience=patiences[0],
                    swap_patience=patiences[1],
                )
            )
            if depth >= 30 and min(patiences) >= 40:
                break
        for attr in ["raw", "first", "second"]:
            assert torch.equal(getattr(live, attr), getattr(frozen, attr))
        assert live.step == frozen.step and live.lr == frozen.lr
        # Require swap to beat BOTH live and equally explored keep by 1%.
        accepted_swap = values[1] < min(before, values[0]) * 0.99
        chosen = 1 if accepted_swap else (0 if values[0] < before * 0.99 else None)
        if chosen is not None:
            live = bests[chosen].clone()
            perm = permutations[chosen].clone()
            assert abs(evaluate(live, perm) - values[chosen]) < 1e-12
            # The whole branch state, not just controls, is adopted.
            assert live.step == bests[chosen].step and live.lr == bests[chosen].lr
            for attr in ["raw", "first", "second"]:
                assert torch.equal(getattr(live, attr), getattr(bests[chosen], attr))
        record = dict(
            pair=[i + 1, j + 1],
            depth=depth,
            live_before=before,
            best_keep=values[0],
            best_swap=values[1],
            accepted_swap=accepted_swap,
            chosen=chosen,
            live_after=describe(live, perm),
            trace=trace,
        )
        trials.append(record)
        print("TRIAL", i + 1, j + 1, depth, accepted_swap, record["live_after"], flush=True)
    adopted = describe(live, perm)
    # Continue with the adopted moments/count/LR; reduce LR only on later plateaus.
    best = live.clone()
    best_value = evaluate(live, perm)
    reference = best_value
    patience = 0
    plateaus = 0
    continuation = []
    for update in range(3001):
        value = evaluate(live, perm)
        if update:
            patience += 1
        if value < best_value:
            best_value = value
            best = live.clone()
        if value < reference * (1 - 1e-4):
            reference = value
            patience = 0
            plateaus = 0
        if update % 100 == 0:
            continuation.append(dict(update=update, **describe(live, perm)))
            print("CONTINUE", update, best_value, flush=True)
        if patience >= 250 or update == 3000:
            break
        if patience >= (plateaus + 1) * 100:
            plateaus += 1
            lr = max(live.lr * 0.3, 1e-5)
            live = best.clone()
            live.lr = lr
            live.first.zero_()
            live.second.zero_()
        live = advance(live, perm)
    final = describe(best, perm)
    # Save exact selected state for future continuation, including moments.
    checkpoint = dict(
        raw=best.raw.tolist(),
        first=best.first.tolist(),
        second=best.second.tolist(),
        step=best.step,
        lr=best.lr,
        permutation=perm.tolist(),
        loss_scale=scale,
    )
    result = dict(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        initial=initial,
        trials=trials,
        adopted=adopted,
        continuation=continuation,
        final=final,
        final_state=checkpoint,
        continuation_updates=update,
        checks="exact live-state isolation, exact full-state takeover and loss reproduction passed",
    )
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("FINAL", final, flush=True)


def math_isfinite(value):
    import math

    return math.isfinite(value)


if __name__ == "__main__":
    main()
