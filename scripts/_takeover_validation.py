"""Frozen takeover pilot settings, generalized to multiple event counts."""

from __future__ import annotations

import math

import numpy as np
import torch
from _amplitude_fit import decode_coordinates, encode_coordinates
from icassp27_phrase.losses import build_loss
from test_amplitude_recovery import metrics
from test_counterfactual_takeover import State


class Engine:
    def __init__(self, synth, target_audio, metadata, scale):
        self.synth = synth
        self.audio = target_audio
        self.metadata = metadata
        self.objective = build_loss("bidirectional_cumulative_energy", target_audio)
        self.scale = scale
        self.backwards = 0
        self.forwards = 0

    def value(self, state, permutation):
        with torch.no_grad():
            f, t = decode_coordinates(state.raw)
            audio = self.synth.render_batch(f[permutation][None], t[None])[0]
            value = float(self.objective(audio))
        self.forwards += 1
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite canonical loss")
        return value

    def advance(self, state, permutation):
        raw = state.raw.detach().clone().requires_grad_()
        f, t = decode_coordinates(raw)
        loss = self.objective(self.synth.render_batch(f[permutation][None], t[None])[0])
        (g,) = torch.autograd.grad(loss / self.scale, raw)
        assert bool(torch.isfinite(g).all())
        m = 0.9 * state.first + 0.1 * g
        v = 0.999 * state.second + 0.001 * g.square()
        n = state.step + 1
        updated = raw - state.lr * (m / (1 - 0.9**n)) / ((v / (1 - 0.999**n)).sqrt() + 1e-8)
        self.backwards += 1
        self.forwards += 1
        return State(updated.detach(), m.detach(), v.detach(), n, state.lr)

    def describe(self, state, permutation):
        f, t = decode_coordinates(state.raw)
        f = f[permutation]
        result = metrics(f.numpy(), t.numpy(), self.metadata)
        assignment = result["assignment"]
        tf = np.asarray(self.metadata.f0_hz)[assignment]
        tt = np.asarray(self.metadata.onset_seconds)[assignment]
        cost = np.square(np.log2(f.numpy() / tf) / 2) + np.square((t.numpy() - tt) / 1.6)
        result["joint_event_error"] = float(np.sqrt(cost).mean())
        # Objective-independent waveform diagnostic; not used for acceptance.
        with torch.no_grad():
            candidate = self.synth.render_batch(f[None], t[None])[0]
            result["relative_waveform_l2"] = float(
                (candidate - self.audio).norm() / self.audio.norm()
            )
        self.forwards += 1
        return dict(
            loss=self.value(state, permutation),
            f0_hz=f.tolist(),
            onset_seconds=t.tolist(),
            permutation=permutation.tolist(),
            metrics=result,
            adam_step=state.step,
            lr=state.lr,
        )


def initial_state(phrase):
    # Canonicalise event labels once; audio and continuous controls do not change.
    order = phrase.onset_seconds.argsort(stable=True)
    raw = encode_coordinates(phrase.f0_hz[order], phrase.onset_seconds[order])
    return State(raw, torch.zeros_like(raw), torch.zeros_like(raw))


def serialize_state(state, permutation):
    return dict(
        raw=state.raw.tolist(),
        first=state.first.tolist(),
        second=state.second.tolist(),
        step=state.step,
        lr=state.lr,
        permutation=permutation.tolist(),
    )


def paired_sweep(engine, initial, permutation, *, progress=print):
    live = initial.clone()
    perm = permutation.clone()
    records = []
    n = len(perm)
    for rank in range(n - 1):
        # Swap adjacent events in current chronological order, not arbitrary latent slots.
        _, times = decode_coordinates(live.raw)
        order = times.argsort(stable=True)
        i, j = int(order[rank]), int(order[rank + 1])
        frozen = live.clone()
        before = engine.value(live, perm)
        alternative = perm.clone()
        alternative[i], alternative[j] = perm[j], perm[i]
        permutations = [perm.clone(), alternative]
        states = [live.clone(), live.clone()]
        bests = [s.clone() for s in states]
        values = [engine.value(s, p) for s, p in zip(states, permutations, strict=True)]
        references = values.copy()
        patience = [0, 0]
        trace = []
        for depth in range(1, 301):
            current = []
            for k in range(2):
                states[k] = engine.advance(states[k], permutations[k])
                value = engine.value(states[k], permutations[k])
                current.append(value)
                patience[k] += 1
                if value < values[k]:
                    values[k] = value
                    bests[k] = states[k].clone()
                if value < references[k] * (1 - 1e-4):
                    references[k] = value
                    patience[k] = 0
            trace.append(
                dict(
                    depth=depth,
                    keep=current[0],
                    swap=current[1],
                    best_keep=values[0],
                    best_swap=values[1],
                )
            )
            if depth >= 30 and min(patience) >= 40:
                break
        for attr in ["raw", "first", "second"]:
            assert torch.equal(getattr(live, attr), getattr(frozen, attr))
        assert live.step == frozen.step and live.lr == frozen.lr
        accepted = values[1] < min(before, values[0]) * 0.99
        chosen = 1 if accepted else (0 if values[0] < before * 0.99 else None)
        if chosen is not None:
            live = bests[chosen].clone()
            perm = permutations[chosen].clone()
            assert abs(engine.value(live, perm) - values[chosen]) < 1e-12
            for attr in ["raw", "first", "second"]:
                assert torch.equal(getattr(live, attr), getattr(bests[chosen], attr))
            assert live.step == bests[chosen].step and live.lr == bests[chosen].lr
        record = dict(
            adjacent_ranks=[rank + 1, rank + 2],
            slots=[i, j],
            depth=depth,
            before=before,
            best_keep=values[0],
            best_swap=values[1],
            accepted_swap=accepted,
            chosen=chosen,
            after=engine.describe(live, perm),
            trace=trace,
        )
        assert record["after"]["loss"] <= before + 1e-12
        records.append(record)
        progress(
            f"pair {rank + 1}/{n - 1}: depth={depth} swap={accepted} "
            f"loss={record['after']['loss']:.8g}"
        )
    return live, perm, records


def descend(engine, initial, permutation, maximum_updates=3000, *, progress=print):
    live = initial.clone()
    best = live.clone()
    best_value = engine.value(live, permutation)
    reference = best_value
    patience = 0
    plateaus = 0
    trace = []
    for update in range(maximum_updates + 1):
        value = engine.value(live, permutation)
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
            trace.append(dict(update=update, loss=value, best_loss=best_value, lr=live.lr))
        if update % 500 == 0:
            progress(f"descent {update}: {best_value:.8g}")
        if patience >= 250 or update == maximum_updates:
            break
        if patience >= (plateaus + 1) * 100:
            plateaus += 1
            lr = max(live.lr * 0.3, 1e-5)
            live = best.clone()
            live.lr = lr
            live.first.zero_()
            live.second.zero_()
        live = engine.advance(live, permutation)
    return best, dict(
        updates=update, trace=trace, stopped_by="patience" if patience >= 250 else "budget"
    )


def matched_control(engine, initial, permutation, budget, *, progress=print):
    best = initial.clone()
    spent = 0
    attempts = []
    while spent < budget:
        # Restart from best with fresh moments/LR, as in the first takeover trial.
        restart = State(
            best.raw.clone(), torch.zeros_like(best.first), torch.zeros_like(best.second)
        )
        best, record = descend(engine, restart, permutation, budget - spent, progress=progress)
        spent += record["updates"]
        record["cumulative_updates"] = spent
        attempts.append(record)
        if record["updates"] == 0:
            raise RuntimeError("control made no progress toward allocated budget")
    assert engine.backwards == budget
    return best, attempts
