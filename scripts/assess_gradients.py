"""Qualify, benchmark, run and report the paired eleven-loss gradient assessment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from icassp27_phrase.gradient_assessment import (
    COLUMNS,
    NAMES,
    SCHEMA,
    SEED,
    SharedObjectives,
    candidates,
    matching,
    score,
    signature,
    targets,
)
from icassp27_phrase.losses import SOT_MSS_HOPS, SOT_MSS_WINDOWS, _stft, build_loss
from icassp27_phrase.runtime import configure_reproducibility, require_df2_backend
from icassp27_phrase.synth import PhraseSynth

ROOT = Path("results/gradient-assessment")
DOCS = Path("docs/gradient-assessment")


def setup(device="cpu"):
    torch.set_num_threads(1)
    configure_reproducibility(SEED)
    require_df2_backend(device)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def shard_path(root, name, condition, count, repeat):
    return root / "raw" / f"n{count:04d}" / f"r{repeat}" / f"{name}-{condition}.npz"


def compute_task(task):
    root, index, count, repeat, conditions, batch, device = task
    setup(device)
    name, target = targets()[index]
    design = candidates(name, target, count, repeat)
    sig, _ = signature()
    pending = []
    for condition in conditions:
        if condition not in design:
            continue
        path = shard_path(Path(root), name, condition, count, repeat)
        if path.exists():
            data = checked_data(path)
            if (
                str(data["signature"]) != sig
                or str(data["device"]) != device
                or not np.array_equal(data["candidates"], design[condition])
                or not np.array_equal(data["target"], target)
            ):
                raise ValueError(f"stale checkpoint: {path}")
            continue
        pending.append((condition, path))
    if not pending:
        return name
    synth = PhraseSynth().to(device)
    with torch.no_grad():
        coords = torch.tensor(target, dtype=torch.float64, device=device)
        audio = synth(80 * 2 ** coords[None, :, 0], coords[None, :, 1])[0]
        objective = SharedObjectives(audio)
    for condition, path in pending:
        started = time.perf_counter()
        positions = design[condition]
        values, gradients = [], []
        for begin in range(0, count, batch):
            value, gradient = objective.evaluate(synth, positions[begin : begin + batch])
            values.append(value)
            gradients.append(gradient)
        assignments, ties = zip(*(matching(p, target) for p in positions), strict=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            schema=SCHEMA,
            signature=sig,
            target=target,
            candidates=positions,
            losses=np.concatenate(values),
            gradients=np.concatenate(gradients),
            assignments=np.stack(assignments),
            ties=np.array(ties),
            names=np.array(NAMES),
            condition=condition,
            repeat=repeat,
            count=count,
            target_id=name,
            renderer_json=json.dumps(synth.provenance(), sort_keys=True),
            torch_version=torch.__version__,
            device=device,
            wall_seconds=time.perf_counter() - started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        temporary.replace(path)
        print(
            f"DONE {name} {condition} n={count} r={repeat} {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return name


def reference_values(audio, target, bank):
    """Independent historical paths, including the original scalar fading class."""
    from test_fading_diagonal import FadingDiagonal

    result = {}
    for name in NAMES:
        if name == "single_stft":
            mag = _stft(
                audio, n_fft=256, hop=64, window=bank.cel.window, center=False, pad_mode="constant"
            ).abs()
            ref = _stft(
                target[None],
                n_fft=256,
                hop=64,
                window=bank.cel.window,
                center=False,
                pad_mode="constant",
            ).abs()
            value = (mag - ref).abs().mean((-2, -1))
        elif name == "linear_mss":
            value = audio.new_zeros(len(audio))
            for n, hop, window, ref in zip(
                SOT_MSS_WINDOWS,
                SOT_MSS_HOPS,
                bank.sot.mss_windows,
                bank.sot.mss_targets,
                strict=True,
            ):
                mag = _stft(
                    audio, n_fft=n, hop=hop, window=window, center=True, pad_mode="reflect"
                ).abs()
                value = value + (mag - ref).abs().mean((-2, -1))
        elif name == "fading":
            objective = FadingDiagonal(target, 1.0, 1000.0)
            value = torch.stack([objective(row) for row in audio])
        else:
            value = build_loss(name, target)(audio)
        result[name] = value
    return result


def qualify(root, device="cpu"):
    setup(device)
    synth = PhraseSynth()
    assert synth.provenance()["exciter"]["fourier_fft_length"] == 16384
    checks = []
    for index in (0, 1, 33):
        name, target = targets()[index]
        positions = candidates(name, target)["joint"][:2].copy()
        near = target.copy()
        near[:, 0] = np.clip(near[:, 0] + 0.001, 0.0, 2.0)
        near[:, 1] = np.clip(near[:, 1] + 0.0013, 0.2, 1.8)
        positions = np.concatenate((positions, near[None], positions[:1, ::-1]), axis=0)
        with torch.no_grad():
            coords = torch.tensor(target, dtype=torch.float64)
            audio = synth(80 * 2 ** coords[None, :, 0], coords[None, :, 1])[0]
            bank = SharedObjectives(audio)
        actual_v, actual_g = bank.evaluate(synth, positions)
        coordinates = torch.tensor(positions, dtype=torch.float64, requires_grad=True)
        candidate_audio = synth(80 * 2 ** coordinates[..., 0], coordinates[..., 1])
        reference = reference_values(candidate_audio, audio, bank)
        for i, loss in enumerate(NAMES):
            (expected_g,) = torch.autograd.grad(
                reference[loss].sum(), coordinates, retain_graph=i < len(NAMES) - 1
            )
            np.testing.assert_allclose(
                actual_v[:, i], reference[loss].detach().numpy(), rtol=1e-9, atol=1e-11
            )
            np.testing.assert_allclose(
                actual_g[:, i], expected_g.detach().numpy(), rtol=1e-9, atol=1e-11
            )
        separate = [bank.evaluate(synth, row[None]) for row in positions]
        np.testing.assert_allclose(
            actual_v, np.concatenate([v for v, _ in separate]), rtol=1e-9, atol=1e-11
        )
        np.testing.assert_allclose(
            actual_g, np.concatenate([g for _, g in separate]), rtol=1e-9, atol=1e-11
        )
        np.testing.assert_allclose(actual_v[0], actual_v[3], rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(actual_g[0], actual_g[3, :, ::-1], rtol=1e-9, atol=1e-11)
        if device == "cuda":
            gpu_synth = PhraseSynth().to(device)
            with torch.no_grad():
                gpu_coords = torch.tensor(target, dtype=torch.float64, device=device)
                gpu_audio = gpu_synth(80 * 2 ** gpu_coords[None, :, 0], gpu_coords[None, :, 1])[0]
                gpu_bank = SharedObjectives(gpu_audio)
            gpu_v, gpu_g = gpu_bank.evaluate(gpu_synth, positions)
            np.testing.assert_allclose(gpu_v, actual_v, rtol=1e-6, atol=1e-8)
            np.testing.assert_allclose(gpu_g, actual_g, rtol=1e-6, atol=1e-8)
            # Direction is the endpoint, so also require identical signs except
            # numerically negligible components under the agreement tolerance.
            significant = np.abs(actual_g) > 1e-8 + 1e-6 * np.abs(actual_g)
            assert np.array_equal(np.sign(gpu_g[significant]), np.sign(actual_g[significant]))
            separate_gpu = [gpu_bank.evaluate(gpu_synth, row[None]) for row in positions]
            np.testing.assert_allclose(
                gpu_g, np.concatenate([g for _, g in separate_gpu]), rtol=1e-6, atol=1e-8
            )
        checks.append({"target": name, "events": len(target), "passed": True})
        print(f"QUALIFIED {name}: values, gradients, batches, permutation", flush=True)
    sig, hashes = signature()
    save_json(
        root / "qualification.json",
        {
            "passed": True,
            "signature": sig,
            "source_hashes": hashes,
            "checks": checks,
            "rtol": 1e-9,
            "atol": 1e-11,
            "renderer": synth.provenance(),
            "device": device,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "cross_backend_rtol": 1e-6 if device == "cuda" else None,
            "cross_backend_atol": 1e-8 if device == "cuda" else None,
        },
    )


def benchmark(root, batch, device="cpu"):
    setup(device)
    results = []
    for index in (0, 1, 33):
        name, target = targets()[index]
        synth = PhraseSynth().to(device)
        coords = torch.tensor(target, dtype=torch.float64, device=device)
        with torch.no_grad():
            audio = synth(80 * 2 ** coords[None, :, 0], coords[None, :, 1])[0]
            bank = SharedObjectives(audio)
        positions = candidates(name, target)["joint"][: max(32, 2 * batch)]
        bank.evaluate(synth, positions[:batch])
        started = time.perf_counter()
        for begin in range(0, len(positions), batch):
            bank.evaluate(synth, positions[begin : begin + batch])
        seconds = time.perf_counter() - started
        row = dict(
            events=len(target),
            batch=batch,
            candidates=len(positions),
            seconds=seconds,
            seconds_per_candidate=seconds / len(positions),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            device=device,
            gpu_peak_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
            gpu=torch.cuda.get_device_name() if device == "cuda" else None,
        )
        results.append(row)
        print(json.dumps(row), flush=True)
    save_json(root / f"benchmark-b{batch}.json", results)


def checked_data(path):
    with np.load(path) as source:
        data = dict(source)
    if str(data["signature"]) != signature()[0]:
        raise ValueError(f"stale shard: {path}")
    count, events, dimensions = data["candidates"].shape
    if (
        str(data["schema"]) != SCHEMA
        or dimensions != 2
        or tuple(data["names"]) != NAMES
        or data["losses"].shape != (count, len(NAMES))
        or data["gradients"].shape != (count, len(NAMES), events, 2)
        or data["assignments"].shape != (count, events)
        or data["ties"].shape != (count,)
    ):
        raise ValueError(f"incomplete/invalid shard: {path}")
    if not np.isfinite(data["losses"]).all() or not np.isfinite(data["gradients"]).all():
        raise FloatingPointError(path)
    np.testing.assert_array_equal(
        np.sort(data["assignments"], axis=-1), np.broadcast_to(np.arange(events), (count, events))
    )
    return data


def scores_from(data):
    return score(
        data["gradients"], data["candidates"], data["target"], data["assignments"], data["ties"]
    )


def finite_mean(values, axis=0):
    values = np.asarray(values)
    count = np.isfinite(values).sum(axis=axis)
    return np.divide(
        np.nansum(values, axis=axis), count, out=np.full(np.shape(count), np.nan), where=count > 0
    )


def sensitivity(root, column, count, metric="both_directed"):
    events, condition = column
    indices = [i for i, (_, t) in enumerate(targets()) if len(t) == events][:8]
    paired = [[], []]
    for i in indices:
        name, _ = targets()[i]
        for repeat in (0, 1):
            data = checked_data(shard_path(root, name, condition, count, repeat))
            paired[repeat].append(finite_mean(scores_from(data)[metric]))
    means = np.array([np.mean(rows, axis=0) for rows in paired])
    difference = 100 * np.abs(means[0] - means[1])
    return dict(
        events=events,
        condition=condition,
        count=count,
        target_count=len(indices),
        original_percent=(100 * means[0]).tolist(),
        repeat_percent=(100 * means[1]).tolist(),
        difference_pp=difference.tolist(),
        passed=bool(difference.max() <= 2.0),
    )


def campaign(root, batch, workers, device="cpu"):
    qualification = json.loads((root / "qualification.json").read_text())
    if (
        not qualification["passed"]
        or qualification["signature"] != signature()[0]
        or qualification["device"] != device
    ):
        raise ValueError("missing/stale qualification")
    if device == "cuda":
        setup(device)
        # The GPU can amortise synthesis/autograd launch overhead over the whole
        # LHS. Extend tuning beyond the conservative initial benchmark sizes.
        limit = 0.65 * torch.cuda.get_device_properties(0).total_memory
        for larger in (128, 256):
            previous = json.loads((root / f"benchmark-b{larger // 2}.json").read_text())
            if max(r["gpu_peak_bytes"] for r in previous) * 2 > limit:
                break
            try:
                benchmark(root, larger, device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"GPU tuning stopped at batch {larger}: memory limit", flush=True)
                break
        options = []
        for path in root.glob("benchmark-b*.json"):
            rows = json.loads(path.read_text())
            if rows[0].get("device") == device and max(r["gpu_peak_bytes"] for r in rows) < limit:
                options.append(
                    (
                        sum(r["seconds_per_candidate"] for r in rows if r["events"] > 1),
                        rows[0]["batch"],
                    )
                )
        batch = min(options)[1]
        torch.cuda.empty_cache()
        print(f"Final GPU batch: {batch}", flush=True)
    registry = targets()
    save_json(
        root / "design.json",
        {
            "schema": SCHEMA,
            "seed": SEED,
            "signature": signature()[0],
            "losses": NAMES,
            "columns": COLUMNS,
            "targets": {name: t.tolist() for name, t in registry},
            "coordinates": ["log2(f/80) octaves", "onset seconds"],
            "matching": "unrestricted Hungarian: squared octaves + squared seconds",
            "base_candidates": 256,
            "batch": batch,
            "workers": workers,
            "device": device,
        },
    )
    active, final_counts, history = list(COLUMNS), {}, []
    executor = (
        ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
        if workers > 1
        else nullcontext()
    )
    with executor as pool:
        for count in (256, 512, 1024):
            tasks = []
            for index, (name, target) in enumerate(registry):
                conditions = [c for n, c in active if n == len(target)]
                if not conditions:
                    continue
                repeats = (0, 1) if int(name.split("T")[1]) < 8 else (0,)
                for repeat in repeats:
                    tasks.append((str(root), index, count, repeat, conditions, batch, device))
            results = pool.map(compute_task, tasks) if pool else map(compute_task, tasks)
            for _ in results:
                pass
            following = []
            for column in active:
                check = sensitivity(root, column, count)
                history.append(check)
                print("SENSITIVITY", json.dumps(check), flush=True)
                if check["passed"] or count == 1024:
                    final_counts[f"{column[0]}-{column[1]}"] = count
                else:
                    following.append(column)
            save_json(
                root / "sampling.json",
                {
                    "signature": signature()[0],
                    "final_counts": final_counts,
                    "checks": history,
                    "complete": not following,
                },
            )
            active = following
            if not active:
                break
    print("CAMPAIGN COMPLETE", flush=True)


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: "" if isinstance(v, float) and not np.isfinite(v) else v
                    for k, v in row.items()
                }
            )


def report(root, output, initial_count=None, metric="both_directed"):
    sampling = (
        json.loads((root / "sampling.json").read_text())
        if initial_count is None
        else {
            "signature": signature()[0],
            "complete": True,
            "status": "initial fixed-count preview; sensitivity campaign may still be running",
            "final_counts": {f"{n}-{condition}": initial_count for n, condition in COLUMNS},
            "checks": [],
        }
    )
    if not sampling["complete"] or sampling["signature"] != signature()[0]:
        raise ValueError("campaign incomplete/stale")
    if metric != "both_directed":
        sampling = {
            **sampling,
            "metric": metric,
            "sample_count_selection_metric": "both_directed",
            "note": "Paired rescore of existing final samples; no new sampling or gradients.",
            "checks": [
                sensitivity(
                    root, column, sampling["final_counts"][f"{column[0]}-{column[1]}"], metric
                )
                for column in COLUMNS
            ],
        }
    output.mkdir(parents=True, exist_ok=True)
    target_rows, summary, quality, matrix, artifacts = [], [], [], [], {}
    for events, condition in COLUMNS:
        count = sampling["final_counts"][f"{events}-{condition}"]
        collected = {}
        for name, target in targets():
            if len(target) != events:
                continue
            path = shard_path(root, name, condition, count, 0)
            data = checked_data(path)
            assert np.array_equal(data["candidates"], candidates(name, target, count)[condition])
            metrics = scores_from(data)
            for metric_name, values in metrics.items():
                mean = finite_mean(values)
                collected.setdefault(metric_name, []).append(mean)
                for i, loss in enumerate(NAMES):
                    target_rows.append(
                        dict(
                            events=events,
                            condition=condition,
                            target=name,
                            candidates=count,
                            loss=loss,
                            metric=metric_name,
                            mean=float(mean[i]),
                            eligible=int(np.isfinite(values[:, i]).sum()),
                        )
                    )
            assignments = data["assignments"]
            active = np.abs(target[assignments] - data["candidates"]) > 1e-12
            quality.append(
                dict(
                    events=events,
                    condition=condition,
                    target=name,
                    candidates=count,
                    ties=int(data["ties"].sum()),
                    reassigned_candidates=int((assignments != np.arange(events)).any(-1).sum()),
                    both_displaced_events=int(active.all(-1).sum()),
                    wall_seconds=float(data["wall_seconds"]),
                    peak_rss_kib=int(data["peak_rss_kib"]),
                )
            )
            artifacts[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        matrix.append(np.mean(collected[metric], axis=0) * 100)
        for metric_name, values in collected.items():
            array = np.array(values)
            for i, loss in enumerate(NAMES):
                valid = array[:, i][np.isfinite(array[:, i])]
                summary.append(
                    dict(
                        events=events,
                        condition=condition,
                        candidates_per_target=count,
                        targets=len(array),
                        loss=loss,
                        metric=metric_name,
                        mean=float(valid.mean()) if len(valid) else float("nan"),
                        sd=float(valid.std(ddof=1)) if len(valid) > 1 else float("nan"),
                        median=float(np.median(valid)) if len(valid) else float("nan"),
                    )
                )
    write_csv(output / "target-scores.csv", target_rows)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "quality.csv", quality)
    save_json(output / "sampling.json", sampling)
    save_json(output / "design.json", json.loads((root / "design.json").read_text()))
    save_json(output / "qualification.json", json.loads((root / "qualification.json").read_text()))
    matrix = np.array(matrix).T
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        r"$L_1$",
        r"$L_2$",
        "Single STFT",
        "Linear MSS",
        "Smooth MSS",
        "SOT",
        r"$\mathrm{TF}\mathcal{W}_2$",
        r"log-$\mathrm{TF}\mathcal{W}_2$",
        r"Ce$\mathcal{L}$",
        r"Log-Ce$\mathcal{L}$",
        r"Fade-Ce$\mathcal{L}$",
    ]
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 3.25))
    fig.subplots_adjust(left=0.245, right=0.985, top=0.84, bottom=0.17)
    im = ax.imshow(matrix, cmap="cividis", vmin=0, vmax=100, aspect="auto")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(7), ["1", "2", "4", "2", "4", "2", "4"])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=3)
    for center, title in ((0, "Both"), (1.5, "Pitch"), (3.5, "Time"), (5.5, "Both")):
        ax.text(center, -1.25, title, ha="center", va="bottom", clip_on=False)
    for i in range(11):
        for j in range(7):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[i, j] < 48 else "black",
            )
    ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 11, 1), minor=True)
    ax.grid(which="minor", color="white", alpha=0.4, linewidth=0.4)
    ax.tick_params(which="minor", length=0)
    for boundary in (0.5, 2.5, 4.5):
        ax.axvline(boundary, color="white", linewidth=1.1)
    for boundary in (1.5, 4.5, 7.5):
        ax.axhline(boundary, color="white", linewidth=1.1)
    cax = fig.add_axes([0.245, 0.085, 0.74, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    cb.set_label(
        "Positive dot product (%)" if metric == "dot_directed" else "Target-directed events (%)",
        labelpad=2,
    )
    cb.ax.tick_params(length=2, pad=2)
    cb.ax.get_xticklabels()[0].set_horizontalalignment("left")
    cb.ax.get_xticklabels()[-1].set_horizontalalignment("right")
    fig.savefig(output / "gradient-assessment.pdf")
    fig.savefig(output / "gradient-assessment.png", dpi=300)
    plt.close(fig)
    provenance = dict(
        schema=SCHEMA,
        signature=signature()[0],
        source_hashes=signature()[1],
        raw_artifacts=artifacts,
        candidates=sum(q["candidates"] for q in quality),
        losses=NAMES,
        columns=COLUMNS,
        renderer=PhraseSynth().provenance(),
        final_percentages=matrix.tolist(),
        metric=metric,
    )
    save_json(output / "provenance.json", provenance)
    print("REPORT", output / "gradient-assessment.png", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("qualify", "benchmark", "campaign", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DOCS)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--metric", choices=("both_directed", "dot_directed"), default="both_directed"
    )
    parser.add_argument(
        "--initial-count",
        type=int,
        help="Report a completed fixed-count stage separately from final results",
    )
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if args.command == "qualify":
        qualify(args.root, args.device)
    elif args.command == "benchmark":
        benchmark(args.root, args.batch, args.device)
    elif args.command == "campaign":
        if args.device == "cuda" and args.workers != 1:
            raise ValueError("Use one GPU worker with batched candidates")
        campaign(args.root, args.batch, args.workers, args.device)
    else:
        if args.initial_count is not None and args.output == DOCS:
            raise ValueError("Use a separate --output directory for the initial preview")
        if args.metric != "both_directed" and args.output == DOCS:
            raise ValueError("Use a separate --output directory for alternative scores")
        report(args.root, args.output, args.initial_count, args.metric)


if __name__ == "__main__":
    main()
