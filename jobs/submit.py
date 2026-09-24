"""Submit explicit Slurm arrays; default to a readable dry run."""

import argparse
import json
import os
import shlex
import subprocess
import tomllib

from icassp27_phrase.paths import CACHE, LOGS, REPO, scientific_signature


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "experiment", choices=("gradients", "recovery", "efficiency", "slices", "qualify")
    )
    p.add_argument("--profile", default=os.getenv("PHRASE_SLURM_PROFILE", "generic"))
    p.add_argument("--profiles", default=str(REPO / "jobs/profiles.toml"))
    p.add_argument("--partition")
    p.add_argument("--account")
    p.add_argument("--concurrency", type=int)
    p.add_argument("--array", help="Explicit subset, e.g. 0-15 or 0,3,8")
    p.add_argument(
        "--submit", action="store_true", help="Actually call sbatch instead of printing the command"
    )
    a = p.parse_args()
    from icassp27_phrase.paths import resolve_path

    profiles = tomllib.loads(resolve_path(a.profiles).read_text())
    if a.profile not in profiles:
        p.error("Unknown profile")
    config = profiles[a.profile].copy()
    for name in ("partition", "account", "concurrency"):
        if getattr(a, name) is not None:
            config[name] = getattr(a, name)
    if config["concurrency"] < 1:
        p.error("Concurrency must be positive")
    count = {"gradients": 224, "recovery": 95, "efficiency": 5, "slices": 1, "qualify": 1}[
        a.experiment
    ]
    array = a.array or f"0-{count - 1}"
    import re

    if not re.fullmatch(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*", array):
        p.error("Invalid array specification")
    for part in array.split(","):
        numbers = list(map(int, part.split("-")))
        if min(numbers) < 0 or max(numbers) >= count or numbers[0] > numbers[-1]:
            p.error("Task outside experiment range")
    args = [
        "sbatch",
        "--parsable",
        f"--chdir={REPO}",
        f"--job-name=phrase-{a.experiment}",
        f"--array={array}%{config['concurrency']}",
        f"--cpus-per-task={config['cpus']}",
        f"--mem={config['memory']}",
        f"--time={config['time']}",
        f"--gres={config['gres']}",
        f"--output={LOGS}/{a.experiment}-%A_%a.log",
    ]
    for name in ("partition", "account", "constraint"):
        if config.get(name):
            args.append(f"--{name}={config[name]}")
    if a.experiment == "efficiency":
        args = [x for x in args if not x.startswith("--gres=")]
        args.append("--gres=gpu:nvidia_a100-pcie-40gb:1")
    args.extend([str(REPO / "jobs/run.sh"), a.experiment])
    print(shlex.join(args))
    if a.submit:
        LOGS.mkdir(parents=True, exist_ok=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PHRASE_REPO"] = str(REPO)
        env["PHRASE_SOURCE_SIGNATURE"] = scientific_signature()[0]
        result = subprocess.run(args, check=True, text=True, capture_output=True, env=env)
        job = result.stdout.strip()
        print(job)
        (LOGS / f"submission-{job.split(';')[0]}.json").write_text(
            json.dumps(
                {
                    "command": args,
                    "signature": scientific_signature()[0],
                    "job": job,
                    "profile": config,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
