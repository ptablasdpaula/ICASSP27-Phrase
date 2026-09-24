"""The paper's 224 target jobs: 160 main-design jobs and 64 State/7-D jobs."""

import argparse

import assess_limitation_columns as extra
import run_fixed_gradient_assessment as base
from icassp27_phrase.paths import OUTPUT, resolve_path
from icassp27_phrase.synth.config import CARDINALITIES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("qualify", "compute", "report", "all"))
    p.add_argument("--task", type=int, help="One task from 0 to 223; omit to run all")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--output", type=resolve_path, default=OUTPUT)
    a = p.parse_args()
    if a.batch < 1 or a.task is not None and not 0 <= a.task < 224:
        p.error("Invalid batch size or task")
    main_root = a.output / "gradient-analysis"
    extra_root = a.output / "gradient-limitations"
    if a.command in ("qualify", "all"):
        base.qualify(main_root, a.device, a.batch)
        extra.qualify(extra_root, a.device)
    if a.command in ("compute", "all"):
        for task in range(224) if a.task is None else (a.task,):
            if task < 160:
                group, index = divmod(task, 32)
                base.compute_cardinality(main_root, CARDINALITIES[group], a.batch, a.device, index)
            else:
                group, index = divmod(task - 160, 32)
                (extra.compute_persistent if group == 0 else extra.compute_all_controls)(
                    extra_root, index, a.batch, a.device
                )
    if a.command in ("report", "all"):
        base.report(main_root, main_root, None)
        extra.report(extra_root, extra_root)


if __name__ == "__main__":
    main()
