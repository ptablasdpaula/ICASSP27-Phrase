#!/usr/bin/env python3
"""Compute or assemble the supplementary renderer-sensitivity explorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from icassp27_phrase.renderer_screen import assemble_web_data, compute_cell


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    cell = commands.add_parser("compute-cell")
    cell.add_argument("--index", required=True, type=int)
    cell.add_argument("--output-root", required=True, type=Path)
    cell.add_argument("--device", default="cuda")
    cell.add_argument("--surface-chunk-size", type=int, default=16)
    cell.add_argument("--gradient-chunk-size", type=int, default=4)
    assemble = commands.add_parser("assemble")
    assemble.add_argument("--input-root", required=True, type=Path)
    assemble.add_argument("--output-root", required=True, type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.command == "compute-cell":
        report = compute_cell(
            args.index,
            args.output_root,
            device_name=args.device,
            surface_chunk_size=args.surface_chunk_size,
            gradient_chunk_size=args.gradient_chunk_size,
        )
    else:
        report = assemble_web_data(args.input_root, args.output_root)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
