"""Command-line entry point.

Usage::

    python -m robustness_lab.cli --config experiments/configs/baseline.yaml \\
        --tag baseline --device auto
"""

from __future__ import annotations

import argparse
import logging

import torch

from .config import load_config
from .evaluation.report import save_results
from .evaluation.runner import run_benchmark


def _resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MNIST robustness benchmark")
    parser.add_argument("--config", required=True, help="path to YAML experiment config")
    parser.add_argument("--output", default="", help="override results output directory")
    parser.add_argument("--tag", default="", help="name tag for output files")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"], help="device override"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    if args.output:
        cfg.output_dir = args.output
    device = _resolve_device(args.device)

    results = run_benchmark(cfg, device)
    json_path = save_results(results, cfg.output_dir, tag=args.tag)

    print(
        f"[done] defense={results['defense']} device={device} "
        f"clean_acc={results['clean_accuracy']:.4f} -> {json_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
