"""Command-line interface for beginners and repeatable project runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_RANDOM_SEED, PipelineConfig
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic HR data and train an employee performance classifier."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of synthetic employees to generate (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_SEED}).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction reserved for testing (default: 0.20).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project directory where data, models, outputs, and images are saved.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = PipelineConfig(
            n_samples=args.samples,
            test_size=args.test_size,
            random_seed=args.seed,
            project_root=args.project_root.resolve(),
        )
        result = run_pipeline(config)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    print("Employee Performance Predictor completed successfully.")
    print(f"Generated rows: {result.metrics['n_samples']}")
    print(f"Held-out accuracy: {result.metrics['accuracy']:.3f}")
    print(f"Held-out F1 score: {result.metrics['f1_score']:.3f}")
    print(f"ROC-AUC: {result.metrics['roc_auc']:.3f}" if result.metrics["roc_auc"] is not None else "ROC-AUC: unavailable")
    print("Artifacts:")
    for name, path in result.artifact_paths.items():
        print(f"  - {name}: {path}")
    return 0
