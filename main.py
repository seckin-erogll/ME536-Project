"""Entry point for the Auto-Schematic project."""

from __future__ import annotations

import argparse
from pathlib import Path

from FH_Circuit.config import DEFAULT_SAMPLES_PER_CLASS
from FH_Circuit.data import build_dataset, save_samples
from FH_Circuit.gui import run_gui
from FH_Circuit.training import save_artifacts, train_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-Schematic utilities.")
    subparsers = parser.add_subparsers(dest="command")

    generate_parser = subparsers.add_parser("generate", help="Generate synthetic dataset images.")
    generate_parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES_PER_CLASS)
    generate_parser.add_argument("--output", type=Path, default=Path("./synthetic"))

    train_parser = subparsers.add_parser("train", help="Train autoencoder and clustering artifacts.")
    train_parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES_PER_CLASS)
    train_parser.add_argument("--output", type=Path, default=Path("./artifacts"))

    subparsers.add_parser("gui", help="Launch the drawing GUI.")

    args = parser.parse_args()

    if args.command == "generate":
        samples = build_dataset(args.samples)
        save_samples(samples, args.output)
    elif args.command == "train":
        samples = build_dataset(args.samples)
        artifacts = train_pipeline(samples)
        save_artifacts(artifacts, args.output)
        print(f"Training complete. Artifacts saved to {args.output}")
    else:
        run_gui()


if __name__ == "__main__":
    main()
