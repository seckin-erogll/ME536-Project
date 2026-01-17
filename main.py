"""Auto-Schematic command-line entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from FH_Circuit.classify import classify_sketch, load_artifacts
from FH_Circuit.data import load_training_dataset
from FH_Circuit.gui import launch_gui
from FH_Circuit.train import train_pipeline


def run_train(args: argparse.Namespace) -> None:
    samples, labels = load_training_dataset(args.dataset_dir)
    label_counts = {label: 0 for label in labels}
    for sample in samples:
        label_counts[sample.label] += 1
    print("Loaded training data:")
    for label in labels:
        print(f"  - {label}: {label_counts[label]} samples")
    train_pipeline(
        samples,
        labels,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
    )
    print(f"Training complete. Artifacts saved to {args.output}.")


def run_gui(args: argparse.Namespace) -> None:
    launch_gui(args.model_dir)


def run_classify(args: argparse.Namespace) -> None:
    model, pca, kmeans, labels = load_artifacts(args.model_dir)
    image = Image.open(args.image).convert("L")
    image = image.resize((64, 64), resample=Image.BILINEAR)
    sketch = np.array(image)
    result = classify_sketch(model, pca, kmeans, sketch, labels)
    print(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-Schematic pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train autoencoder and fit PCA + k-means.")
    train.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    train.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    train.add_argument("--latent-dim", type=int, default=32, help="Latent embedding dimension.")
    train.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("FH_Circuit/Training_Data"),
        help="Path to the training dataset directory.",
    )
    train.add_argument("--output", type=Path, default=Path("./artifacts"), help="Output folder.")
    train.set_defaults(func=run_train)

    gui = subparsers.add_parser("gui", help="Launch the sketch GUI.")
    gui.add_argument("--model-dir", type=Path, default=Path("./artifacts"), help="Model artifacts folder.")
    gui.set_defaults(func=run_gui)

    classify = subparsers.add_parser("classify", help="Classify a sketch image.")
    classify.add_argument("image", type=Path, help="Path to the image file.")
    classify.add_argument("--model-dir", type=Path, default=Path("./artifacts"), help="Model artifacts folder.")
    classify.set_defaults(func=run_classify)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
