"""Auto-Schematic command-line entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from circuit_gui import launch_circuit_gui
from FH_Circuit.classify import classify_sketch, load_artifacts
from FH_Circuit.data import ensure_train_val_split, load_split_datasets
from FH_Circuit.gui import launch_gui
from FH_Circuit.train import train_pipeline


def _prompt_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        if raw.isdigit():
            return int(raw)
        print("Please enter a valid integer.")


def _prompt_path(prompt: str, default: Path) -> Path:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    return Path(raw)


def _prompt_file_path(prompt: str, default: Path | None = None) -> Path:
    while True:
        if default is None:
            raw = input(f"{prompt}: ").strip()
        else:
            raw = input(f"{prompt} [{default}]: ").strip()
            if not raw:
                raw = str(default)
        path = Path(raw)
        if path.is_file():
            return path
        print("Please provide a valid file path.")


def _prompt_choice(prompt: str, choices: dict[str, str]) -> str:
    while True:
        options = ", ".join([f"{key}={label}" for key, label in choices.items()])
        raw = input(f"{prompt} ({options}): ").strip().lower()
        if raw in choices:
            return raw
        print("Please choose one of the listed options.")


def run_train(args: argparse.Namespace) -> None:
    if args.prompt:
        print("Enter training parameters (press Enter to accept defaults).")
        args.epochs = _prompt_int("Epochs", args.epochs)
        args.batch_size = _prompt_int("Batch size", args.batch_size)
        args.latent_dim = _prompt_int("Latent dimension", args.latent_dim)
        args.dataset_dir = _prompt_path("Dataset directory", args.dataset_dir)
        args.output = _prompt_path("Output directory", args.output)
    ensure_train_val_split(args.dataset_dir)
    train_samples, val_samples, labels = load_split_datasets(args.dataset_dir)
    label_counts = {label: 0 for label in labels}
    val_counts = {label: 0 for label in labels}
    for sample in train_samples:
        label_counts[sample.label] += 1
    for sample in val_samples:
        val_counts[sample.label] += 1
    print("Loaded training data:")
    for label in labels:
        print(
            f"  - {label}: {label_counts[label]} train samples,"
            f" {val_counts[label]} validation samples"
        )
    train_pipeline(
        train_samples,
        val_samples,
        labels,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        save_reconstructions_outputs=args.save_reconstructions,
        use_ocsvm=args.use_ocsvm,
        ocsvm_nu=args.ocsvm_nu,
        ocsvm_gamma=args.ocsvm_gamma,
        novelty_recon_quantile=args.novelty_recon_quantile,
    )
    print(f"Training complete. Artifacts saved to {args.output}.")


def run_gui(args: argparse.Namespace) -> None:
    if args.prompt:
        print("Enter GUI parameters (press Enter to accept defaults).")
        args.model_dir = _prompt_path("Model directory", args.model_dir)
        args.dataset_dir = _prompt_path("Dataset directory", args.dataset_dir)
    launch_gui(args.model_dir, args.dataset_dir, min_confidence=args.min_confidence)


def run_classify(args: argparse.Namespace) -> None:
    if args.prompt and args.image is None:
        args.image = _prompt_file_path("Image path")
    if args.prompt:
        args.model_dir = _prompt_path("Model directory", args.model_dir)
    if args.image is None:
        raise ValueError("Image path is required for classification.")
    artifacts = load_artifacts(args.model_dir)
    image = Image.open(args.image).convert("L")
    image = image.resize((64, 64), resample=Image.BILINEAR)
    sketch = np.array(image)
    result = classify_sketch(artifacts, sketch, min_confidence=args.min_confidence)
    print(result)


def run_circuit_gui(args: argparse.Namespace) -> None:
    if args.prompt:
        print("Enter circuit GUI parameters (press Enter to accept defaults).")
        args.model_dir = _prompt_path("Model directory", args.model_dir)
        args.dataset_dir = _prompt_path("Dataset directory", args.dataset_dir)
    launch_circuit_gui(args.model_dir, args.dataset_dir, min_confidence=args.min_confidence)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-Schematic pipeline.")
    subparsers = parser.add_subparsers(dest="command")

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
    train.add_argument(
        "--save-reconstructions",
        action="store_true",
        help="Save reconstruction images after training.",
    )
    train.add_argument(
        "--use-ocsvm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable One-Class SVM novelty detection.",
    )
    train.add_argument("--ocsvm-nu", type=float, default=0.05, help="One-Class SVM nu parameter.")
    train.add_argument(
        "--ocsvm-gamma",
        type=str,
        default="scale",
        help="One-Class SVM gamma parameter.",
    )
    train.add_argument(
        "--novelty-recon-quantile",
        type=float,
        default=0.99,
        help="Quantile for reconstruction-error novelty threshold.",
    )
    train.add_argument("--no-prompt", action="store_false", dest="prompt", help="Disable prompts.")
    train.set_defaults(func=run_train, prompt=True)

    gui = subparsers.add_parser("gui", help="Launch the sketch GUI.")
    gui.add_argument("--model-dir", type=Path, default=Path("./artifacts"), help="Model artifacts folder.")
    gui.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("FH_Circuit/Training_Data"),
        help="Path to the training dataset directory.",
    )
    gui.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Minimum SVC confidence before marking as novelty.",
    )
    gui.add_argument("--no-prompt", action="store_false", dest="prompt", help="Disable prompts.")
    gui.set_defaults(func=run_gui, prompt=True)

    circuit_gui = subparsers.add_parser(
        "circuit-gui",
        help="Launch the circuit segmentation + classification GUI.",
    )
    circuit_gui.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./artifacts"),
        help="Model artifacts folder.",
    )
    circuit_gui.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("FH_Circuit/Training_Data"),
        help="Path to the training dataset directory.",
    )
    circuit_gui.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Minimum SVC confidence before marking as novelty.",
    )
    circuit_gui.add_argument("--no-prompt", action="store_false", dest="prompt", help="Disable prompts.")
    circuit_gui.set_defaults(func=run_circuit_gui, prompt=True)

    classify = subparsers.add_parser("classify", help="Classify a sketch image.")
    classify.add_argument("image", type=Path, nargs="?", help="Path to the image file.")
    classify.add_argument("--model-dir", type=Path, default=Path("./artifacts"), help="Model artifacts folder.")
    classify.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Minimum SVC confidence before marking as novelty.",
    )
    classify.add_argument("--no-prompt", action="store_false", dest="prompt", help="Disable prompts.")
    classify.set_defaults(func=run_classify, prompt=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        choice = _prompt_choice(
            "Select mode",
            {
                "train": "Train",
                "gui": "GUI",
                "classify": "Classify",
                "circuit-gui": "Circuit GUI",
            },
        )
        args = parser.parse_args([choice])
    args.func(args)


if __name__ == "__main__":
    main()
