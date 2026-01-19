"""Auto-Schematic command-line entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from FH_Circuit.classify import load_artifacts, predict_symbol
from FH_Circuit.data import load_training_dataset
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
        args.dataset_dir = _prompt_path("Dataset directory", args.dataset_dir)
        args.output = _prompt_path("Output directory", args.output)
    samples, labels = load_training_dataset(args.dataset_dir)
    label_counts = {label: 0 for label in labels}
    for sample in samples:
        label_counts[sample.label] += 1
    print("Loaded training data:")
    for label in labels:
        print(f"  - {label}: {label_counts[label]} samples")
    train_pipeline(samples, labels, output_dir=args.output, epochs=args.epochs, batch_size=args.batch_size)
    print(f"Training complete. Artifacts saved to {args.output}.")


def run_gui(args: argparse.Namespace) -> None:
    if args.prompt:
        print("Enter GUI parameters (press Enter to accept defaults).")
        args.model_dir = _prompt_path("Model directory", args.model_dir)
        args.dataset_dir = _prompt_path("Dataset directory", args.dataset_dir)
    launch_gui(args.model_dir, args.dataset_dir)


def run_predict(args: argparse.Namespace) -> None:
    if args.prompt and args.image is None and args.image_path is None:
        args.image = _prompt_file_path("Image path")
    if args.prompt:
        args.artifacts = _prompt_path("Artifacts directory", args.artifacts)
    image_path = args.image_path or args.image
    if image_path is None:
        raise ValueError("Image path is required for prediction.")
    model, _, labels, thresholds, prototypes = load_artifacts(args.artifacts, model_path=args.weights)
    image = np.array(Image.open(image_path))
    label_idx, details = predict_symbol(model, image, prototypes, thresholds)
    if label_idx is None:
        print(
            "Prediction: unknown\n"
            f"  pmax={details['pmax']:.4f} (conf_threshold={details['conf_threshold']:.4f})\n"
            f"  dmin={details['dmin']:.4f} (dist_threshold={details['dist_threshold']:.4f})"
        )
    else:
        top_indices = np.argsort(details["probs"])[-3:][::-1]
        top_probs = [(labels[idx], float(details["probs"][idx])) for idx in top_indices]
        print(f"Prediction: {labels[label_idx]}")
        print(
            f"  pmax={details['pmax']:.4f} (conf_threshold={details['conf_threshold']:.4f})\n"
            f"  dmin={details['dmin']:.4f} (dist_threshold={details['dist_threshold']:.4f})"
        )
        print("  Top-3 probs:")
        for label, prob in top_probs:
            print(f"    - {label}: {prob:.4f}")


def run_debug_graph(args: argparse.Namespace) -> None:
    if args.prompt and args.image is None:
        args.image = _prompt_file_path("Image path")
    if args.image is None:
        raise ValueError("Image path is required for debug_graph.")
    from FH_Circuit.graph_extract import extract_graph
    from FH_Circuit.preprocess import preprocess_image

    image = np.array(Image.open(args.image))
    result = preprocess_image(
        image,
        debug_dir=args.output,
        debug_prefix="debug",
    )
    extract_graph(result.cleaned, debug_dir=args.output, debug_prefix="debug")
    print(f"Debug outputs saved to {args.output}")


def run_debug_preprocess_dir(args: argparse.Namespace) -> None:
    from FH_Circuit.preprocess import preprocess_image

    input_dir = args.input_dir
    output_dir = args.output
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for image_path in sorted(input_dir.iterdir()):
        if image_path.suffix.lower() not in image_extensions:
            continue
        prefix = image_path.stem
        preprocess_image(
            np.array(Image.open(image_path)),
            debug_dir=output_dir,
            debug_prefix=prefix,
        )
    print(f"Preprocess debug outputs saved to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-Schematic pipeline.")
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser("train", help="Train supervised classifier.")
    train.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    train.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    train.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("FH_Circuit/Training_Data"),
        help="Path to the training dataset directory.",
    )
    train.add_argument("--output", type=Path, default=Path("./artifacts"), help="Output folder.")
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
    gui.add_argument("--no-prompt", action="store_false", dest="prompt", help="Disable prompts.")
    gui.set_defaults(func=run_gui, prompt=True)

    predict = subparsers.add_parser("predict", help="Predict a symbol label from an image.")
    predict.add_argument("image", type=Path, nargs="?", help="Path to the image file.")
    predict.add_argument("--image", type=Path, dest="image_path", help="Path to the image file.")
    predict.add_argument("--weights", type=Path, default=None, help="Path to model.pt weights.")
    predict.add_argument(
        "--artifacts",
        type=Path,
        default=Path("./artifacts"),
        help="Artifacts folder with thresholds/prototypes/class_map.",
    )
    predict.add_argument("--no-prompt", action="store_false", dest="prompt", help="Disable prompts.")
    predict.set_defaults(func=run_predict, prompt=True)

    debug_graph = subparsers.add_parser("debug_graph", help="Debug skeleton graph extraction.")
    debug_graph.add_argument("image", type=Path, nargs="?", help="Path to the image file.")
    debug_graph.add_argument(
        "--output",
        type=Path,
        default=Path("./artifacts/debug_graph_one"),
        help="Output directory for debug artifacts.",
    )
    debug_graph.add_argument("--no-prompt", action="store_false", dest="prompt", help="Disable prompts.")
    debug_graph.set_defaults(func=run_debug_graph, prompt=True)

    debug_preprocess = subparsers.add_parser("debug_preprocess_dir", help="Debug preprocessing on a folder.")
    debug_preprocess.add_argument(
        "--input-dir",
        type=Path,
        default=Path("FH_Circuit/Training_Data"),
        help="Input folder with images to preprocess.",
    )
    debug_preprocess.add_argument(
        "--output",
        type=Path,
        default=Path("./artifacts/debug_preprocess_dir"),
        help="Output directory for debug artifacts.",
    )
    debug_preprocess.add_argument("--no-prompt", action="store_false", dest="prompt", help="Disable prompts.")
    debug_preprocess.set_defaults(func=run_debug_preprocess_dir, prompt=False)

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
                "predict": "Predict",
                "debug_graph": "Debug Graph",
                "debug_preprocess_dir": "Debug Preprocess Dir",
            },
        )
        args = parser.parse_args([choice])
    args.func(args)


if __name__ == "__main__":
    main()
