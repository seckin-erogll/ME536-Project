# Auto-Schematic (Intelligent Circuit Digitizer)

Auto-Schematic converts rough, hand-drawn circuit sketches into clean CAD-like symbols while detecting ambiguous or novel components. The code is segmented into the `FH_Circuit` package, similar to the referenced structure.

## Dataset Used

Training uses the **Handdrawn Circuit Schematic Components** dataset from Kaggle:
<https://www.kaggle.com/datasets/moodrammer/handdrawn-circuit-schematic-components>. The code also includes a synthetic generator for quick prototyping, but the Kaggle dataset is the expected source for real hand-drawn inputs.

Supported classes:

- Resistor
- Capacitor
- Inductor
- Source
- Ground

## Project Structure

- `FH_Circuit/config.py`: shared constants and thresholds.
- `FH_Circuit/data.py`: symbol rendering + synthetic data generation.
- `FH_Circuit/preprocess.py`: thresholding, skeletonization, dilation.
- `FH_Circuit/model.py`: convolutional autoencoder.
- `FH_Circuit/train.py`: training + PCA/k-means artifacts.
- `FH_Circuit/classify.py`: inference logic.
- `FH_Circuit/gui.py`: drawing GUI for sketch recognition.
- `main.py`: CLI entrypoint.

## Quick Start

Generate synthetic samples (optional):

```bash
python main.py generate --samples 200 --output ./synthetic
```

Train the autoencoder and save artifacts with the Kaggle dataset:

```bash
python main.py train --dataset-dir /path/to/handdrawn-circuit-schematic-components --epochs 5 --output ./artifacts
```

Train with synthetic data (optional demo):

```bash
python main.py train --use-synthetic --samples 200 --epochs 5 --output ./artifacts
```

Launch the GUI (draw symbols and classify):

```bash
python main.py gui --model-dir ./artifacts
```

Classify a saved image:

```bash
python main.py classify path/to/sketch.png --model-dir ./artifacts
```

## Dependencies

- `numpy`, `Pillow`, `scikit-image`, `scikit-learn`, `torch`, `tkinter`
