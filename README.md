# Auto-Schematic (Intelligent Circuit Digitizer)

Auto-Schematic classifies isolated, hand-drawn circuit symbols with a supervised classifier and explicit unknown detection gates. The pipeline combines image features with a skeleton topology graph for each symbol.

## Dataset Used

Training uses the dataset stored in `FH_Circuit/Training_Data`. Each component class is defined by its folder name (for example, `resistor`, `cap`, `inductor`). Place your downloaded dataset inside this folder and keep the per-class subfolders.

## Project Structure

**Training pipeline files**
- `FH_Circuit/data.py`: training data loader (reads labels from folder names).
- `FH_Circuit/preprocess.py`: preprocessing (denoise, binarize, morphology, crop/pad/resize).
- `FH_Circuit/graph_extract.py`: skeleton topology graph + feature extraction.
- `FH_Circuit/model.py`: hybrid image + graph classifier.
- `FH_Circuit/train.py`: supervised training loop + unknown detection artifacts.
- `FH_Circuit/dataset.py`: PyTorch dataset wrapper used by training.

**Inference/GUI files**
- `FH_Circuit/classify.py`: inference logic (loads artifacts + predicts labels).
- `FH_Circuit/gui.py`: drawing GUI for sketch recognition.

**Shared dependencies**
- `FH_Circuit/config.py`: shared constants and thresholds.
- `main.py`: CLI entrypoint.

## Quick Start

### 1) Install dependencies

```bash
pip install numpy Pillow scikit-image torch
```

### 2) Verify training data layout

Your dataset should live in `FH_Circuit/Training_Data` with one folder per class label:

```
FH_Circuit/Training_Data/
  resistor/
  cap/
  inductor/
  ...
```

The folder name becomes the label used during training and inference.

### 3) Train the model

```bash
python main.py train --dataset-dir FH_Circuit/Training_Data --epochs 50 --output ./artifacts
```

### 4) Run the GUI

```bash
python main.py gui --model-dir ./artifacts
```

### 5) Classify a saved image

```bash
python main.py predict path/to/sketch.png --model-dir ./artifacts
```

### 6) Debug graph extraction

```bash
python main.py debug_graph path/to/sketch.png --output ./artifacts/debug_graph_one
```

## Dependencies

- Training: `numpy`, `Pillow`, `scikit-image`, `torch`
- GUI: `tkinter` (bundled with most Python distributions)
