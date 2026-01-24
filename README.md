# Auto-Schematic (Intelligent Circuit Digitizer)

Auto-Schematic converts rough, hand-drawn circuit sketches into clean CAD-like symbols while detecting ambiguous or novel components. The code is segmented into the `FH_Circuit` package, similar to the referenced structure.

## Dataset Used

Training uses the dataset stored in `FH_Circuit/Training_Data`. Each component class is defined by its folder name (for example, `resistor`, `cap`, `inductor`). Place your downloaded dataset inside this folder and keep the per-class subfolders.

## Project Structure

**Training pipeline files**
- `FH_Circuit/data.py`: training data loader (reads labels from folder names).
- `FH_Circuit/preprocess.py`: thresholding and morphological cleanup.
- `FH_Circuit/model.py`: convolutional autoencoder.
- `FH_Circuit/train.py`: training loop + PCA/SVM artifacts.
- `FH_Circuit/dataset.py`: PyTorch dataset wrapper used by training.
- `FH_Circuit/latent_density.py`: latent-space Mahalanobis density estimation.

**Inference/GUI files**
- `FH_Circuit/classify.py`: inference logic (loads artifacts + predicts labels).
- `FH_Circuit/gui.py`: drawing GUI for sketch recognition.

**Shared dependencies**
- `FH_Circuit/config.py`: shared constants and thresholds.
- `main.py`: CLI entrypoint.

## Quick Start

### 1) Install dependencies

```bash
pip install numpy Pillow scikit-learn torch
```

### 2) Verify training data layout

Your dataset should live in `FH_Circuit/Training_Data` with one folder per class label. Training will
create `train/` and `validation/` subfolders inside each class folder (80/20 split) if they do not
already exist:

```
FH_Circuit/Training_Data/
  resistor/
    train/
    validation/
  cap/
    train/
    validation/
  inductor/
    train/
    validation/
  ...
```

The folder name becomes the label used during training and inference.

### 3) Train the model

```bash
python main.py train --dataset-dir FH_Circuit/Training_Data --epochs 5 --output ./artifacts
```

Training now stores per-class latent centroids and covariance statistics in `latent_density.pkl`.
Inference uses Mahalanobis distance in latent space to gate novelty detection before trusting SVM
probabilities.

### 4) Run the GUI

```bash
python main.py gui --model-dir ./artifacts
```

### 5) Classify a saved image

```bash
python main.py classify path/to/sketch.png --model-dir ./artifacts
```

## Dependencies

- Training: `numpy`, `Pillow`, `scikit-learn`, `torch`, `torchvision`, `matplotlib`
- GUI: `tkinter` (bundled with most Python distributions)
