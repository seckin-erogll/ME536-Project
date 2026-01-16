# Auto-Schematic (Intelligent Circuit Digitizer)

Auto-Schematic converts rough, hand-drawn circuit sketches into clean CAD-like symbols while detecting ambiguous or novel components. The code is segmented under the `FH_Circuit/` package, and the GUI entry point lets you draw components and classify them in real time.

## Project Structure

```
FH_Circuit/
  config.py        # constants and defaults
  data.py          # synthetic data generation
  preprocessing.py # thresholding/skeletonization/dilation
  model.py         # convolutional autoencoder
  training.py      # training + artifact save/load
  inference.py     # detection/ambiguity/novelty logic
  gui.py           # Tkinter drawing GUI
main.py            # CLI + GUI entry point
```

## Dataset Used

**Synthetic dataset generated in code.**
- Five classes: Resistor, Capacitor, Inductor, Source, Ground.
- Each sample starts as a clean symbol and is warped, rotated, noised, and thickened to imitate sketches.

No external datasets are required.

## Quick Start

Launch the GUI:

```bash
python main.py gui
```

Generate synthetic samples:

```bash
python main.py generate --samples 200 --output ./synthetic
```

Train the autoencoder and save PCA/k-means artifacts:

```bash
python main.py train --samples 200 --output ./artifacts
```

## Notes

- The GUI has **Train**, **Load**, **Classify**, and **Clear** buttons. Train builds the synthetic dataset and fits the autoencoder + PCA/k-means.
- Classification yields one of:
  - **Detected** (low reconstruction error, nearest cluster)
  - **Ambiguity** (low error, two clusters too close)
  - **Novelty** (high reconstruction error)

## Dependencies

- `numpy`, `Pillow`, `scikit-image`, `scikit-learn`, `torch`, `joblib`, `tkinter`
