"""Training utilities for Auto-Schematic."""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from FH_Circuit.data import Sample
from FH_Circuit.dataset import SymbolDataset
from FH_Circuit.model import ConvAutoencoder, SupervisedAutoencoder
from FH_Circuit.preprocess import preprocess


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return device.type


def train_autoencoder(
    train_dataset: SymbolDataset,
    validation_dataset: SymbolDataset | None,
    epochs: int = 5,
    batch_size: int = 32,
    latent_dim: int = 32,
    log_interval: int = 0,
    num_classes: int = 1,
    class_loss_weight: float = 0.3,
) -> tuple[SupervisedAutoencoder, dict[str, list[float]]]:
    device = resolve_device()
    print(f"Training on device: {describe_device(device)}")
    print(f"Dataset size: {len(train_dataset)} | Batch size: {batch_size} | Latent dim: {latent_dim}")
    model = SupervisedAutoencoder(latent_dim=latent_dim, num_classes=num_classes).to(device)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = (
        DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        if validation_dataset is not None
        else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_recon_loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_recon_loss = 0.0
        start = time.perf_counter()
        for step, (inputs, labels) in enumerate(loader, start=1):
            inputs = inputs.to(device)
            labels = labels.to(device)
            recon, _, logits = model(inputs)
            recon_loss = recon_criterion(recon, inputs)
            class_loss = class_criterion(logits, labels)
            loss = recon_loss + class_loss_weight * class_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_recon_loss += recon_loss.item()
            if epoch == 1 and step == 1:
                print(
                    "Debug first batch:",
                    f"inputs={tuple(inputs.shape)}",
                    f"min={inputs.min().item():.3f}",
                    f"max={inputs.max().item():.3f}",
                    f"loss={loss.item():.6f}",
                )
            if log_interval and step % log_interval == 0:
                avg_loss = running_loss / step
                print(f"Epoch {epoch}/{epochs} | Step {step}/{len(loader)} | Loss {avg_loss:.6f}")
        avg_epoch_loss = running_loss / max(1, len(loader))
        avg_epoch_recon = running_recon_loss / max(1, len(loader))
        history["train_loss"].append(avg_epoch_loss)
        history["train_recon_loss"].append(avg_epoch_recon)
        val_loss = float("nan")
        if validation_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for inputs, labels in validation_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    recon, _, logits = model(inputs)
                    recon_loss = recon_criterion(recon, inputs)
                    class_loss = class_criterion(logits, labels)
                    loss = recon_loss + class_loss_weight * class_loss
                    val_running += loss.item()
            val_loss = val_running / max(1, len(validation_loader))
        history["val_loss"].append(val_loss)
        duration = time.perf_counter() - start
        print(
            f"Epoch {epoch}/{epochs} complete | "
            f"Train Loss {avg_epoch_loss:.6f} | "
            f"Recon Loss {avg_epoch_recon:.6f} | "
            f"Val Loss {val_loss:.6f} | "
            f"{duration:.1f}s"
        )
    return model, history


def fit_pca_classifier(
    latents: np.ndarray,
    sample_labels: List[int],
    components: int,
) -> Tuple[PCA, SVC]:
    pca = PCA(n_components=components)
    reduced = pca.fit_transform(latents)
    classifier = SVC(kernel="rbf", probability=True, C=1.0)
    classifier.fit(reduced, sample_labels)
    return pca, classifier


def extract_latents(model, dataset) -> tuple[np.ndarray, list[int]]:
    model.eval()
    latents = []
    sample_labels = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for image, label in dataset:
            image = image.to(device)
            outputs = model(image.unsqueeze(0))
            if len(outputs) == 2:
                _, latent = outputs
            else:
                _, latent, _ = outputs
            latents.append(latent.detach().cpu())
            sample_labels.append(int(label))

    return torch.cat(latents, dim=0).numpy(), sample_labels


def save_reconstructions(
    model: ConvAutoencoder,
    samples: List[Sample],
    output_dir: Path,
) -> None:
    recon_dir = output_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    label_counts: dict[str, int] = {}
    with torch.no_grad():
        for sample in samples:
            label = sample.label
            label_counts[label] = label_counts.get(label, 0) + 1
            label_dir = recon_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            processed = preprocess(sample.image)
            tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).float().to(device)
            outputs = model(tensor)
            if len(outputs) == 2:
                recon, _ = outputs
            else:
                recon, _, _ = outputs
            recon_np = recon.squeeze(0).squeeze(0).cpu().numpy()
            recon_img = (np.clip(recon_np, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(sample.image).save(label_dir / f"{label_counts[label]:04d}_input.png")
            Image.fromarray(recon_img).save(label_dir / f"{label_counts[label]:04d}_recon.png")



def save_artifacts(
    output_dir: Path,
    model: nn.Module,
    pca: PCA,
    classifier: SVC,
    latent_dim: int,
    labels: List[str],
    latent_scaler: StandardScaler,
    model_type: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "latent_dim": latent_dim,
            "labels": labels,
            "model_type": model_type,
            "num_classes": len(labels),
        },
        output_dir / "autoencoder.pt",
    )
    with (output_dir / "pca.pkl").open("wb") as file:
        pickle.dump(pca, file)
    with (output_dir / "classifier.pkl").open("wb") as file:
        pickle.dump(classifier, file)
    with (output_dir / "latent_scaler.pkl").open("wb") as file:
        pickle.dump(latent_scaler, file)


def train_stage(
    train_samples: List[Sample],
    validation_samples: List[Sample] | None,
    labels: List[str],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    class_loss_weight: float,
) -> None:
    train_transform = _build_train_transform()
    train_dataset = SymbolDataset(train_samples, labels, transform=train_transform)
    validation_dataset = (
        SymbolDataset(validation_samples, labels) if validation_samples is not None else None
    )
    model, history = train_autoencoder(
        train_dataset,
        validation_dataset,
        epochs=epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        num_classes=len(labels),
        class_loss_weight=class_loss_weight,
    )
    latents, sample_labels = extract_latents(model, train_dataset)
    latent_scaler = StandardScaler()
    latents_norm = latent_scaler.fit_transform(latents)
    components = min(latent_dim, 16)
    pca, classifier = fit_pca_classifier(latents_norm, sample_labels, components)
    save_artifacts(
        output_dir,
        model,
        pca,
        classifier,
        latent_dim,
        labels,
        latent_scaler,
        model_type="supervised",
    )
    print(f"Saving reconstructions to: {output_dir / 'reconstructions'}")
    save_reconstructions(model, train_samples, output_dir)
    _save_loss_plot(history, output_dir)


def train_pipeline(
    train_samples: List[Sample],
    validation_samples: List[Sample],
    labels: List[str],
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 32,
    latent_dim: int = 32,
    class_loss_weight: float = 0.3,
) -> None:
    print(f"Training single stage with labels: {', '.join(labels)}")
    validation_subset = validation_samples if validation_samples else None
    train_stage(
        train_samples,
        validation_subset,
        labels,
        output_dir,
        epochs=epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        class_loss_weight=class_loss_weight,
    )


def _build_train_transform(noise_std: float = 0.05) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
            transforms.Lambda(
                lambda tensor: torch.clamp(
                    tensor + torch.randn_like(tensor) * noise_std, 0.0, 1.0
                )
            ),
        ]
    )


def _save_loss_plot(history: dict[str, list[float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["train_recon_loss"], label="Reconstruction Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()
