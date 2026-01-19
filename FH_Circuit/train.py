"""Training utilities for Auto-Schematic."""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from FH_Circuit.data import (
    AMBIGUOUS_COARSE_GROUPS,
    Sample,
    labels_for_coarse_group,
    list_coarse_labels,
    resolve_coarse_label,
)
from FH_Circuit.dataset import SymbolDataset
from FH_Circuit.model import ConvAutoencoder, SupervisedAutoencoder
from FH_Circuit.preprocess import extract_graph_features, preprocess


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
    dataset: SymbolDataset,
    epochs: int = 5,
    batch_size: int = 32,
    latent_dim: int = 32,
    log_interval: int = 0,
    num_classes: int = 1,
    class_loss_weight: float = 0.3,
) -> SupervisedAutoencoder:
    device = resolve_device()
    print(f"Training on device: {describe_device(device)}")
    print(f"Dataset size: {len(dataset)} | Batch size: {batch_size} | Latent dim: {latent_dim}")
    model = SupervisedAutoencoder(latent_dim=latent_dim, num_classes=num_classes).to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
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
        duration = time.perf_counter() - start
        print(f"Epoch {epoch}/{epochs} complete | Avg Loss {avg_epoch_loss:.6f} | {duration:.1f}s")
    return model


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


def extract_latents(model, dataset) -> np.ndarray:
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
            recon, _ = model(tensor)
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
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    latent_scaler: StandardScaler,
    model_type: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "latent_dim": latent_dim,
            "labels": labels,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
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
    samples: List[Sample],
    labels: List[str],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    class_loss_weight: float,
) -> None:
    dataset = SymbolDataset(samples, labels)
    model = train_autoencoder(
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        num_classes=len(labels),
        class_loss_weight=class_loss_weight,
    )
    latents, sample_labels = extract_latents(model, dataset)
    latent_scaler = StandardScaler()
    latents_norm = latent_scaler.fit_transform(latents)
    graph_features = np.stack([extract_graph_features(sample.image) for sample in samples])
    feature_mean = graph_features.mean(axis=0)
    feature_std = graph_features.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    normalized_features = (graph_features - feature_mean) / feature_std
    combined = np.concatenate([latents_norm, normalized_features], axis=1)
    components = min(latent_dim, 16)
    pca, classifier = fit_pca_classifier(combined, sample_labels, components)
    save_artifacts(
        output_dir,
        model,
        pca,
        classifier,
        latent_dim,
        labels,
        feature_mean,
        feature_std,
        latent_scaler,
        model_type="supervised",
    )
    print(f"Saving reconstructions to: {output_dir / 'reconstructions'}")
    save_reconstructions(model, samples, output_dir)


def train_pipeline(
    samples: List[Sample],
    labels: List[str],
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 32,
    latent_dim: int = 32,
    class_loss_weight: float = 0.3,
) -> None:
    coarse_labels = list_coarse_labels(labels)
    coarse_samples = [
        Sample(image=sample.image, label=resolve_coarse_label(sample.label)) for sample in samples
    ]
    coarse_dir = output_dir / "coarse"
    print(f"Training coarse stage with labels: {', '.join(coarse_labels)}")
    train_stage(
        coarse_samples,
        coarse_labels,
        coarse_dir,
        epochs=epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        class_loss_weight=class_loss_weight,
    )
    for group in sorted(AMBIGUOUS_COARSE_GROUPS):
        group_labels = labels_for_coarse_group(group)
        fine_labels = [label for label in labels if label in group_labels]
        fine_samples = [sample for sample in samples if sample.label in group_labels]
        if not fine_samples:
            print(f"Skipping fine stage for '{group}': no samples found.")
            continue
        fine_dir = output_dir / "fine" / group
        print(f"Training fine stage for '{group}' with labels: {', '.join(fine_labels)}")
        train_stage(
            fine_samples,
            fine_labels,
            fine_dir,
            epochs=epochs,
            batch_size=batch_size,
            latent_dim=latent_dim,
            class_loss_weight=class_loss_weight,
        )
