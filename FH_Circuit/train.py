"""Training utilities for Auto-Schematic."""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM, SVC
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from FH_Circuit.data import Sample, ensure_train_val_split, load_split_datasets
from FH_Circuit.dataset import SymbolDataset
from FH_Circuit.latent_density import LatentDensityArtifacts, compute_latent_density
from FH_Circuit.model import ConvAutoencoder, SupervisedAutoencoder
from FH_Circuit.preprocess import preprocess
from FH_Circuit.config import (
    ENABLE_FLIPS,
    MAHALANOBIS_QUANTILE,
    MAHALANOBIS_REG_EPS,
    MAHALANOBIS_THRESHOLD_SCALE,
)


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


class AddGaussianNoise:
    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class RandomRotate90:
    """Rotate by k*90 degrees without interpolation blur."""

    def __init__(self, p: float = 0.75):
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) > self.p:
            return tensor
        k = int(torch.randint(0, 4, (1,)).item())
        if k == 0:
            return tensor
        return torch.rot90(tensor, k=k, dims=(-2, -1))


def build_training_transforms() -> T.Compose:
    transforms: list = [RandomRotate90(p=0.75)]
    if ENABLE_FLIPS:
        transforms.extend([T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5)])
    transforms.extend(
        [
            # NEAREST avoids blurring thin strokes after discrete rotations.
            T.RandomAffine(
                degrees=15,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                fill=0,
                interpolation=InterpolationMode.NEAREST,
            ),
            AddGaussianNoise(std=0.06),
        ]
    )
    return T.Compose(transforms)


def plot_loss_curves(history: Dict[str, List[float]], output_dir: Path) -> Path:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history["train_total"]) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_total"], label="Training loss")
    plt.plot(epochs, history["train_recon"], label="Reconstruction loss (train)")
    if history["val_total"]:
        plt.plot(epochs, history["val_total"], label="Validation loss")
    if history["val_recon"]:
        plt.plot(epochs, history["val_recon"], label="Reconstruction loss (val)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss Curves")
    plt.legend()
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "loss_curves.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def train_autoencoder(
    dataset: SymbolDataset,
    val_dataset: SymbolDataset | None = None,
    epochs: int = 5,
    batch_size: int = 32,
    latent_dim: int = 32,
    log_interval: int = 0,
    num_classes: int = 1,
    class_loss_weight: float = 0.3,
) -> Tuple[SupervisedAutoencoder, Dict[str, List[float]]]:
    device = resolve_device()
    print(f"Training on device: {describe_device(device)}")
    print(f"Dataset size: {len(dataset)} | Batch size: {batch_size} | Latent dim: {latent_dim}")
    model = SupervisedAutoencoder(latent_dim=latent_dim, num_classes=num_classes).to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    history: Dict[str, List[float]] = {
        "train_total": [],
        "train_recon": [],
        "val_total": [],
        "val_recon": [],
    }
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_recon = 0.0
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
            running_recon += recon_loss.item()
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
        avg_epoch_recon = running_recon / max(1, len(loader))
        history["train_total"].append(avg_epoch_loss)
        history["train_recon"].append(avg_epoch_recon)
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_recon_running = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    recon, _, logits = model(inputs)
                    recon_loss = recon_criterion(recon, inputs)
                    class_loss = class_criterion(logits, labels)
                    loss = recon_loss + class_loss_weight * class_loss
                    val_running += loss.item()
                    val_recon_running += recon_loss.item()
            avg_val_loss = val_running / max(1, len(val_loader))
            avg_val_recon = val_recon_running / max(1, len(val_loader))
            history["val_total"].append(avg_val_loss)
            history["val_recon"].append(avg_val_recon)
        duration = time.perf_counter() - start
        if val_loader is not None:
            print(
                "Epoch"
                f" {epoch}/{epochs} complete | Train Loss {avg_epoch_loss:.6f} |"
                f" Val Loss {avg_val_loss:.6f} | {duration:.1f}s"
            )
        else:
            print(f"Epoch {epoch}/{epochs} complete | Avg Loss {avg_epoch_loss:.6f} | {duration:.1f}s")
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


def _collect_latents(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, list[int], list[float]]:
    model.eval()
    latents: list[np.ndarray] = []
    sample_labels: list[int] = []
    recon_errors: list[float] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if len(outputs) == 2:
                recon, latent = outputs
            else:
                recon, latent, _ = outputs
            batch_errors = torch.mean((recon - inputs) ** 2, dim=(1, 2, 3)).detach().cpu().numpy()
            latents.append(latent.detach().cpu().numpy())
            sample_labels.extend([int(label) for label in labels])
            recon_errors.extend(batch_errors.tolist())
    return np.concatenate(latents, axis=0), sample_labels, recon_errors


def extract_latents(
    model: nn.Module,
    dataset: SymbolDataset,
    *,
    batch_size: int = 64,
    shuffle: bool = False,
) -> tuple[np.ndarray, list[int], list[float]]:
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return _collect_latents(model, loader, device)


def extract_augmented_latents(
    model: nn.Module,
    samples: List[Sample],
    labels: List[str],
    *,
    transform: T.Compose,
    passes: int,
    batch_size: int,
) -> tuple[np.ndarray, list[int], list[float]]:
    device = next(model.parameters()).device
    all_latents: list[np.ndarray] = []
    all_labels: list[int] = []
    all_recon_errors: list[float] = []
    for _ in range(max(1, passes)):
        dataset = SymbolDataset(samples, labels, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        latents, sample_labels, recon_errors = _collect_latents(model, loader, device)
        all_latents.append(latents)
        all_labels.extend(sample_labels)
        all_recon_errors.extend(recon_errors)
    return np.concatenate(all_latents, axis=0), all_labels, all_recon_errors


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
    latent_density: LatentDensityArtifacts,
    *,
    ocsvm: OneClassSVM | None = None,
    ocsvm_meta: dict | None = None,
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
    with (output_dir / "latent_density.pkl").open("wb") as file:
        pickle.dump(latent_density, file)
    if ocsvm is not None:
        with (output_dir / "ocsvm.pkl").open("wb") as file:
            pickle.dump(ocsvm, file)
    if ocsvm_meta is not None:
        with (output_dir / "ocsvm_meta.json").open("w", encoding="utf-8") as file:
            json.dump(ocsvm_meta, file, indent=2)


def train_stage(
    train_samples: List[Sample],
    val_samples: List[Sample],
    labels: List[str],
    output_dir: Path,
    epochs: int,
    batch_size: int,
    latent_dim: int,
    class_loss_weight: float,
    save_reconstructions_outputs: bool,
    use_ocsvm: bool,
    ocsvm_nu: float,
    ocsvm_gamma: str,
    novelty_recon_quantile: float,
    ocsvm_aug_passes: int,
) -> None:
    train_dataset = SymbolDataset(train_samples, labels, transform=build_training_transforms())
    val_dataset = SymbolDataset(val_samples, labels)
    model, history = train_autoencoder(
        train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        num_classes=len(labels),
        class_loss_weight=class_loss_weight,
    )
    loss_curve_path = plot_loss_curves(history, output_dir)
    print(f"Saved loss curves to: {loss_curve_path}")
    eval_dataset = SymbolDataset(train_samples, labels)
    base_latents, sample_labels, base_recon_errors = extract_latents(
        model,
        eval_dataset,
        batch_size=batch_size,
    )
    latents_for_scaler = base_latents
    recon_errors_for_threshold = base_recon_errors
    ocsvm = None
    ocsvm_meta = None
    if use_ocsvm:
        augmented_latents, _aug_labels, aug_recon_errors = extract_augmented_latents(
            model,
            train_samples,
            labels,
            transform=build_training_transforms(),
            passes=ocsvm_aug_passes,
            batch_size=batch_size,
        )
        latents_for_scaler = augmented_latents
        recon_errors_for_threshold = aug_recon_errors
    latent_scaler = StandardScaler()
    latent_scaler.fit(latents_for_scaler)
    latents_norm = latent_scaler.transform(base_latents)
    if use_ocsvm:
        augmented_norm = latent_scaler.transform(latents_for_scaler)
        gamma_value: str | float = ocsvm_gamma
        if isinstance(gamma_value, str) and gamma_value not in {"scale", "auto"}:
            gamma_value = float(gamma_value)
        ocsvm = OneClassSVM(kernel="rbf", nu=ocsvm_nu, gamma=gamma_value)
        ocsvm.fit(augmented_norm)
        recon_threshold = float(np.quantile(recon_errors_for_threshold, novelty_recon_quantile))
        ocsvm_meta = {
            "nu": ocsvm_nu,
            "gamma": ocsvm_gamma,
            "kernel": "rbf",
            "latent_dim": latent_dim,
            "train_samples": int(len(latents_for_scaler)),
            "augmentation_passes": int(max(1, ocsvm_aug_passes)),
            "recon_error_quantile": float(novelty_recon_quantile),
            "recon_error_threshold": recon_threshold,
        }
    latent_density = compute_latent_density(
        latents_norm,
        sample_labels,
        labels,
        reg_eps=MAHALANOBIS_REG_EPS,
        quantile=MAHALANOBIS_QUANTILE,
        threshold_scale=MAHALANOBIS_THRESHOLD_SCALE,
    )
    combined = latents_norm
    components = min(latent_dim, 16)
    pca, classifier = fit_pca_classifier(combined, sample_labels, components)
    save_artifacts(
        output_dir,
        model,
        pca,
        classifier,
        latent_dim,
        labels,
        latent_scaler,
        model_type="supervised",
        latent_density=latent_density,
        ocsvm=ocsvm,
        ocsvm_meta=ocsvm_meta,
    )
    if save_reconstructions_outputs:
        print(f"Saving reconstructions to: {output_dir / 'reconstructions'}")
        save_reconstructions(model, train_samples, output_dir)


def train_pipeline(
    train_samples: List[Sample],
    val_samples: List[Sample],
    labels: List[str],
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 32,
    latent_dim: int = 32,
    class_loss_weight: float = 0.3,
    save_reconstructions_outputs: bool = False,
    use_ocsvm: bool = True,
    ocsvm_nu: float = 0.05,
    ocsvm_gamma: str = "scale",
    novelty_recon_quantile: float = 0.99,
    ocsvm_aug_passes: int = 3,
) -> None:
    print(f"Training single-stage classifier with labels: {', '.join(labels)}")
    train_stage(
        train_samples,
        val_samples,
        labels,
        output_dir,
        epochs=epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        class_loss_weight=class_loss_weight,
        save_reconstructions_outputs=save_reconstructions_outputs,
        use_ocsvm=use_ocsvm,
        ocsvm_nu=ocsvm_nu,
        ocsvm_gamma=ocsvm_gamma,
        novelty_recon_quantile=novelty_recon_quantile,
        ocsvm_aug_passes=ocsvm_aug_passes,
    )


def _load_supervised_checkpoint(model_dir: Path) -> tuple[SupervisedAutoencoder, dict, int, list[str]]:
    checkpoint_path = model_dir / "autoencoder.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_type = checkpoint.get("model_type", "supervised")
    if model_type != "supervised":
        raise ValueError("Incremental updates require a supervised autoencoder checkpoint.")
    latent_dim = int(checkpoint["latent_dim"])
    labels = list(checkpoint.get("labels", []))
    num_classes = int(checkpoint.get("num_classes", len(labels)))
    model = SupervisedAutoencoder(latent_dim=latent_dim, num_classes=num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint, latent_dim, labels


def _unfreeze_last_encoder_linear(model: SupervisedAutoencoder) -> None:
    last_linear: nn.Linear | None = None
    for module in reversed(list(model.encoder)):
        if isinstance(module, nn.Linear):
            last_linear = module
            break
    if last_linear is None:
        return
    for param in last_linear.parameters():
        param.requires_grad = True


def incremental_update_pipeline(
    dataset_dir: Path,
    model_dir: Path,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    unfreeze_last_encoder_layer: bool = False,
    use_ocsvm: bool = True,
    ocsvm_nu: float = 0.05,
    ocsvm_gamma: str = "scale",
    novelty_recon_quantile: float = 0.99,
    ocsvm_aug_passes: int = 3,
) -> None:
    device = resolve_device()
    print(f"Incremental update on device: {describe_device(device)} | epochs={epochs}")
    old_model, checkpoint, latent_dim, old_labels = _load_supervised_checkpoint(model_dir)
    old_model = old_model.to(device)

    ensure_train_val_split(dataset_dir)
    train_samples, _val_samples, new_labels = load_split_datasets(dataset_dir)

    new_model = SupervisedAutoencoder(latent_dim=latent_dim, num_classes=len(new_labels)).to(device)
    new_model.encoder.load_state_dict(old_model.encoder.state_dict())
    new_model.decoder.load_state_dict(old_model.decoder.state_dict())

    old_label_to_index = {label: idx for idx, label in enumerate(old_labels)}
    new_label_to_index = {label: idx for idx, label in enumerate(new_labels)}
    with torch.no_grad():
        new_model.classifier_head.weight.normal_(mean=0.0, std=0.02)
        new_model.classifier_head.bias.zero_()
        for label, old_idx in old_label_to_index.items():
            if label not in new_label_to_index:
                continue
            new_idx = new_label_to_index[label]
            new_model.classifier_head.weight[new_idx].copy_(
                old_model.classifier_head.weight[old_idx].to(device)
            )
            new_model.classifier_head.bias[new_idx].copy_(old_model.classifier_head.bias[old_idx].to(device))

    for param in new_model.encoder.parameters():
        param.requires_grad = False
    for param in new_model.decoder.parameters():
        param.requires_grad = False
    for param in new_model.classifier_head.parameters():
        param.requires_grad = True
    if unfreeze_last_encoder_layer:
        _unfreeze_last_encoder_linear(new_model)

    train_dataset = SymbolDataset(train_samples, new_labels, transform=build_training_transforms())
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(
        [param for param in new_model.parameters() if param.requires_grad],
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        new_model.train()
        new_model.encoder.eval()
        new_model.decoder.eval()
        running_loss = 0.0
        for inputs, labels_idx in loader:
            inputs = inputs.to(device)
            labels_idx = labels_idx.to(device)
            _recon, _latent, logits = new_model(inputs)
            loss = criterion(logits, labels_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / max(1, len(loader))
        print(f"Epoch {epoch}/{epochs} | Loss {avg_loss:.6f}")

    eval_dataset = SymbolDataset(train_samples, new_labels)
    base_latents, sample_labels, base_recon_errors = extract_latents(
        new_model,
        eval_dataset,
        batch_size=batch_size,
    )
    latents_for_scaler = base_latents
    recon_errors_for_threshold = base_recon_errors
    ocsvm = None
    ocsvm_meta = None
    if use_ocsvm:
        augmented_latents, _aug_labels, aug_recon_errors = extract_augmented_latents(
            new_model,
            train_samples,
            new_labels,
            transform=build_training_transforms(),
            passes=ocsvm_aug_passes,
            batch_size=batch_size,
        )
        latents_for_scaler = augmented_latents
        recon_errors_for_threshold = aug_recon_errors
    latent_scaler = StandardScaler()
    latent_scaler.fit(latents_for_scaler)
    latents_norm = latent_scaler.transform(base_latents)
    if use_ocsvm:
        augmented_norm = latent_scaler.transform(latents_for_scaler)
        gamma_value: str | float = ocsvm_gamma
        if isinstance(gamma_value, str) and gamma_value not in {"scale", "auto"}:
            gamma_value = float(gamma_value)
        ocsvm = OneClassSVM(kernel="rbf", nu=ocsvm_nu, gamma=gamma_value)
        ocsvm.fit(augmented_norm)
        recon_threshold = float(np.quantile(recon_errors_for_threshold, novelty_recon_quantile))
        ocsvm_meta = {
            "nu": ocsvm_nu,
            "gamma": ocsvm_gamma,
            "kernel": "rbf",
            "latent_dim": latent_dim,
            "train_samples": int(len(latents_for_scaler)),
            "augmentation_passes": int(max(1, ocsvm_aug_passes)),
            "recon_error_quantile": float(novelty_recon_quantile),
            "recon_error_threshold": recon_threshold,
        }
    latent_density = compute_latent_density(
        latents_norm,
        sample_labels,
        new_labels,
        reg_eps=MAHALANOBIS_REG_EPS,
        quantile=MAHALANOBIS_QUANTILE,
        threshold_scale=MAHALANOBIS_THRESHOLD_SCALE,
    )
    components = min(latent_dim, 16)
    pca, classifier = fit_pca_classifier(latents_norm, sample_labels, components)
    save_artifacts(
        model_dir,
        new_model,
        pca,
        classifier,
        latent_dim,
        new_labels,
        latent_scaler,
        model_type=checkpoint.get("model_type", "supervised"),
        latent_density=latent_density,
        ocsvm=ocsvm,
        ocsvm_meta=ocsvm_meta,
    )
    print("updated artifacts saved")
