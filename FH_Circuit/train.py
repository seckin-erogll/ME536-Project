"""Training utilities for supervised circuit symbol classifier."""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from FH_Circuit.config import (
    CONFIDENCE_QUANTILE,
    DEFAULT_SEED,
    DISTANCE_QUANTILE,
    GRAPH_EMBEDDING_DIM,
    IMAGE_EMBEDDING_DIM,
)
from FH_Circuit.data import Sample
from FH_Circuit.dataset import SymbolDataset
from FH_Circuit.graph_extract import extract_graph
from FH_Circuit.model import HybridClassifier
from FH_Circuit.preprocess import preprocess_image


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def split_indices(count: int, val_ratio: float = 0.2, seed: int = DEFAULT_SEED) -> Tuple[List[int], List[int]]:
    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = int(count * (1 - val_ratio))
    return indices[:split], indices[split:]


def train_classifier(
    dataset: SymbolDataset,
    labels: List[str],
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    log_interval: int = 20,
) -> Dict[str, List[float]]:
    device = resolve_device()
    print(f"Training on device: {describe_device(device)}")
    print(f"Dataset size: {len(dataset)} | Batch size: {batch_size}")

    train_indices, val_indices = split_indices(len(dataset))
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = HybridClassifier(
        num_classes=len(labels),
        image_embedding_dim=IMAGE_EMBEDDING_DIM,
        graph_embedding_dim=GRAPH_EMBEDDING_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    metrics: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        start = time.perf_counter()

        for step, (images, graph_feats, labels_batch) in enumerate(train_loader, start=1):
            images = images.float().to(device)
            graph_feats = graph_feats.float().to(device)
            labels_batch = labels_batch.to(device)
            logits, _ = model(images, graph_feats)
            loss = criterion(logits, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == labels_batch).sum().item()
            total += images.size(0)

            if log_interval and step % log_interval == 0:
                print(
                    f"Epoch {epoch}/{epochs} | Step {step}/{len(train_loader)} | "
                    f"Loss {running_loss / total:.4f} | Acc {running_correct / total:.3f}"
                )

        train_loss = running_loss / max(1, total)
        train_acc = running_correct / max(1, total)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)

        duration = time.perf_counter() - start
        print(
            f"Epoch {epoch}/{epochs} complete | Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.3f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.3f} | {duration:.1f}s"
        )

        save_epoch_debug(dataset, output_dir / "debug_preprocess", output_dir / "debug_graph", epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "labels": labels}, output_dir / "model.pt")

    with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics


def evaluate(
    model: HybridClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for images, graph_feats, labels_batch in loader:
            images = images.float().to(device)
            graph_feats = graph_feats.float().to(device)
            labels_batch = labels_batch.to(device)
            logits, _ = model(images, graph_feats)
            loss = criterion(logits, labels_batch)
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels_batch).sum().item()
            total += images.size(0)
    return total_loss / max(1, total), total_correct / max(1, total)


def save_epoch_debug(
    dataset: SymbolDataset,
    preprocess_dir: Path,
    graph_dir: Path,
    epoch: int,
    max_samples: int = 3,
) -> None:
    for idx in range(min(max_samples, len(dataset))):
        sample = dataset.samples[idx]
        prefix = f"epoch{epoch:02d}_sample{idx:02d}"
        preprocess_result = preprocess_image(sample.image, debug_dir=preprocess_dir, debug_prefix=prefix)
        extract_graph(preprocess_result.cleaned, debug_dir=graph_dir, debug_prefix=prefix)


def compute_prototypes(
    model: HybridClassifier,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    embeddings: List[List[np.ndarray]] = [[] for _ in range(num_classes)]
    with torch.no_grad():
        for images, graph_feats, labels_batch in loader:
            images = images.float().to(device)
            graph_feats = graph_feats.float().to(device)
            logits, fused = model(images, graph_feats)
            for emb, label in zip(fused.cpu().numpy(), labels_batch.numpy()):
                embeddings[int(label)].append(emb)
    prototypes = []
    for class_embs in embeddings:
        if not class_embs:
            prototypes.append(np.zeros(fused.shape[1], dtype=np.float32))
        else:
            prototypes.append(np.mean(class_embs, axis=0))
    return torch.tensor(np.stack(prototypes), dtype=torch.float32)


def compute_thresholds(
    model: HybridClassifier,
    val_loader: DataLoader,
    prototypes: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    correct_pmax: List[float] = []
    correct_distances: List[float] = []
    with torch.no_grad():
        for images, graph_feats, labels_batch in val_loader:
            images = images.float().to(device)
            graph_feats = graph_feats.float().to(device)
            labels_batch = labels_batch.to(device)
            logits, fused = model(images, graph_feats)
            probs = torch.softmax(logits, dim=1)
            pmax, preds = torch.max(probs, dim=1)
            for idx in range(images.size(0)):
                if preds[idx] != labels_batch[idx]:
                    continue
                correct_pmax.append(float(pmax[idx].cpu().item()))
                emb = fused[idx].cpu().numpy()
                distances = np.linalg.norm(prototypes.numpy() - emb[None, :], axis=1)
                correct_distances.append(float(distances.min()))
    if not correct_pmax:
        conf_threshold = 0.0
    else:
        conf_threshold = float(np.quantile(correct_pmax, CONFIDENCE_QUANTILE))
    if not correct_distances:
        dist_threshold = float("inf")
    else:
        dist_threshold = float(np.quantile(correct_distances, DISTANCE_QUANTILE))
    return conf_threshold, dist_threshold


def train_pipeline(
    samples: List[Sample],
    labels: List[str],
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
) -> None:
    set_seed()
    dataset = SymbolDataset(samples, labels)
    metrics = train_classifier(dataset, labels, output_dir, epochs=epochs, batch_size=batch_size)

    train_indices, val_indices = split_indices(len(dataset))
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    device = resolve_device()
    checkpoint = torch.load(output_dir / "model.pt", map_location=device)
    model = HybridClassifier(
        num_classes=len(labels),
        image_embedding_dim=IMAGE_EMBEDDING_DIM,
        graph_embedding_dim=GRAPH_EMBEDDING_DIM,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    prototypes = compute_prototypes(model, train_loader, len(labels), device)
    torch.save({"prototypes": prototypes, "labels": labels}, output_dir / "prototypes.pt")

    conf_threshold, dist_threshold = compute_thresholds(model, val_loader, prototypes, device)
    with (output_dir / "thresholds.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"conf_threshold": conf_threshold, "dist_threshold": dist_threshold},
            file,
            indent=2,
        )

    with (output_dir / "class_map.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"class_to_idx": {label: idx for idx, label in enumerate(labels)}, "idx_to_class": labels},
            file,
            indent=2,
        )

    print("Saved artifacts:")
    print(f"  - model.pt")
    print(f"  - prototypes.pt")
    print(f"  - thresholds.json")
    print(f"  - class_map.json")
    print(f"  - train_metrics.json")
