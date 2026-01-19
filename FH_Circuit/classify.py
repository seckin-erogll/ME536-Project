"""Prediction utilities for supervised circuit symbol classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from FH_Circuit.config import GRAPH_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM
from FH_Circuit.graph_extract import extract_graph
from FH_Circuit.model import HybridClassifier
from FH_Circuit.preprocess import preprocess_image


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_artifacts(
    model_dir: Path,
    model_path: Optional[Path] = None,
) -> Tuple[HybridClassifier, Dict[str, int], List[str], dict, torch.Tensor]:
    checkpoint_path = model_path or (model_dir / "model.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    labels: List[str] = checkpoint["labels"]
    model = HybridClassifier(
        num_classes=len(labels),
        image_embedding_dim=IMAGE_EMBEDDING_DIM,
        graph_embedding_dim=GRAPH_EMBEDDING_DIM,
    )
    model.load_state_dict(checkpoint["state_dict"])
    with (model_dir / "class_map.json").open("r", encoding="utf-8") as file:
        class_map = json.load(file)
    with (model_dir / "thresholds.json").open("r", encoding="utf-8") as file:
        thresholds = json.load(file)
    proto_checkpoint = torch.load(model_dir / "prototypes.pt", map_location="cpu")
    prototypes = proto_checkpoint["prototypes"]
    return model, class_map["class_to_idx"], class_map["idx_to_class"], thresholds, prototypes


def predict_symbol(
    model: HybridClassifier,
    image: np.ndarray,
    prototypes: torch.Tensor,
    thresholds: dict,
) -> Tuple[int | None, dict]:
    preprocess_result = preprocess_image(image)
    graph = extract_graph(preprocess_result.cleaned)
    device = resolve_device()
    model = model.to(device)
    image_tensor = torch.from_numpy(preprocess_result.final).unsqueeze(0).unsqueeze(0).float().to(device)
    graph_tensor = torch.from_numpy(graph.graph_features).unsqueeze(0).float().to(device)
    prototypes = prototypes.to(device)
    model.eval()
    with torch.no_grad():
        logits, fused_embedding = model(image_tensor, graph_tensor)
        probs = torch.softmax(logits, dim=1)
    pmax, pred_idx = torch.max(probs, dim=1)
    pmax_value = float(pmax.item())
    pred_idx_value = int(pred_idx.item())
    emb = fused_embedding.squeeze(0).cpu().numpy()
    distances = np.linalg.norm(prototypes.cpu().numpy() - emb[None, :], axis=1)
    dmin = float(distances.min())
    conf_threshold = float(thresholds.get("conf_threshold", 0.0))
    dist_threshold = float(thresholds.get("dist_threshold", float("inf")))

    # Unknown detection uses confidence + prototype distance thresholds on fused_embedding.
    is_unknown = pmax_value < conf_threshold or dmin > dist_threshold
    details = {
        "pmax": pmax_value,
        "conf_threshold": conf_threshold,
        "dmin": dmin,
        "dist_threshold": dist_threshold,
        "probs": probs.squeeze(0).cpu().numpy().tolist(),
        "pred_idx": pred_idx_value,
    }
    label_idx = None if is_unknown else pred_idx_value
    return label_idx, details
