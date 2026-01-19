"""Dataset wrapper for torch training."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from FH_Circuit.data import Sample
from FH_Circuit.graph_extract import extract_graph
from FH_Circuit.preprocess import preprocess_image


class SymbolDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        labels: List[str],
        deskew: bool = False,
        train: bool = False,
        augment_seed: int = 0,
    ):
        self.samples = samples
        self.deskew = deskew
        self.train = train
        self.augment_seed = augment_seed
        self.label_to_index: Dict[str, int] = {label: idx for idx, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]
        preprocess_result = preprocess_image(
            sample.image,
            deskew=self.deskew,
            augment=self.train,
            augment_seed=self.augment_seed + idx,
        )
        graph = extract_graph(preprocess_result.cleaned)
        image_tensor = torch.from_numpy(preprocess_result.final).unsqueeze(0)
        graph_tensor = torch.from_numpy(graph.graph_features)
        label = self.label_to_index[sample.label]
        return image_tensor, graph_tensor, label
