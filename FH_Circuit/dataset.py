"""Dataset wrapper for torch training."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from FH_Circuit.data import Sample
from FH_Circuit.preprocess import preprocess


class SymbolDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        labels: List[str],
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.samples = samples
        self.label_to_index: Dict[str, int] = {label: idx for idx, label in enumerate(labels)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        image = preprocess(sample.image)
        tensor = torch.from_numpy(image).unsqueeze(0).float()
        if self.transform is not None:
            tensor = self.transform(tensor)
        label = self.label_to_index[sample.label]
        return tensor, label
