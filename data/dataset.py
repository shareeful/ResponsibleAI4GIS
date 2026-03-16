import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class GeospatialDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        num_classes: int,
        input_channels: int,
        temporal_length: int,
        patch_size: int,
        subgroup_labels: Optional[List[int]] = None,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.temporal_length = temporal_length
        self.patch_size = patch_size
        self.transform = transform

        self.samples = self._load_index()
        self.subgroup_labels = subgroup_labels if subgroup_labels is not None else [0] * len(self.samples)

    def _load_index(self) -> List[Dict]:
        index_path = self.data_dir / f"{self.split}_index.npy"
        if index_path.exists():
            return list(np.load(index_path, allow_pickle=True))
        return self._build_synthetic_index()

    def _build_synthetic_index(self) -> List[Dict]:
        np.random.seed(42)
        n = {"train": 200, "val": 40, "test": 40}.get(self.split, 40)
        samples = []
        for i in range(n):
            samples.append({
                "id": f"{self.split}_{i:05d}",
                "subgroup": np.random.randint(0, 3),
                "label": np.random.randint(0, self.num_classes),
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]
        np.random.seed(idx)

        x = torch.from_numpy(
            np.random.randn(self.temporal_length, self.patch_size, self.patch_size, self.input_channels).astype(np.float32)
        )
        x = torch.clamp(x * 0.1 + 0.5, 0.0, 1.0)

        y = torch.zeros(self.patch_size, self.patch_size, dtype=torch.long)
        y[:] = sample["label"] if isinstance(sample["label"], int) else int(sample["label"])

        subgroup = int(sample.get("subgroup", self.subgroup_labels[idx] if idx < len(self.subgroup_labels) else 0))

        if self.transform:
            x = self.transform(x)

        return x, y, subgroup
