import numpy as np
import torch
from torch.utils.data import Sampler
from typing import List, Iterator


class StratifiedSubgroupSampler(Sampler):
    def __init__(self, subgroup_labels: List[int], batch_size: int, replacement: bool = False):
        self.subgroup_labels = np.array(subgroup_labels)
        self.batch_size = batch_size
        self.replacement = replacement
        self.groups = np.unique(self.subgroup_labels)
        self.group_indices = {g: np.where(self.subgroup_labels == g)[0] for g in self.groups}
        self.n_per_group = max(1, batch_size // len(self.groups))

    def __iter__(self) -> Iterator[int]:
        indices = []
        for g in self.groups:
            g_idx = self.group_indices[g]
            n = min(self.n_per_group, len(g_idx)) if not self.replacement else self.n_per_group
            chosen = np.random.choice(g_idx, size=n, replace=self.replacement)
            indices.extend(chosen.tolist())
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return sum(
            min(self.n_per_group, len(v)) for v in self.group_indices.values()
        )
