import torch
import numpy as np
from typing import Tuple


class GeospatialAugmentation:
    def __init__(
        self,
        flip_prob: float = 0.5,
        spectral_offset_pct: float = 0.03,
        spectral_channels: int = 12,
        rotations: Tuple[int, ...] = (0, 90, 180, 270),
    ):
        self.flip_prob = flip_prob
        self.spectral_offset_pct = spectral_offset_pct
        self.spectral_channels = spectral_channels
        self.rotations = rotations

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() < self.flip_prob:
            x = torch.flip(x, dims=[-2])
        if np.random.random() < self.flip_prob:
            x = torch.flip(x, dims=[-1])

        k = np.random.choice([r // 90 for r in self.rotations])
        if k > 0:
            x = torch.rot90(x, k=k, dims=[-2, -1])

        offset = (np.random.uniform(-1, 1) * self.spectral_offset_pct)
        if x.dim() == 4:
            x[:, :, :, :self.spectral_channels] = torch.clamp(
                x[:, :, :, :self.spectral_channels] + offset, 0.0, 1.0
            )
        return x


class ParcelboundaryAugmentation:
    def __init__(self, crop_size: int = 64, edge_noise_std: float = 0.02):
        self.crop_size = crop_size
        self.edge_noise_std = edge_noise_std

    def __call__(self, x: torch.Tensor, boundary_mask: torch.Tensor = None) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        if h > self.crop_size and w > self.crop_size:
            top = np.random.randint(0, h - self.crop_size)
            left = np.random.randint(0, w - self.crop_size)
            x = x[..., top:top + self.crop_size, left:left + self.crop_size]

        if boundary_mask is not None:
            noise = torch.randn_like(x) * self.edge_noise_std
            edge_region = boundary_mask.unsqueeze(0).unsqueeze(0).float()
            x = x + noise * edge_region

        return x
