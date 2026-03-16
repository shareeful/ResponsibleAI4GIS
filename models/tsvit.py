import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TemporalAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, attn_weights = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=False)
        self._last_attn_weights = attn_weights.detach()
        x = self.norm(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class SpatialAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._last_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        self._last_attn_weights = attn_weights.detach()
        x = self.norm(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, h, w


class TSViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_channels: int,
        temporal_length: int,
        spatial_patch_size: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.temporal_length = temporal_length
        self.d_model = d_model

        self.patch_embed = PatchEmbedding(input_channels, spatial_patch_size, d_model)
        self.temporal_pos = nn.Embedding(temporal_length, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.temporal_layers = nn.ModuleList([
            TemporalAttention(d_model, nhead, dropout) for _ in range(num_encoder_layers // 2)
        ])
        self.spatial_layers = nn.ModuleList([
            SpatialAttention(d_model, nhead, dropout) for _ in range(num_encoder_layers // 2)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, H, W, C = x.shape

        tokens_list = []
        for t in range(T):
            tok, h_p, w_p = self.patch_embed(x[:, t])
            tokens_list.append(tok)
        tokens = torch.stack(tokens_list, dim=1)

        t_pos = self.temporal_pos(torch.arange(T, device=x.device))
        tokens = tokens + t_pos.unsqueeze(0).unsqueeze(2)

        B, T, N, D = tokens.shape
        temporal_in = tokens.mean(dim=2)

        for layer in self.temporal_layers:
            temporal_in = layer(temporal_in, key_padding_mask=temporal_mask)

        temporal_ctx = temporal_in.mean(dim=1).unsqueeze(1).expand(-1, N, -1)
        spatial_in = tokens.mean(dim=1) + temporal_ctx

        for layer in self.spatial_layers:
            spatial_in = layer(spatial_in)

        out = self.norm(spatial_in)
        logits = self.head(out)

        logits = logits.mean(dim=1)
        return logits

    def get_temporal_attention_weights(self) -> Optional[torch.Tensor]:
        if self.temporal_layers:
            return self.temporal_layers[-1]._last_attn_weights
        return None

    def get_spatial_attention_weights(self) -> Optional[torch.Tensor]:
        if self.spatial_layers:
            return self.spatial_layers[-1]._last_attn_weights
        return None

    def replace_patch_embedding(self, new_in_channels: int) -> None:
        old = self.patch_embed
        self.patch_embed = PatchEmbedding(new_in_channels, old.patch_size, self.d_model)
