from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from grid_cells.ensembles import CellEnsemble, soft_cross_entropy


@dataclass
class VisionOutput:
    embedding: torch.Tensor
    place_probs: torch.Tensor
    hd_probs: torch.Tensor
    place_mask: torch.Tensor
    hd_mask: torch.Tensor


class VisionCNN(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2), nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 4 * 4, embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize CNN and FC using a Sonnet-style truncated normal."""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                fan_in = layer.weight[0].numel()
                std = fan_in ** -0.5
                nn.init.trunc_normal_(layer.weight, mean=0.0, std=std,
                                      a=-2.0 * std, b=2.0 * std)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv(images).flatten(1)
        return F.relu(self.fc(x))


class VisionModule(nn.Module):
    def __init__(self, place_ensemble: CellEnsemble, hd_ensemble: CellEnsemble,
                 embed_dim: int = 256, mask_prob: float = 0.95, mask_mode: str = "whole"):
        super().__init__()
        if mask_mode not in ("per_unit", "whole"):
            raise ValueError(f"unknown mask_mode {mask_mode!r}")
        self.mask_mode = mask_mode
        self.cnn = VisionCNN(embed_dim)
        self.place_head = nn.Linear(embed_dim, place_ensemble.n_cells)
        self.hd_head = nn.Linear(embed_dim, hd_ensemble.n_cells)
        for head in (self.place_head, self.hd_head):
            std = head.in_features ** -0.5
            nn.init.trunc_normal_(head.weight, mean=0.0, std=std,
                                  a=-2.0 * std, b=2.0 * std)
            nn.init.zeros_(head.bias)
        self.mask_prob = mask_prob

    def forward(self, images: torch.Tensor) -> VisionOutput:
        e = self.cnn(images)
        place_probs = F.softmax(self.place_head(e), dim=-1)
        hd_probs = F.softmax(self.hd_head(e), dim=-1)

        if self.mask_mode == "whole":
            keep = (torch.rand(place_probs.shape[:-1] + (1,), device=place_probs.device)
                    >= self.mask_prob).float()
            place_mask = keep.expand_as(place_probs)
            hd_mask = keep.expand_as(hd_probs)
        else:
            place_mask = (torch.rand_like(place_probs) >= self.mask_prob).float()
            hd_mask = (torch.rand_like(hd_probs) >= self.mask_prob).float()
        place_probs = place_probs * place_mask
        hd_probs = hd_probs * hd_mask

        return VisionOutput(embedding=e, place_probs=place_probs, hd_probs=hd_probs,
                             place_mask=place_mask, hd_mask=hd_mask)

    def loss(self, images: torch.Tensor, place_targets: torch.Tensor,
             hd_targets: torch.Tensor) -> torch.Tensor:
        e = self.cnn(images)
        place_logits = self.place_head(e)
        hd_logits = self.hd_head(e)
        return (soft_cross_entropy(place_logits, place_targets)
                + soft_cross_entropy(hd_logits, hd_targets)).mean()
