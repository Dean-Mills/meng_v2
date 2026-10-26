"""
Detector-agnostic image evidence for PG-GAT-H (PEng #3).

The 2026-10-20 gap analysis showed the remaining PG-GAT vs HigherHRNet-AE
deficit is on ordinary heatmap peaks: AE's tags are image-derived, positions
and types alone stop at ~0.975 PGA on detections. This module supplies image
evidence without touching the detector: a frozen ImageNet backbone runs on the
same 512 x 512 letterboxed image the training pipeline already loads
(dataset.PoseDataset), and PG-GAT-H samples its feature map along every
candidate bone ("is there a limb between these two joints?").

  FrozenBackbone   torchvision backbone truncated at a fixed stride, weights
                   frozen, always in eval mode (BatchNorm statistics fixed).
  sample_points    bilinear feature lookup at graph positions (x, y normalised
                   by the 512 frame, as in the preprocessor).
  attach_images    puts each batch image on its graph (graph.image).
  letterbox_image  dataset.PoseDataset._transform for a raw image, for E2E.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

BACKBONES = {
    # name: (constructor, stride, channels)
    "resnet18_s16": ("resnet18", 16, 256),
}


class FrozenBackbone(nn.Module):
    """ImageNet backbone truncated at a fixed stride; no gradients, always eval."""

    def __init__(self, name: str):
        super().__init__()
        import torchvision.models as tvm
        assert name in BACKBONES, f"unknown image backbone {name!r}; known: {list(BACKBONES)}"
        arch, self.stride, self.out_channels = BACKBONES[name]
        if arch == "resnet18":
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            self.body = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3)
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.train(False)

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: [3, H, W] or [B, 3, H, W] uint8 -> [B, C, H/stride, W/stride]."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        x = (image.float() / 255.0 - self.mean) / self.std
        return self.body(x)


def sample_points(fmap: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Args:
        fmap:   [1, C, h, w] feature map covering the whole 512 x 512 frame
        points: [..., 2] (x, y) normalised to [0, 1] by the frame size
    Returns:
        [..., C]
    """
    shape = points.shape[:-1]
    grid = (points.reshape(1, -1, 1, 2) * 2.0 - 1.0).to(fmap.dtype)
    out = F.grid_sample(fmap, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out[0, :, :, 0].t().reshape(*shape, fmap.size(1))


def attach_images(graphs: List, batch: dict) -> List:
    """graph.image = the batch image of the graph's img_id ([3, 512, 512] uint8)."""
    index = {img_id: i for i, img_id in enumerate(batch["img_id"])}
    for g in graphs:
        g.image = batch["image"][index[g.img_id]]
    return graphs


def letterbox_image(image: torch.Tensor, target: int) -> torch.Tensor:
    """dataset.PoseDataset._transform for the image alone ([3, H, W] uint8)."""
    _, H, W = image.shape
    scale = target / max(H, W)
    new_H, new_W = int(H * scale), int(W * scale)
    image = TF.resize(image, [new_H, new_W], antialias=True)
    pad_left, pad_top = (target - new_W) // 2, (target - new_H) // 2
    return TF.pad(image, [pad_left, pad_top, target - new_W - pad_left, target - new_H - pad_top], fill=0)
