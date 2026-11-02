"""
Detection-realistic training augmentation (PEng).

Ground-truth training graphs differ from detector output in two ways the
2026-10-19 shift diagnostic showed matter: detections miss ~21 % of the
annotated joints (HigherHRNet recall 0.79 on the matched protocol), and every
detection is fed with v = 1 (the occluded/visible flag is an annotation, not
a detector output). This module removes both differences without using any
detector, so the model stays detector-agnostic:

  * joint dropout — each joint is dropped independently with a per-graph
    probability p ~ U(0, joint_drop_max); kNN edges are rebuilt on the kept
    joints with the preprocessor's own rule;
  * force_visible — the visibility feature is set to 1 for every joint.

Applied to training graphs only (validation and every eval are unchanged).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def detection_augment(
    graph: Data,
    preprocessor,
    joint_drop_max: Optional[float] = None,
    force_visible: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Optional[Data]:
    """Return an augmented copy of `graph`, or None if fewer than 2 joints remain."""
    x = graph.x.clone()
    v_col = 3 if preprocessor.use_depth else 2
    if force_visible:
        x[:, v_col] = 1.0

    n = x.size(0)
    keep = torch.ones(n, dtype=torch.bool, device=x.device)
    if joint_drop_max:
        p = float(torch.rand(1, generator=generator)) * joint_drop_max
        keep = torch.rand(n, generator=generator).to(x.device) >= p
        if int(keep.sum()) < 2:
            return None

    x = x[keep]
    pl = graph.person_labels[keep]
    spatial = x[:, :3] if preprocessor.use_depth else x[:, :2]
    out = Data(
        x=x,
        edge_index=preprocessor._knn_edges(spatial, k=preprocessor.k_neighbors),
        joint_types=graph.joint_types[keep],
        person_labels=pl,
        num_people=int(torch.unique(pl).numel()),
    )
    if getattr(graph, "image", None) is not None:
        out.image = graph.image
    return out


def warp_image(image: torch.Tensor, s: float, t: torch.Tensor) -> torch.Tensor:
    """Zoom [3, H, W] uint8 by s about the frame centre and shift by t (frame units), zero fill.

    Same convention as image_evidence.sample_points (u in [0, 1] spans the frame edges,
    align_corners=False), so a point at u lands at s (u - 0.5) + 0.5 + t.
    """
    _, H, W = image.shape
    tx, ty = float(t[0]), float(t[1])
    theta = torch.tensor([[[1.0 / s, 0.0, -2.0 * tx / s], [0.0, 1.0 / s, -2.0 * ty / s]]])
    grid = F.affine_grid(theta, [1, 3, H, W], align_corners=False)
    out = F.grid_sample(image.unsqueeze(0).float(), grid.to(image.device), mode="bilinear",
                        padding_mode="zeros", align_corners=False)
    return out[0].round().clamp(0, 255).to(torch.uint8)


def scale_augment(
    graph: Data,
    preprocessor,
    prob: float,
    zoom_min: float,
    zoom_max: float,
    shift_max: float,
    generator: Optional[torch.Generator] = None,
) -> Optional[Data]:
    """Random zoom + shift of the joint coordinates and graph.image together (PEng exp 3).

    With probability `prob`: s ~ log-uniform(zoom_min, zoom_max) about the frame centre,
    t ~ U(-shift_max, shift_max) per axis (frame units), u' = s (u - 0.5) + 0.5 + t.
    Joints that leave the frame are dropped (a detector cannot see them) and kNN edges are
    rebuilt; returns None if fewer than 2 joints remain. Otherwise the graph is returned as is.
    """
    if float(torch.rand(1, generator=generator)) >= prob:
        return graph
    r = torch.rand(3, generator=generator)
    s = math.exp(math.log(zoom_min) + float(r[0]) * (math.log(zoom_max) - math.log(zoom_min)))
    t = (r[1:] * 2.0 - 1.0) * shift_max

    x = graph.x.clone()
    xy = (x[:, :2] - 0.5) * s + 0.5 + t.to(x.device)
    keep = ((xy >= 0.0) & (xy < 1.0)).all(dim=1)
    if int(keep.sum()) < 2:
        return None
    x[:, :2] = xy
    x = x[keep]
    pl = graph.person_labels[keep]
    spatial = x[:, :3] if preprocessor.use_depth else x[:, :2]
    out = Data(
        x=x,
        edge_index=preprocessor._knn_edges(spatial, k=preprocessor.k_neighbors),
        joint_types=graph.joint_types[keep],
        person_labels=pl,
        num_people=int(torch.unique(pl).numel()),
    )
    if getattr(graph, "img_id", None) is not None:
        out.img_id = graph.img_id
    if getattr(graph, "image", None) is not None:
        out.image = warp_image(graph.image, s, t)
    return out
