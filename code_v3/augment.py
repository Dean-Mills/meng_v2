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

from typing import Optional

import torch
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
