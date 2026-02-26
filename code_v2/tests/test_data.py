"""Data pipeline tests
Run with: python test_data.py

Tests each component in isolation then verifies both sources produce
identical graph structures through the full pipeline.

Usage:
    python test_data.py --virtual_dir /path/to/virtual
    python test_data.py --virtual_dir /path/to/virtual --coco_img_dir /path/to/coco/val2017 --coco_ann_file /path/to/annotations/person_keypoints_val2017.json
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from code_v2/ root regardless of where the test is run from
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import List, Optional

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Colours per person ────────────────────────────────────────────────────────
PERSON_COLOURS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def _pass(msg: str):
    print(f"  ✓ {msg}")

def _fail(msg: str):
    print(f"  ✗ {msg}")
    raise AssertionError(msg)

def _check(condition: bool, msg: str):
    if condition:
        _pass(msg)
    else:
        _fail(msg)

def _tensor_check(condition: torch.Tensor, msg: str):
    """Unwrap a scalar tensor before passing to _check."""
    _check(bool(condition.item()), msg)


# ─────────────────────────────────────────────────────────────────────────────
# Adapter tests
# ─────────────────────────────────────────────────────────────────────────────

def test_virtual_adapter(virtual_dir: Path):
    _header("Virtual Adapter")
    from virtual_adapter import VirtualAdapter

    adapter = VirtualAdapter(data_dir=virtual_dir, min_people=1)
    _check(len(adapter) > 0, f"Loaded {len(adapter)} samples")

    sample = adapter[0]

    # Keys
    for key in ["image", "keypoints", "img_id", "num_people"]:
        _check(key in sample, f"Key '{key}' present")

    # Image shape
    img = sample["image"]
    _check(img.ndim == 3,            f"Image is 3D tensor: {img.shape}")
    _check(img.shape[0] == 3,        f"Image has 3 channels: {img.shape}")
    _check(img.dtype == torch.uint8, "Image dtype is uint8")

    # Keypoints
    kps_list = sample["keypoints"]
    _check(len(kps_list) == sample["num_people"],
           f"Keypoint list length matches num_people ({sample['num_people']})")

    for i, kps in enumerate(kps_list):
        _check(kps.shape == (17, 4),          f"Person {i} keypoints shape is [17,4]")
        _check(kps.dtype == torch.float32,    f"Person {i} keypoints dtype is float32")

        vis = kps[:, 3]
        _tensor_check(vis.min() >= 0, f"Person {i} visibility min >= 0")
        _tensor_check(vis.max() <= 2, f"Person {i} visibility max <= 2")

        # Sentinel check — v=0 joints must have x=-1, y=-1
        invalid = kps[:, 3] == 0
        if invalid.any():
            _tensor_check(
                (kps[invalid, 0] == -1.0).all(),
                f"Person {i} v=0 joints have x=-1"
            )
            _tensor_check(
                (kps[invalid, 1] == -1.0).all(),
                f"Person {i} v=0 joints have y=-1"
            )

        # Valid joints — z >= 0
        valid = kps[:, 3] > 0
        if valid.any():
            _tensor_check(
                (kps[valid, 2] >= 0).all(),
                f"Person {i} valid joints have z >= 0"
            )

    _check(isinstance(sample["img_id"], str) and len(sample["img_id"]) > 0,
           f"img_id is non-empty string: '{sample['img_id'][:8]}...'")

    # Min/max filtering
    adapter_filtered = VirtualAdapter(data_dir=virtual_dir, min_people=2, max_people=2)
    for i in range(min(5, len(adapter_filtered))):
        s = adapter_filtered[i]
        _check(s["num_people"] == 2, f"Sample {i} has exactly 2 people after filter")

    print(f"\n  Virtual adapter: all checks passed ({len(adapter)} samples)")
    return adapter


def test_coco_adapter(img_dir: Path, ann_file: Path):
    _header("COCO Adapter")
    from coco_adapter import CocoAdapter

    adapter = CocoAdapter(
        img_dir=img_dir,
        ann_file=ann_file,
        min_people=1,
        max_people=4,
    )
    _check(len(adapter) > 0, f"Loaded {len(adapter)} samples")

    sample = adapter[0]

    for key in ["image", "keypoints", "img_id", "num_people"]:
        _check(key in sample, f"Key '{key}' present")

    img = sample["image"]
    _check(img.ndim == 3 and img.shape[0] == 3, f"Image shape: {img.shape}")

    for i, kps in enumerate(sample["keypoints"]):
        _check(kps.shape == (17, 4), f"Person {i} keypoints shape [17,4]")

        invalid = kps[:, 3] == 0
        if invalid.any():
            _tensor_check(
                (kps[invalid, 0] == -1.0).all(),
                f"Person {i} v=0 joints have x=-1"
            )
            _tensor_check(
                (kps[invalid, 1] == -1.0).all(),
                f"Person {i} v=0 joints have y=-1"
            )

        valid = kps[:, 3] > 0
        if valid.any():
            _tensor_check(
                (kps[valid, 2] != 0).any(),
                f"Person {i} valid joints have non-zero MiDaS depth"
            )

    _check(isinstance(sample["img_id"], str), "img_id is string")

    print(f"\n  COCO adapter: all checks passed ({len(adapter)} samples)")
    return adapter


# ─────────────────────────────────────────────────────────────────────────────
# Dataset tests
# ─────────────────────────────────────────────────────────────────────────────

def test_dataset(adapter, target_size: int = 512):
    _header(f"Dataset (target_size={target_size})")
    from dataset import PoseDataset

    dataset = PoseDataset(adapter=adapter, target_size=target_size)
    _check(len(dataset) == len(adapter), "Dataset length matches adapter")

    sample  = dataset[0]
    raw     = adapter[0]

    img = sample["image"]
    _check(
        img.shape == (3, target_size, target_size),
        f"Image shape after transform: {img.shape}"
    )

    for i, kps in enumerate(sample["keypoints"]):
        valid = kps[:, 3] > 0
        if valid.any():
            _tensor_check(
                (kps[valid, 0] >= 0).all() and (kps[valid, 0] <= target_size).all(),
                f"Person {i} x coords within [0, {target_size}]"
            )
            _tensor_check(
                (kps[valid, 1] >= 0).all() and (kps[valid, 1] <= target_size).all(),
                f"Person {i} y coords within [0, {target_size}]"
            )

        invalid = kps[:, 3] == 0
        if invalid.any():
            _tensor_check(
                (kps[invalid, 0] == -1.0).all() and (kps[invalid, 1] == -1.0).all(),
                f"Person {i} sentinel joints unchanged after transform"
            )

        raw_kps = raw["keypoints"][i]
        _check(
            torch.allclose(kps[:, 2], raw_kps[:, 2]),
            f"Person {i} z column unchanged by transform"
        )
        _check(
            torch.allclose(kps[:, 3], raw_kps[:, 3]),
            f"Person {i} v column unchanged by transform"
        )

    print(f"\n  Dataset: all checks passed")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessor tests
# ─────────────────────────────────────────────────────────────────────────────

def test_preprocessor(dataset, target_size: int = 512):
    _header("Preprocessor")
    from dataloader import create_dataloader
    from preprocessor import PosePreprocessor

    loader       = create_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0)
    preprocessor = PosePreprocessor(image_size=target_size, k_neighbors=8)

    batch  = next(iter(loader))
    graphs = preprocessor.process_batch(batch)

    _check(len(graphs) > 0, f"Produced {len(graphs)} graphs from batch")

    for i, graph in enumerate(graphs):
        assert graph.x is not None, f"Graph {i} has no node features"
        assert graph.edge_index is not None, f"Graph {i} has no edge index"

        n  = graph.x.shape[0]
        ei = graph.edge_index

        _check(graph.x.shape[1] == 4, f"Graph {i}: node features are [N, 4]")

        _tensor_check(
            graph.x.min() >= 0.0,
            f"Graph {i}: node feature min >= 0 (got {graph.x.min():.3f})"
        )
        _tensor_check(
            graph.x.max() <= 1.0,
            f"Graph {i}: node feature max <= 1 (got {graph.x.max():.3f})"
        )

        _check(ei.shape[0] == 2, f"Graph {i}: edge_index has 2 rows")

        _tensor_check(
            ei.min() >= 0,
            f"Graph {i}: edge indices min >= 0"
        )
        _tensor_check(
            ei.max() < n,
            f"Graph {i}: edge indices max < n ({n})"
        )

        # No self-loops
        self_loops = int((ei[0] == ei[1]).sum().item())
        _check(self_loops == 0, f"Graph {i}: no self-loops")

        # Joint types in [0, 16]
        _tensor_check(
            graph.joint_types.min() >= 0,
            f"Graph {i}: joint_types min >= 0"
        )
        _tensor_check(
            graph.joint_types.max() <= 16,
            f"Graph {i}: joint_types max <= 16"
        )

        # Person labels
        unique_people = int(graph.person_labels.unique().shape[0])
        _check(
            unique_people == graph.num_people,
            f"Graph {i}: {unique_people} unique labels matches num_people={graph.num_people}"
        )

        # Node count sanity
        max_nodes = 17 * graph.num_people
        _check(n <= max_nodes, f"Graph {i}: {n} nodes <= max {max_nodes}")

    print(f"\n  Preprocessor: all checks passed ({len(graphs)} graphs)")
    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# Cross-source shape consistency
# ─────────────────────────────────────────────────────────────────────────────

def test_cross_source_consistency(virtual_graphs, coco_graphs):
    _header("Cross-source Shape Consistency")

    vg = virtual_graphs[0]
    cg = coco_graphs[0]

    assert vg.x is not None and cg.x is not None

    _check(
        vg.x.shape[1] == cg.x.shape[1],
        f"Node feature dim matches: virtual={vg.x.shape[1]}, coco={cg.x.shape[1]}"
    )
    _check(
        vg.edge_index.shape[0] == cg.edge_index.shape[0],
        "Edge index has 2 rows in both"
    )
    _check(
        vg.joint_types.dtype == cg.joint_types.dtype,
        f"joint_types dtype matches: {vg.joint_types.dtype}"
    )
    _check(
        vg.person_labels.dtype == cg.person_labels.dtype,
        f"person_labels dtype matches: {vg.person_labels.dtype}"
    )

    print(f"\n  Cross-source consistency: all checks passed")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise_sample(dataset, idx: int = 0, title: str = "sample", save_dir: Path = Path(".")):
    """Render image with keypoints overlaid, coloured by person."""
    sample   = dataset[idx]
    image    = sample["image"].permute(1, 2, 0).numpy()
    kps_list = sample["keypoints"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    ax.set_title(f"{title} — {sample['num_people']} people  |  id: {sample['img_id'][:8]}...")
    ax.axis("off")

    patches = []
    for p_idx, kps in enumerate(kps_list):
        colour = PERSON_COLOURS[p_idx % len(PERSON_COLOURS)]
        valid  = kps[:, 3] > 0

        ax.scatter(
            kps[valid, 0].numpy(), kps[valid, 1].numpy(),
            c=colour, s=40, zorder=5, edgecolors="white", linewidths=0.5,
        )

        for j1, j2 in SKELETON:
            if kps[j1, 3] > 0 and kps[j2, 3] > 0:
                ax.plot(
                    [kps[j1, 0].item(), kps[j2, 0].item()],
                    [kps[j1, 1].item(), kps[j2, 1].item()],
                    c=colour, linewidth=2, zorder=4,
                )

        patches.append(mpatches.Patch(color=colour, label=f"Person {p_idx + 1}"))

    ax.legend(handles=patches, loc="upper right")

    out = save_dir / f"vis_{title}_{idx}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    print(f"  Saved visualisation → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--virtual_dir",   type=Path, required=True)
    parser.add_argument("--coco_img_dir",  type=Path, default=None)
    parser.add_argument("--coco_ann_file", type=Path, default=None)
    parser.add_argument("--save_dir",      type=Path, default=Path("."))
    parser.add_argument("--target_size",   type=int,  default=512)
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)

    passed: List[str] = []
    failed: List[str] = []

    v_graphs: Optional[list] = None
    coco_graphs: Optional[list] = None

    # ── Virtual pipeline ──────────────────────────────────────────────────
    try:
        v_adapter = test_virtual_adapter(args.virtual_dir)
        passed.append("Virtual adapter")

        try:
            v_dataset = test_dataset(v_adapter, args.target_size)
            passed.append("Dataset (virtual)")

            try:
                v_graphs = test_preprocessor(v_dataset, args.target_size)
                passed.append("Preprocessor (virtual)")
                visualise_sample(v_dataset, idx=0, title="virtual", save_dir=args.save_dir)
            except AssertionError as e:
                failed.append(f"Preprocessor (virtual): {e}")

        except AssertionError as e:
            failed.append(f"Dataset (virtual): {e}")

    except AssertionError as e:
        failed.append(f"Virtual adapter: {e}")

    # ── COCO pipeline ─────────────────────────────────────────────────────
    if args.coco_img_dir and args.coco_ann_file:
        try:
            c_adapter = test_coco_adapter(args.coco_img_dir, args.coco_ann_file)
            passed.append("COCO adapter")

            try:
                c_dataset = test_dataset(c_adapter, args.target_size)
                passed.append("Dataset (COCO)")

                try:
                    coco_graphs = test_preprocessor(c_dataset, args.target_size)
                    passed.append("Preprocessor (COCO)")
                    visualise_sample(c_dataset, idx=0, title="coco", save_dir=args.save_dir)
                except AssertionError as e:
                    failed.append(f"Preprocessor (COCO): {e}")

            except AssertionError as e:
                failed.append(f"Dataset (COCO): {e}")

        except AssertionError as e:
            failed.append(f"COCO adapter: {e}")

    # ── Cross-source consistency ───────────────────────────────────────────
    if v_graphs is not None and coco_graphs is not None:
        try:
            test_cross_source_consistency(v_graphs, coco_graphs)
            passed.append("Cross-source consistency")
        except AssertionError as e:
            failed.append(f"Cross-source consistency: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Passed: {len(passed)}")
    for p in passed:
        print(f"    ✓ {p}")
    if failed:
        print(f"  Failed: {len(failed)}")
        for f in failed:
            print(f"    ✗ {f}")
    else:
        print(f"\n  All tests passed ✅")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()