"""Frame sampling and rendering — v2
Changes from v1:
  - Files named <guid>_<n_people>.png / .json — no collisions, filterable by count
  - Keypoint format updated to [x, y, z, v] per joint (68 values)
  - distance_to_camera removed (replaced by per-joint z depth)
  - owner_armature passed to project_keypoints_to_2d for occlusion raycasting
"""
import bpy     # type: ignore
import os
import json
import uuid
import random

from config import (
    OUTPUT_DIR, SCENE_LENGTH, SAMPLE_INTERVAL,
    NUM_CAMERA_ANGLES, CAMERA_TILT_MIN, CAMERA_TILT_MAX,
    RENDER_WIDTH, RENDER_HEIGHT,
)
from scene.camera import position_camera_at_angle, calculate_characters_center
from keypoints.extractor import (
    extract_keypoints_3d,
    project_keypoints_to_2d,
    calculate_bbox,
)


def generate_sample_frames():
    """Return the list of frame numbers to render."""
    return list(range(1, SCENE_LENGTH + 1, SAMPLE_INTERVAL))


def render_multi_view(scene, camera, sample_frames, characters):
    """
    Render each sampled frame from a random camera angle and tilt.

    Output files per sample:
        <guid>_<n>.png   — RGB render
        <guid>_<n>.json  — annotation with [x,y,z,v] keypoints

    where <n> is the number of people in the scene, making it trivial
    to filter a mixed folder by person count.
    """
    target     = calculate_characters_center(characters)
    num_people = len(characters)

    print(f"Camera target: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")
    print(f"\n[RENDERING] {len(sample_frames)} frames, {num_people} people per frame...")
    print("=" * 70)

    for i, frame_num in enumerate(sample_frames):

        # ── Frame + camera ────────────────────────────────────────────────
        scene.frame_set(frame_num)
        bpy.context.view_layer.update()

        angle_idx = random.randint(0, NUM_CAMERA_ANGLES - 1)
        tilt      = random.uniform(CAMERA_TILT_MIN, CAMERA_TILT_MAX)
        position_camera_at_angle(camera, angle_idx, tilt, target)

        # ── Filenames: <guid>_<n_people>.png / .json ─────────────────────
        image_id  = str(uuid.uuid4())
        file_stem = f"{image_id}_{num_people}"
        png_name  = file_stem + ".png"
        filepath  = os.path.join(OUTPUT_DIR, png_name)
        json_path = os.path.join(OUTPUT_DIR, file_stem + ".json")

        # ── Annotations ───────────────────────────────────────────────────
        annotations = []

        for char_idx, character in enumerate(characters):
            keypoints_3d = extract_keypoints_3d(character)

            keypoints_2d = project_keypoints_to_2d(
                keypoints_3d,
                scene,
                camera,
                RENDER_WIDTH,
                RENDER_HEIGHT,
                owner_armature=character,
            )

            bbox, area = calculate_bbox(keypoints_2d)

            # Stride is 4: [x, y, z, v] — visibility is at index 3, 7, 11, ...
            num_keypoints = sum(
                1 for j in range(3, len(keypoints_2d), 4)
                if keypoints_2d[j] > 0
            )

            annotations.append({
                "id":            char_idx + 1,
                "image_id":      image_id,
                "category_id":   1,
                "keypoints":     keypoints_2d,   # 68 values: [x,y,z,v] × 17
                "num_keypoints": num_keypoints,
                "bbox":          bbox,
                "area":          area,
                "iscrowd":       0,
                "score":         1.0,
            })

        # ── Sidecar JSON ──────────────────────────────────────────────────
        annotation_data = {
            "image": {
                "id":         image_id,
                "file_name":  png_name,
                "width":      RENDER_WIDTH,
                "height":     RENDER_HEIGHT,
                "num_people": num_people,
                "frame":      frame_num,     # retained for debugging
            },
            "annotations": annotations,
            "camera_info": {
                "angle_index":  angle_idx,
                "tilt_degrees": round(tilt, 2),
                "target":       list(target),
            },
        }

        with open(json_path, "w") as f:
            json.dump(annotation_data, f, indent=2)

        # ── Render ────────────────────────────────────────────────────────
        scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(sample_frames)}")

    print(f"✓ Rendered {len(sample_frames)} images with annotations")