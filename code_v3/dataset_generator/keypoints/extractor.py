"""COCO keypoint extraction from Mixamo armatures — v2
Changes from v1:
  - Keypoint format is now [x, y, z, v] per joint (68 values for 17 joints)
  - z is world-space distance from camera to joint in metres
  - Visibility uses all three COCO states: 0=not in frame, 1=occluded, 2=visible
  - Occlusion is detected via raycast from camera to joint position
  - calculate_distance_to_camera removed (replaced by per-joint depth)
"""
import bpy          # type: ignore
import mathutils    # type: ignore
from bpy_extras.object_utils import world_to_camera_view  # type: ignore


# ── Joint ordering ────────────────────────────────────────────────────────────

COCO_ORDER = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

MIXAMO_BONE_MAPPING = {
    'left_shoulder':  'mixamorig:LeftShoulder',
    'right_shoulder': 'mixamorig:RightShoulder',
    'left_elbow':     'mixamorig:LeftForeArm',
    'right_elbow':    'mixamorig:RightForeArm',
    'left_wrist':     'mixamorig:LeftHand',
    'right_wrist':    'mixamorig:RightHand',
    'left_hip':       'mixamorig:LeftUpLeg',
    'right_hip':      'mixamorig:RightUpLeg',
    'left_knee':      'mixamorig:LeftLeg',
    'right_knee':     'mixamorig:RightLeg',
    'left_ankle':     'mixamorig:LeftFoot',
    'right_ankle':    'mixamorig:RightFoot',
}


# ── 3-D extraction ────────────────────────────────────────────────────────────

def extract_keypoints_3d(armature):
    """
    Extract world-space 3D positions for all 17 COCO joints from a
    Mixamo armature.

    Returns
    -------
    dict[str, mathutils.Vector]  keyed by COCO joint name
    """
    bpy.context.view_layer.objects.active = armature
    if bpy.context.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')

    pose_bones = armature.pose.bones
    keypoints_3d = {}

    # Body joints — direct bone mapping
    for keypoint_name, bone_name in MIXAMO_BONE_MAPPING.items():
        if bone_name in pose_bones:
            bone = pose_bones[bone_name]
            keypoints_3d[keypoint_name] = armature.matrix_world @ bone.head

    # Head joints — estimated from head bone geometry
    if 'mixamorig:Head' in pose_bones:
        head_bone  = pose_bones['mixamorig:Head']
        head_pos   = armature.matrix_world @ head_bone.head
        head_tail  = armature.matrix_world @ head_bone.tail
        head_vec   = head_tail - head_pos
        head_len   = head_vec.length

        keypoints_3d['nose']      = head_pos + head_vec * 0.7  + mathutils.Vector((0,                head_len * 0.15, 0))
        keypoints_3d['left_eye']  = head_pos + head_vec * 0.6  + mathutils.Vector(( head_len * 0.15, head_len * 0.10, head_len * 0.05))
        keypoints_3d['right_eye'] = head_pos + head_vec * 0.6  + mathutils.Vector((-head_len * 0.15, head_len * 0.10, head_len * 0.05))
        keypoints_3d['left_ear']  = head_pos + head_vec * 0.5  + mathutils.Vector(( head_len * 0.25, 0,               0))
        keypoints_3d['right_ear'] = head_pos + head_vec * 0.5  + mathutils.Vector((-head_len * 0.25, 0,               0))

    bpy.ops.object.mode_set(mode='OBJECT')
    return keypoints_3d


# ── Occlusion ─────────────────────────────────────────────────────────────────

def _is_occluded(camera, joint_world_pos, owner_armature):
    """
    Cast a ray from the camera to the joint position.
    Returns True if the ray hits any mesh that is NOT the owning character.

    We shorten the ray slightly (factor 0.98) so we don't self-intersect
    on the target joint.
    """
    cam_pos   = camera.matrix_world.translation
    direction = joint_world_pos - cam_pos
    distance  = direction.length

    if distance < 1e-6:
        return False

    direction_norm = direction.normalized()

    # Pull the origin a tiny bit away from the camera to avoid hitting the
    # camera object itself, and shorten the target so we stop just before
    # the joint surface.
    ray_origin = cam_pos + direction_norm * 0.01
    ray_target = cam_pos + direction_norm * (distance * 0.98)
    ray_dir    = (ray_target - ray_origin).normalized()
    ray_len    = (ray_target - ray_origin).length

    result, _location, _normal, _index, hit_obj, _matrix = (
        bpy.context.scene.ray_cast(
            bpy.context.view_layer.depsgraph,
            ray_origin,
            ray_dir,
            distance=ray_len,
        )
    )

    if not result:
        return False

    # A hit on the owner's own mesh is not occlusion
    if hit_obj is not None:
        # The armature can have one or more child mesh objects
        owner_meshes = {child for child in owner_armature.children
                        if child.type == 'MESH'}
        if hit_obj in owner_meshes:
            return False

    return True


# ── 2-D projection ────────────────────────────────────────────────────────────

def project_keypoints_to_2d(keypoints_3d, scene, camera,
                             render_width, render_height,
                             owner_armature=None):
    """
    Project 3D keypoints to 2D pixel coordinates and compute per-joint depth.

    Keypoint format (per joint, 4 values):
        x          — pixel x
        y          — pixel y
        z          — world-space distance from camera to joint in metres
        visibility — 0 = outside frame / behind camera
                     1 = in frame but occluded by another character
                     2 = in frame and visible

    Returns
    -------
    list[float]  length = 17 * 4 = 68
    """
    cam_pos = camera.matrix_world.translation
    keypoints_array = []

    for keypoint_name in COCO_ORDER:
        if keypoint_name in keypoints_3d:
            world_pos = keypoints_3d[keypoint_name]

            # Project to camera-normalised coordinates
            cam_coords = world_to_camera_view(scene, camera, world_pos)

            pixel_x = cam_coords.x * render_width
            pixel_y = (1.0 - cam_coords.y) * render_height

            # World-space distance (metres)
            depth = (mathutils.Vector(world_pos) - cam_pos).length

            # Determine visibility
            in_frame = (cam_coords.z > 0
                        and 0.0 <= cam_coords.x <= 1.0
                        and 0.0 <= cam_coords.y <= 1.0)

            if not in_frame:
                visibility = 0
                pixel_x = -1.0
                pixel_y = -1.0
            elif owner_armature is not None and _is_occluded(camera, world_pos, owner_armature):
                visibility = 1
            else:
                visibility = 2

            keypoints_array.extend([pixel_x, pixel_y, depth, visibility])
        else:
            # Joint not found in armature — mark as missing
            keypoints_array.extend([-1.0, -1.0, 0.0, 0])

    return keypoints_array  # 68 values


# ── Bounding box ──────────────────────────────────────────────────────────────

def calculate_bbox(keypoints_array):
    """
    Calculate bounding box from visible keypoints.

    Expects the new [x, y, z, v] format (stride = 4).
    Only joints with visibility > 0 contribute.
    """
    visible_x = []
    visible_y = []

    for i in range(0, len(keypoints_array), 4):
        x, y, _z, v = (keypoints_array[i],     keypoints_array[i + 1],
                       keypoints_array[i + 2],  keypoints_array[i + 3])
        if v > 0:
            visible_x.append(x)
            visible_y.append(y)

    if visible_x and visible_y:
        bbox_x = min(visible_x)
        bbox_y = min(visible_y)
        bbox_w = max(visible_x) - bbox_x
        bbox_h = max(visible_y) - bbox_y
        return [bbox_x, bbox_y, bbox_w, bbox_h], bbox_w * bbox_h

    return [0.0, 0.0, 0.0, 0.0], 0.0