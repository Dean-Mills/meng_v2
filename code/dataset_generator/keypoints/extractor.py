"""COCO keypoint extraction from Mixamo armatures"""
import bpy # type: ignore
import mathutils # type: ignore
from bpy_extras.object_utils import world_to_camera_view # type: ignore

COCO_ORDER = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

MIXAMO_BONE_MAPPING = {
    'left_shoulder': 'mixamorig:LeftShoulder',
    'right_shoulder': 'mixamorig:RightShoulder',
    'left_elbow': 'mixamorig:LeftForeArm',
    'right_elbow': 'mixamorig:RightForeArm',
    'left_wrist': 'mixamorig:LeftHand',
    'right_wrist': 'mixamorig:RightHand',
    'left_hip': 'mixamorig:LeftUpLeg',
    'right_hip': 'mixamorig:RightUpLeg',
    'left_knee': 'mixamorig:LeftLeg',
    'right_knee': 'mixamorig:RightLeg',
    'left_ankle': 'mixamorig:LeftFoot',
    'right_ankle': 'mixamorig:RightFoot',
}

def extract_keypoints_3d(armature):
    """Extract 3D keypoints from Mixamo armature"""
    
    bpy.context.view_layer.objects.active = armature
    if bpy.context.mode != 'POSE':
        bpy.ops.object.mode_set(mode='POSE')
    
    pose_bones = armature.pose.bones
    keypoints_3d = {}
    
    for keypoint_name, bone_name in MIXAMO_BONE_MAPPING.items():
        if bone_name in pose_bones:
            bone = pose_bones[bone_name]
            world_pos = armature.matrix_world @ bone.head
            keypoints_3d[keypoint_name] = world_pos
    
    if 'mixamorig:Head' in pose_bones:
        head_bone = pose_bones['mixamorig:Head']
        head_pos = armature.matrix_world @ head_bone.head
        head_tail = armature.matrix_world @ head_bone.tail
        
        head_vec = head_tail - head_pos
        head_length = head_vec.length
        
        # Estimate positions
        keypoints_3d['nose'] = head_pos + head_vec * 0.7 + mathutils.Vector((0, head_length * 0.15, 0))
        keypoints_3d['left_eye'] = head_pos + head_vec * 0.6 + mathutils.Vector((head_length * 0.15, head_length * 0.1, head_length * 0.05))
        keypoints_3d['right_eye'] = head_pos + head_vec * 0.6 + mathutils.Vector((-head_length * 0.15, head_length * 0.1, head_length * 0.05))
        keypoints_3d['left_ear'] = head_pos + head_vec * 0.5 + mathutils.Vector((head_length * 0.25, 0, 0))
        keypoints_3d['right_ear'] = head_pos + head_vec * 0.5 + mathutils.Vector((-head_length * 0.25, 0, 0))
    
    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return keypoints_3d

def project_keypoints_to_2d(keypoints_3d, scene, camera, render_width, render_height):
    """Project 3D keypoints to 2D pixel coordinates"""
    
    keypoints_array = []
    
    for keypoint_name in COCO_ORDER:
        if keypoint_name in keypoints_3d:
            world_pos = keypoints_3d[keypoint_name]
            
            # Convert to camera view
            cam_coords = world_to_camera_view(scene, camera, world_pos)
            
            # Convert to pixel coordinates
            pixel_x = cam_coords.x * render_width
            pixel_y = (1.0 - cam_coords.y) * render_height
            
            # Visibility: 2 = visible, 0 = not visible (behind camera or out of frame)
            if cam_coords.z > 0 and 0 <= cam_coords.x <= 1 and 0 <= cam_coords.y <= 1:
                visibility = 2
            else:
                visibility = 0
            
            keypoints_array.extend([pixel_x, pixel_y, visibility])
        else:
            keypoints_array.extend([0, 0, 0])
    
    return keypoints_array

def calculate_bbox(keypoints_array):
    """Calculate bounding box from visible keypoints"""
    visible_x = []
    visible_y = []
    
    for i in range(0, len(keypoints_array), 3):
        x, y, v = keypoints_array[i], keypoints_array[i+1], keypoints_array[i+2]
        if v > 0:
            visible_x.append(x)
            visible_y.append(y)
    
    if visible_x and visible_y:
        bbox_x = min(visible_x)
        bbox_y = min(visible_y)
        bbox_w = max(visible_x) - bbox_x
        bbox_h = max(visible_y) - bbox_y
        bbox = [bbox_x, bbox_y, bbox_w, bbox_h]
        area = bbox_w * bbox_h
    else:
        bbox = [0, 0, 0, 0]
        area = 0
    
    return bbox, area

def calculate_distance_to_camera(armature, camera):
    """Calculate distance from character to camera"""
    char_pos = armature.matrix_world.translation
    cam_pos = camera.matrix_world.translation
    distance = (char_pos - cam_pos).length
    return distance