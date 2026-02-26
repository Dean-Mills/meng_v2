"""Camera positioning for multi-view dataset"""
import bpy # type: ignore
import math
import mathutils # type: ignore
from config import CAMERA_DISTANCE, NUM_CAMERA_ANGLES

def calculate_characters_center(characters):
    """Calculate the center point of all characters"""
    if not characters:
        return (0, 0, 1)
    
    avg_x = sum(char.location.x for char in characters) / len(characters)
    avg_y = sum(char.location.y for char in characters) / len(characters)
    avg_z = 1.0  # Look at chest height
    
    return (avg_x, avg_y, avg_z)

def position_camera_at_angle(camera, angle_index, tilt_degrees=0.0, target=(0, 0, 1)):
    """Position camera at specific angle around circle with tilt"""
    
    angle_rad = (angle_index / NUM_CAMERA_ANGLES) * 2 * math.pi
    
    cam_x = target[0] + CAMERA_DISTANCE * math.cos(angle_rad)
    cam_y = target[1] + CAMERA_DISTANCE * math.sin(angle_rad)
    
    tilt_rad = math.radians(tilt_degrees)
    cam_z = target[2] + CAMERA_DISTANCE * math.tan(tilt_rad)
    
    camera.location = (cam_x, cam_y, cam_z)
    
    direction = mathutils.Vector(target) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    
    bpy.context.view_layer.update()