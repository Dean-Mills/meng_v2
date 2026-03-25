"""Scene setup — v2
Depth is captured per-joint as world-space distance in extractor.py and
saved in the JSON annotations. No compositor/EXR needed.
"""
import bpy       # type: ignore
import mathutils # type: ignore
from config import *


def setup_render_settings():
    """Configure render settings."""
    scene = bpy.context.scene

    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.engine        = RENDER_ENGINE
    scene.cycles.samples       = RENDER_SAMPLES

    print(f"✓ Render: {RENDER_WIDTH}x{RENDER_HEIGHT}, engine={RENDER_ENGINE}")
    return scene


def create_camera(location=CAMERA_POSITION):
    """Create camera and set it as the active scene camera."""
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.active_object
    camera.name = "Camera"

    direction = mathutils.Vector((0, 0, 1)) - mathutils.Vector(location)
    rot_quat  = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    bpy.context.scene.camera = camera
    print("✓ Camera created")
    return camera


def create_lighting():
    """Add a sun light."""
    bpy.ops.object.light_add(type='SUN', location=SUN_POSITION)
    sun = bpy.context.active_object
    sun.data.energy = SUN_ENERGY
    print("✓ Lighting created")
    return sun


def clean_scene():
    """Remove all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    print("✓ Scene cleaned")