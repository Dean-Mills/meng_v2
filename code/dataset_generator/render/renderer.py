"""Frame sampling and rendering"""
import bpy # type: ignore
import os
from config import OUTPUT_DIR, SCENE_LENGTH, SAMPLE_INTERVAL, NUM_CAMERA_ANGLES,CAMERA_TILT_MIN, CAMERA_TILT_MAX, RENDER_WIDTH, RENDER_HEIGHT, OUTPUT_DIR
import random
from scene.camera import position_camera_at_angle, calculate_characters_center
import json
from keypoints.extractor import extract_keypoints_3d, project_keypoints_to_2d, calculate_bbox, calculate_distance_to_camera

def render_frame(frame_number, output_name):
    """Set frame and render"""
    bpy.context.scene.frame_set(frame_number)
    bpy.context.view_layer.update()
    
    filepath = os.path.join(OUTPUT_DIR, output_name)
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    print(f"  ✓ Rendered frame {frame_number}")

def generate_sample_frames():
    """Generate list of frames to sample"""
    return list(range(1, SCENE_LENGTH + 1, SAMPLE_INTERVAL))

def render_all_samples(sample_frames):
    """Render all sampled frames"""
    print(f"\n[RENDERING] {len(sample_frames)} frames...")
    print("="*70)
    
    for i, frame_num in enumerate(sample_frames):
        output_name = f"sample_{i+1:03d}_frame_{frame_num:04d}.png"
        render_frame(frame_num, output_name)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(sample_frames)}")

def render_multi_view(scene, camera, sample_frames, characters):
    """Render each frame from a random camera angle and tilt with annotations"""
    
    # Calculate where to look
    target = calculate_characters_center(characters)
    print(f"Camera target: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")
    
    print(f"\n[RENDERING] {len(sample_frames)} frames with random angles and tilts...")
    print("="*70)
    
    for i, frame_num in enumerate(sample_frames):
        # Set the frame
        scene.frame_set(frame_num)
        bpy.context.view_layer.update()
        
        # Pick random angle and tilt
        angle_idx = random.randint(0, NUM_CAMERA_ANGLES - 1)
        tilt = random.uniform(CAMERA_TILT_MIN, CAMERA_TILT_MAX)
        
        # Position camera
        position_camera_at_angle(camera, angle_idx, tilt, target)
        
        # Filename
        output_name = f"frame_{frame_num:04d}_angle_{angle_idx:02d}_tilt_{int(tilt):+03d}.png"
        filepath = os.path.join(OUTPUT_DIR, output_name)
        
        # Extract annotations for all characters
        annotations = []
        for char_idx, character in enumerate(characters):
            # Extract 3D keypoints
            keypoints_3d = extract_keypoints_3d(character)
            
            # Project to 2D
            keypoints_2d = project_keypoints_to_2d(keypoints_3d, scene, camera, RENDER_WIDTH, RENDER_HEIGHT)
            
            # Calculate bbox
            bbox, area = calculate_bbox(keypoints_2d)
            
            # Calculate distance from camera
            distance = calculate_distance_to_camera(character, camera)
            
            # Count visible keypoints
            num_keypoints = sum(1 for j in range(2, len(keypoints_2d), 3) if keypoints_2d[j] > 0)
            
            annotation = {
                "id": char_idx + 1,
                "image_id": frame_num,
                "category_id": 1,  # person
                "keypoints": keypoints_2d,
                "num_keypoints": num_keypoints,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "score": 1.0,
                "distance_to_camera": round(distance, 2)  # COCO extension
            }
            annotations.append(annotation)
        
        # Create COCO-format annotation
        coco_annotation = {
            "image": {
                "id": frame_num,
                "file_name": output_name,
                "width": RENDER_WIDTH,
                "height": RENDER_HEIGHT
            },
            "annotations": annotations,
            "camera_info": {
                "angle_index": angle_idx,
                "tilt_degrees": round(tilt, 2),
                "target": list(target)
            }
        }
        
        # Save annotation
        json_path = filepath.replace('.png', '.json')
        with open(json_path, 'w') as f:
            json.dump(coco_annotation, f, indent=2)
        
        # Render
        scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(sample_frames)}")
    
    print(f"✓ Rendered {len(sample_frames)} images with annotations")