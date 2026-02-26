"""Main entry point for dataset generation"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy # type: ignore
from config import *
from scene.setup import clean_scene, setup_render_settings, create_camera, create_lighting
from scene.loader import load_characters, select_random_fbx_files
from render.renderer import generate_sample_frames, render_multi_view

def main():
    print("\n" + "="*70)
    print("MIXAMO POSE DATASET GENERATOR")
    print("="*70)
    
    fbx_files = select_random_fbx_files(NUM_CHARACTERS)
    
    if not fbx_files:
        print("✗ ERROR: No FBX files found")
        return
    
    print(f"\n[SETUP] Preparing scene with {len(fbx_files)} characters...")
    print("="*70)
    
    # Setup scene
    clean_scene()
    scene = setup_render_settings()
    camera = create_camera()
    lighting = create_lighting()
    
    # Load characters
    print(f"\n[LOADING] Importing characters...")
    print("="*70)
    characters, animation_lengths = load_characters(fbx_files)
    
    if not characters:
        print("✗ ERROR: No characters loaded")
        return
    
    print(f"\n✓ Loaded {len(characters)} characters")
    print(f"  Animation lengths: {animation_lengths}")
    
    # Set scene timeline
    scene.frame_start = 1
    scene.frame_end = SCENE_LENGTH
    print(f"✓ Scene timeline: 1 to {SCENE_LENGTH} frames")
    
    # Generate and render samples
    sample_frames = generate_sample_frames()
    print(f"✓ Will render {len(sample_frames)} frames")
    
    # Changed this line - use render_multi_view instead
    render_multi_view(scene, camera, sample_frames, characters)
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print(f"✓ Output: {OUTPUT_DIR}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()