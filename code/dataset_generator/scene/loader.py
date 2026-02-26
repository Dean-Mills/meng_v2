"""FBX loading and character placement"""
import bpy # type: ignore
import os
import glob
from config import FBX_DIRECTORY, NUM_CHARACTERS, CHARACTER_SPACING
from animation.looping import setup_looping_animation # type: ignore
import random

def get_fbx_files():
    """Get all FBX files from directory"""
    fbx_files = glob.glob(os.path.join(FBX_DIRECTORY, "*.fbx"))
    print(f"Found {len(fbx_files)} FBX files")
    return fbx_files

def select_random_fbx_files(num_people):
    """Randomly select N FBX files"""
    fbx_files = get_fbx_files() 
        
    if len(fbx_files) < num_people:
        print(f"⚠ WARNING: Only {len(fbx_files)} files available, need {num_people}")
        return fbx_files
    
    selected = random.sample(fbx_files, num_people)
    print(f"Randomly selected {num_people} files")
    return selected

def import_fbx_at_position(fbx_path, position, index):
    """Import FBX and position it"""
    print(f"  Importing: {os.path.basename(fbx_path)}")
    
    # Import FBX
    bpy.ops.import_scene.fbx(
        filepath=fbx_path,
        automatic_bone_orientation=True,
        use_anim=True,
    )
    
    # Find imported armature
    armature = None
    for obj in bpy.context.selected_objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break
    
    if not armature:
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE' and 'Armature' in obj.name:
                armature = obj
                break
    
    if armature:
        armature.name = f"Character_{index+1}"
        armature.location = position
        
        # Setup animation looping
        anim_length = setup_looping_animation(armature)
        
        return armature, anim_length
    
    return None, 0

def load_characters(fbx_files):
    """Load all characters into scene with random placement in a circle"""
    import random
    import math
    
    characters = []
    animation_lengths = []
    positions = []
    
    circle_radius = 3.0
    min_distance = 1.2
    
    for i, fbx_path in enumerate(fbx_files):
        max_attempts = 50
        for attempt in range(max_attempts):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, circle_radius)
            x = distance * math.cos(angle)
            y = distance * math.sin(angle)
            new_pos = (x, y, 0)
            
            valid = True
            for existing_pos in positions:
                dist = math.sqrt((x - existing_pos[0])**2 + (y - existing_pos[1])**2)
                if dist < min_distance:
                    valid = False
                    break
            
            if valid:
                positions.append(new_pos)
                break
        
        armature, anim_length = import_fbx_at_position(fbx_path, new_pos, i)
        
        if armature:
            characters.append(armature)
            animation_lengths.append(anim_length)
            print(f"    Position: ({new_pos[0]:.2f}, {new_pos[1]:.2f})")
    
    return characters, animation_lengths