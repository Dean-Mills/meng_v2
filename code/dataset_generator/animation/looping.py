"""Animation looping using NLA"""
import bpy # type: ignore

def setup_looping_animation(armature, repeat_count=1000):
    """Setup looping animation for armature using NLA"""
    
    if not armature.animation_data or not armature.animation_data.action:
        print(f"    ⚠ No animation data for {armature.name}")
        return 0
    
    action = armature.animation_data.action
    frame_start = int(action.frame_range[0])
    frame_end = int(action.frame_range[1])
    anim_length = frame_end - frame_start
    
    # Create NLA track
    if not armature.animation_data.nla_tracks:
        nla_track = armature.animation_data.nla_tracks.new()
    else:
        nla_track = armature.animation_data.nla_tracks[0]
    
    # Add action as NLA strip
    nla_strip = nla_track.strips.new(
        name=action.name,
        start=1,
        action=action
    )
    
    # Enable repeat/looping
    nla_strip.repeat = float(repeat_count)
    nla_strip.use_animated_time_cyclic = True
    
    # Clear active action so NLA takes over
    armature.animation_data.action = None
    
    print(f"    ✓ {armature.name}: {anim_length} frames (looping)")
    return anim_length