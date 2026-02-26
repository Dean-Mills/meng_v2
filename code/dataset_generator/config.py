"""Configuration settings for dataset generation"""
import os

# Paths
FBX_DIRECTORY = "/home/dean/projects/mills_ds/data/mixamo"
OUTPUT_DIR = "/home/dean/projects/mills_ds/data/virtual/two_persons_test"

# Scene setup
NUM_CHARACTERS = 2
CHARACTER_SPACING = 3.0  # meters between characters

# Render settings
RENDER_WIDTH = 512
RENDER_HEIGHT = 512
RENDER_SAMPLES = 64
RENDER_ENGINE = 'CYCLES'

# Camera settings
CAMERA_DISTANCE = 10.0
CAMERA_HEIGHT = 1.5
CAMERA_POSITION = (0, -CAMERA_DISTANCE, CAMERA_HEIGHT)

NUM_CAMERA_ANGLES = 8  # Positions around the circle
CAMERA_TILT_MIN = -15  # degrees (looking down)
CAMERA_TILT_MAX = 15   # degrees (looking up)

# Sampling settings
SCENE_LENGTH = 3000   # Total frames
SAMPLE_INTERVAL = 20  # Sample every N frames
JITTER_RANGE = 5      # Random +/- frames

# Lighting
SUN_ENERGY = 3.0
SUN_POSITION = (5, 5, 10)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)