"""
Tools package for keypoint annotation, training, and visualization.

Available modules:
- annotate_keypoints: Manual keypoint annotation GUI
- train_keypoint: Train keypoint regression model
- predict_keypoints: Batch keypoint prediction
- render_openpose_skeleton: Render OpenPose-style skeleton images
- pose_heatmaps: Generate pose heatmaps for diffusion conditioning
- replace_background_white: Background replacement utility
- group_similar_images: Image similarity grouping utility
"""

from pathlib import Path

# Package metadata
__version__ = "0.1.0"
__author__ = "Video-Model-Dev"

# Useful constants
TOOLS_DIR = Path(__file__).parent
PROJECT_ROOT = TOOLS_DIR.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints" / "keypoint"

# OpenPose 18-keypoint body keypoints
BODY_KEYPOINTS = [
    "nose",           # 0
    "neck",           # 1
    "right_shoulder", # 2
    "right_elbow",    # 3
    "right_wrist",    # 4
    "left_shoulder",  # 5
    "left_elbow",     # 6
    "left_wrist",     # 7
    "right_hip",      # 8
    "right_knee",     # 9
    "right_ankle",    # 10
    "left_hip",       # 11
    "left_knee",      # 12
    "left_ankle",     # 13
    "right_eye",      # 14
    "left_eye",       # 15
    "right_ear",      # 16
    "left_ear",       # 17
]

# OpenPose skeleton connections
SKELETON_CONNECTIONS = [
    (0, 1),   # nose to neck
    (1, 2),   # neck to right shoulder
    (2, 3),   # right shoulder to right elbow
    (3, 4),   # right elbow to right wrist
    (1, 5),   # neck to left shoulder
    (5, 6),   # left shoulder to left elbow
    (6, 7),   # left elbow to left wrist
    (1, 8),   # neck to right hip
    (8, 9),   # right hip to right knee
    (9, 10),  # right knee to right ankle
    (1, 11),  # neck to left hip
    (11, 12), # left hip to left knee
    (12, 13), # left knee to left ankle
    (0, 14),  # nose to right eye
    (14, 16), # right eye to right ear
    (0, 15),  # nose to left eye
    (15, 17), # left eye to left ear
]

# Left/right keypoint pairs for horizontal flip augmentation
FLIP_PAIRS = [
    (2, 5),   # right_shoulder <-> left_shoulder
    (3, 6),   # right_elbow <-> left_elbow
    (4, 7),   # right_wrist <-> left_wrist
    (8, 11),  # right_hip <-> left_hip
    (9, 12),  # right_knee <-> left_knee
    (10, 13), # right_ankle <-> left_ankle
    (14, 15), # right_eye <-> left_eye
    (16, 17), # right_ear <-> left_ear
]