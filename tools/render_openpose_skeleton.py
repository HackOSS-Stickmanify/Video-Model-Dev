"""
Render OpenPose-style 18-keypoint skeletons from annotations.json
Generates color-coded skeleton images matching ControlNet DWPose output style
"""

import cv2
import json
import numpy as np
from pathlib import Path
import argparse

# OpenPose body keypoints (18 points, indices 0-17)
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
    "left_ear"        # 17
]

# OpenPose skeleton connections
SKELETON = [
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
    (15, 17)  # left eye to left ear
]

# OpenPose color scheme for limbs (BGR format for OpenCV)
# These colors match the standard OpenPose/DWPose visualization
LIMB_COLORS = [
    (255, 0, 0),      # 0: nose to neck - blue
    (255, 85, 0),     # 1: neck to right shoulder
    (255, 170, 0),    # 2: right shoulder to right elbow
    (255, 255, 0),    # 3: right elbow to right wrist
    (170, 255, 0),    # 4: neck to left shoulder
    (85, 255, 0),     # 5: left shoulder to left elbow
    (0, 255, 0),      # 6: left elbow to left wrist
    (0, 255, 85),     # 7: neck to right hip
    (0, 255, 170),    # 8: right hip to right knee
    (0, 255, 255),    # 9: right knee to right ankle
    (0, 170, 255),    # 10: neck to left hip
    (0, 85, 255),     # 11: left hip to left knee
    (0, 0, 255),      # 12: left knee to left ankle
    (170, 0, 255),    # 13: nose to right eye
    (255, 0, 255),    # 14: right eye to right ear
    (255, 0, 170),    # 15: nose to left eye
    (255, 0, 85),     # 16: left eye to left ear
]

# OpenPose color scheme for keypoints (BGR format)
# Colors correspond to each body part
KEYPOINT_COLORS = [
    (255, 0, 0),      # 0: nose
    (255, 85, 0),     # 1: neck
    (255, 170, 0),    # 2: right_shoulder
    (255, 255, 0),    # 3: right_elbow
    (170, 255, 0),    # 4: right_wrist
    (85, 255, 0),     # 5: left_shoulder
    (0, 255, 0),      # 6: left_elbow
    (0, 255, 85),     # 7: left_wrist
    (0, 255, 170),    # 8: right_hip
    (0, 255, 255),    # 9: right_knee
    (0, 170, 255),    # 10: right_ankle
    (0, 85, 255),     # 11: left_hip
    (0, 0, 255),      # 12: left_knee
    (85, 0, 255),     # 13: left_ankle
    (170, 0, 255),    # 14: right_eye
    (255, 0, 255),    # 15: left_eye
    (255, 0, 170),    # 16: right_ear
    (255, 0, 85),     # 17: left_ear
]


def draw_openpose_skeleton(canvas, keypoints, stickwidth=4, show_keypoints=True):
    """
    Draw OpenPose-style skeleton on canvas
    
    Args:
        canvas: numpy array (H, W, 3) to draw on
        keypoints: list of [x, y, visibility] for 18 keypoints
        stickwidth: line thickness for skeleton limbs
        show_keypoints: whether to draw keypoint circles
    
    Returns:
        canvas with skeleton drawn
    """
    # Draw limbs first (so keypoints are on top)
    for limb_idx, (i, j) in enumerate(SKELETON):
        if i < len(keypoints) and j < len(keypoints):
            pt1 = keypoints[i]
            pt2 = keypoints[j]
            
            # Check if both points are visible (visibility > 0)
            if pt1[2] > 0 and pt2[2] > 0:
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                
                # Skip invalid coordinates
                if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                    continue
                
                color = LIMB_COLORS[limb_idx % len(LIMB_COLORS)]
                
                # Draw anti-aliased line using polygon for better appearance
                # This matches the DWPose/controlnet_aux style
                mX = (x1 + x2) / 2
                mY = (y1 + y2) / 2
                length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                angle = np.degrees(np.arctan2(y1 - y2, x1 - x2))
                
                polygon = cv2.ellipse2Poly(
                    (int(mX), int(mY)),
                    (int(length / 2), stickwidth),
                    int(angle),
                    0, 360, 1
                )
                cv2.fillConvexPoly(canvas, polygon, color)
    
    # Draw keypoints on top
    if show_keypoints:
        for idx, (x, y, visible) in enumerate(keypoints):
            if visible > 0 and x >= 0 and y >= 0:
                color = KEYPOINT_COLORS[idx % len(KEYPOINT_COLORS)]
                cv2.circle(canvas, (int(x), int(y)), stickwidth + 2, color, -1)
    
    return canvas


def render_skeleton_image(keypoints, width, height, background_color=(0, 0, 0), 
                          stickwidth=4, show_keypoints=True):
    """
    Create a new image with only the skeleton on a solid background
    
    Args:
        keypoints: list of [x, y, visibility] for 18 keypoints
        width: output image width
        height: output image height
        background_color: BGR tuple for background
        stickwidth: line thickness
        show_keypoints: whether to draw keypoint circles
    
    Returns:
        numpy array (H, W, 3) with skeleton
    """
    canvas = np.full((height, width, 3), background_color, dtype=np.uint8)
    return draw_openpose_skeleton(canvas, keypoints, stickwidth, show_keypoints)


def process_annotations(annotations_path, image_dir, output_dir, 
                        use_original_size=True, default_size=(512, 512),
                        background_color=(0, 0, 0), stickwidth=4):
    """
    Process all annotations and save skeleton images
    
    Args:
        annotations_path: path to annotations.json
        image_dir: directory containing original images (for size reference)
        output_dir: directory to save skeleton images
        use_original_size: if True, use original image dimensions
        default_size: (width, height) if not using original size
        background_color: BGR background color
        stickwidth: line thickness for skeleton
    """
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_dir_path = Path(image_dir) if image_dir else None
    
    processed = 0
    skipped = 0
    
    for img_name, keypoints in annotations.items():
        # Determine image size
        if use_original_size and image_dir_path:
            # Try to find original image
            orig_img_path = image_dir_path / img_name
            if orig_img_path.exists():
                orig_img = cv2.imread(str(orig_img_path))
                if orig_img is not None:
                    height, width = orig_img.shape[:2]
                else:
                    width, height = default_size
            else:
                width, height = default_size
        else:
            width, height = default_size
        
        # Render skeleton
        skeleton_img = render_skeleton_image(
            keypoints, width, height, 
            background_color=background_color,
            stickwidth=stickwidth
        )
        
        # Save skeleton image
        out_name = Path(img_name).stem + "_skeleton.png"
        out_path = output_path / out_name
        cv2.imwrite(str(out_path), skeleton_img)
        processed += 1
        
        if processed % 100 == 0:
            print(f"Processed {processed} images...")
    
    print(f"\nDone! Processed {processed} images, saved to {output_dir}")
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Render OpenPose-style skeletons from keypoint annotations"
    )
    parser.add_argument(
        "--annotations", "-a",
        default="../input_stickman_video/keypoint_annotations/annotations.json",
        help="Path to annotations.json file"
    )
    parser.add_argument(
        "--image_dir", "-i",
        default="../input_stickman_video/all_colored_images",
        help="Directory containing original images (for size reference)"
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="../input_stickman_video/skeleton_images",
        help="Output directory for skeleton images"
    )
    parser.add_argument(
        "--width", "-W", type=int, default=512,
        help="Output width (if not using original size)"
    )
    parser.add_argument(
        "--height", "-H", type=int, default=512,
        help="Output height (if not using original size)"
    )
    parser.add_argument(
        "--use_original_size", action="store_true", default=True,
        help="Use original image dimensions"
    )
    parser.add_argument(
        "--fixed_size", action="store_true",
        help="Use fixed output size instead of original"
    )
    parser.add_argument(
        "--background", "-bg", type=str, default="black",
        choices=["black", "white"],
        help="Background color"
    )
    parser.add_argument(
        "--stickwidth", "-sw", type=int, default=4,
        help="Line thickness for skeleton limbs"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    
    annotations_path = Path(args.annotations)
    if not annotations_path.is_absolute():
        annotations_path = script_dir / annotations_path
    
    image_dir = Path(args.image_dir)
    if not image_dir.is_absolute():
        image_dir = script_dir / image_dir
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    
    # Check if annotations file exists
    if not annotations_path.exists():
        print(f"Error: Annotations file not found: {annotations_path}")
        return 1
    
    background_color = (0, 0, 0) if args.background == "black" else (255, 255, 255)
    use_original = args.use_original_size and not args.fixed_size
    
    print(f"Annotations: {annotations_path}")
    print(f"Image directory: {image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Using original size: {use_original}")
    print(f"Background: {args.background}")
    print(f"Stick width: {args.stickwidth}")
    print()
    
    process_annotations(
        annotations_path=str(annotations_path),
        image_dir=str(image_dir) if image_dir.exists() else None,
        output_dir=str(output_dir),
        use_original_size=use_original,
        default_size=(args.width, args.height),
        background_color=background_color,
        stickwidth=args.stickwidth
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
