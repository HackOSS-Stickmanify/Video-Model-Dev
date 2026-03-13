"""
Pose Heatmap Generation Module

Generates Gaussian heatmaps from keypoint annotations for conditioning
pose-guided diffusion models (Stage 2).

Features:
- Gaussian heatmap generation with configurable sigma
- Limb heatmaps (PAF-style) for skeleton connections
- Batch processing with GPU acceleration
- Multiple output formats (numpy, tensor, image)

Usage:
    from tools.pose_heatmaps import PoseHeatmapGenerator
    
    # Create generator
    generator = PoseHeatmapGenerator(output_size=(64, 48), sigma=3.0)
    
    # Generate heatmaps from keypoints
    keypoints = [[x, y, vis], ...]  # 18 keypoints
    heatmaps = generator.generate(keypoints)  # (18, 64, 48)
    
    # Generate with limb heatmaps
    heatmaps = generator.generate(keypoints, include_limbs=True)  # (35, 64, 48)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


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

# OpenPose skeleton connections (for limb heatmaps)
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

NUM_KEYPOINTS = 18
NUM_LIMBS = len(SKELETON_CONNECTIONS)


class PoseHeatmapGenerator:
    """
    Generate Gaussian heatmaps from keypoint annotations.
    
    Args:
        output_size: (height, width) of output heatmaps
        sigma: Standard deviation of Gaussian kernel
        normalize: Whether to normalize heatmaps to [0, 1]
        include_background: Whether to add background channel
        device: Device for tensor operations (None for numpy-only)
    """
    
    def __init__(
        self,
        output_size: tuple[int, int] = (80, 60),  # Matches VAE latent size
        sigma: float = 2.0,
        normalize: bool = True,
        include_background: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.output_size = output_size  # (H, W)
        self.sigma = sigma
        self.normalize = normalize
        self.include_background = include_background
        self.device = device
        
        # Pre-compute coordinate grids
        self.h, self.w = output_size
        y_coords = np.arange(self.h)
        x_coords = np.arange(self.w)
        self.yy, self.xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        if device is not None:
            self.yy_t = torch.tensor(self.yy, dtype=torch.float32, device=device)
            self.xx_t = torch.tensor(self.xx, dtype=torch.float32, device=device)
    
    def generate_gaussian(
        self,
        x: float,
        y: float,
        use_tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate a single Gaussian heatmap centered at (x, y).
        
        Args:
            x: X coordinate (in output_size scale)
            y: Y coordinate (in output_size scale)
            use_tensor: Return PyTorch tensor instead of numpy array
        """
        if use_tensor and self.device is not None:
            dist_sq = (self.xx_t - x) ** 2 + (self.yy_t - y) ** 2
            heatmap = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        else:
            dist_sq = (self.xx - x) ** 2 + (self.yy - y) ** 2
            heatmap = np.exp(-dist_sq / (2 * self.sigma ** 2))
        
        return heatmap
    
    def generate_limb_heatmap(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        limb_width: float = 1.0,
        use_tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate a limb (line) heatmap between two points.
        
        Uses a capsule-shaped region (line with rounded ends).
        """
        if use_tensor and self.device is not None:
            xx, yy = self.xx_t, self.yy_t
        else:
            xx, yy = self.xx, self.yy
        
        # Vector from p1 to p2
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        
        if length < 1e-6:
            # Points are the same, return point heatmap
            return self.generate_gaussian(x1, y1, use_tensor)
        
        # Unit vector along the limb
        ux = dx / length
        uy = dy / length
        
        # Project each point onto the line and compute distance
        # t is the parameter along the line (0 at p1, 1 at p2)
        px = xx - x1
        py = yy - y1
        
        if use_tensor and self.device is not None:
            t = torch.clamp((px * ux + py * uy) / length, 0, 1)
        else:
            t = np.clip((px * ux + py * uy) / length, 0, 1)
        
        # Closest point on the line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Distance to closest point
        dist_sq = (xx - closest_x) ** 2 + (yy - closest_y) ** 2
        
        if use_tensor and self.device is not None:
            heatmap = torch.exp(-dist_sq / (2 * (self.sigma * limb_width) ** 2))
        else:
            heatmap = np.exp(-dist_sq / (2 * (self.sigma * limb_width) ** 2))
        
        return heatmap
    
    def generate(
        self,
        keypoints: Union[list, np.ndarray],
        original_size: Optional[tuple[int, int]] = None,
        include_limbs: bool = False,
        limb_width: float = 1.0,
        return_tensor: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate heatmaps for all keypoints.
        
        Args:
            keypoints: List of [x, y, visibility] for 18 keypoints
                       Coordinates are in original image space
            original_size: (width, height) of original image for coordinate scaling
                          If None, assumes keypoints are already in output_size scale
            include_limbs: Whether to include limb (skeleton) heatmaps
            limb_width: Width multiplier for limb heatmaps
            return_tensor: Return PyTorch tensor instead of numpy array
        
        Returns:
            Heatmaps of shape:
            - (18, H, W) if include_limbs=False
            - (35, H, W) if include_limbs=True (18 keypoints + 17 limbs)
            - Add 1 to first dim if include_background=True
        """
        keypoints = np.array(keypoints)
        use_tensor = return_tensor and self.device is not None
        
        # Scale keypoints to output size
        if original_size is not None:
            orig_w, orig_h = original_size
            scale_x = self.w / orig_w
            scale_y = self.h / orig_h
            keypoints = keypoints.copy()
            keypoints[:, 0] *= scale_x
            keypoints[:, 1] *= scale_y
        
        num_channels = NUM_KEYPOINTS
        if include_limbs:
            num_channels += NUM_LIMBS
        if self.include_background:
            num_channels += 1
        
        if use_tensor:
            heatmaps = torch.zeros((num_channels, self.h, self.w), 
                                   dtype=torch.float32, device=self.device)
        else:
            heatmaps = np.zeros((num_channels, self.h, self.w), dtype=np.float32)
        
        # Generate keypoint heatmaps
        channel_idx = 0
        if self.include_background:
            channel_idx = 1  # Reserve channel 0 for background
        
        for i in range(NUM_KEYPOINTS):
            x, y, vis = keypoints[i]
            if vis > 0 and 0 <= x < self.w and 0 <= y < self.h:
                heatmaps[channel_idx + i] = self.generate_gaussian(x, y, use_tensor)
        
        # Generate limb heatmaps
        if include_limbs:
            limb_offset = channel_idx + NUM_KEYPOINTS
            for i, (j1, j2) in enumerate(SKELETON_CONNECTIONS):
                x1, y1, vis1 = keypoints[j1]
                x2, y2, vis2 = keypoints[j2]
                
                if vis1 > 0 and vis2 > 0:
                    heatmaps[limb_offset + i] = self.generate_limb_heatmap(
                        x1, y1, x2, y2, limb_width, use_tensor
                    )
        
        # Generate background channel (inverse of max keypoint activation)
        if self.include_background:
            if use_tensor:
                fg_max = torch.amax(heatmaps[1:], dim=0)
                heatmaps[0] = 1.0 - fg_max
            else:
                fg_max = np.max(heatmaps[1:], axis=0)
                heatmaps[0] = 1.0 - fg_max
        
        # Normalize each channel to [0, 1]
        if self.normalize:
            if use_tensor:
                max_vals = torch.amax(heatmaps, dim=(1, 2), keepdim=True)
                max_vals = torch.clamp(max_vals, min=1e-8)
                heatmaps = heatmaps / max_vals
            else:
                max_vals = heatmaps.max(axis=(1, 2), keepdims=True)
                max_vals = np.maximum(max_vals, 1e-8)
                heatmaps = heatmaps / max_vals
        
        return heatmaps
    
    def to_visualization(
        self,
        heatmaps: Union[np.ndarray, torch.Tensor],
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.7,
        background: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert heatmaps to visualization image.
        
        Args:
            heatmaps: Heatmap tensor of shape (C, H, W)
            colormap: OpenCV colormap for visualization
            alpha: Blend factor if background is provided
            background: Optional background image (H, W) or (H, W, 3)
        
        Returns:
            RGB image of shape (H, W, 3)
        """
        if isinstance(heatmaps, torch.Tensor):
            heatmaps = heatmaps.cpu().numpy()
        
        # Sum all channels and normalize
        combined = heatmaps.sum(axis=0)
        combined = combined / (combined.max() + 1e-8)
        combined = (combined * 255).astype(np.uint8)
        
        # Apply colormap
        vis = cv2.applyColorMap(combined, colormap)
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        
        # Blend with background if provided
        if background is not None:
            if background.ndim == 2:
                background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
            if background.shape[:2] != (self.h, self.w):
                background = cv2.resize(background, (self.w, self.h))
            vis = (alpha * vis + (1 - alpha) * background).astype(np.uint8)
        
        return vis


def load_annotations(annotation_file: str) -> dict:
    """Load keypoint annotations from JSON file."""
    with open(annotation_file, 'r') as f:
        return json.load(f)


def process_annotations_to_heatmaps(
    annotation_file: str,
    image_dir: str,
    output_dir: str,
    output_size: tuple[int, int] = (80, 60),
    sigma: float = 2.0,
    include_limbs: bool = False,
    save_visualization: bool = True,
):
    """
    Process all annotations and save heatmap tensors.
    
    Args:
        annotation_file: Path to annotations.json
        image_dir: Directory containing original images (for size reference)
        output_dir: Directory to save heatmap tensors
        output_size: (height, width) of output heatmaps
        sigma: Gaussian sigma for heatmaps
        include_limbs: Include limb heatmaps
        save_visualization: Save heatmap visualizations as images
    """
    annotations = load_annotations(annotation_file)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_visualization:
        vis_path = output_path / "visualizations"
        vis_path.mkdir(exist_ok=True)
    
    generator = PoseHeatmapGenerator(
        output_size=output_size,
        sigma=sigma,
        normalize=True,
    )
    
    image_dir_path = Path(image_dir)
    
    print(f"Processing {len(annotations)} annotations...")
    print(f"Output size: {output_size[1]}×{output_size[0]} (W×H)")
    print(f"Sigma: {sigma}")
    print(f"Include limbs: {include_limbs}")
    
    for img_name, keypoints in tqdm(annotations.items()):
        # Get original image size
        img_path = image_dir_path / img_name
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                orig_h, orig_w = img.shape[:2]
                original_size = (orig_w, orig_h)
            else:
                original_size = (480, 640)  # Default
        else:
            original_size = (480, 640)  # Default
        
        # Generate heatmaps
        heatmaps = generator.generate(
            keypoints,
            original_size=original_size,
            include_limbs=include_limbs,
        )
        
        # Save as tensor
        tensor_name = Path(img_name).stem + "_heatmap.pt"
        torch.save(torch.from_numpy(heatmaps), output_path / tensor_name)
        
        # Save visualization
        if save_visualization:
            vis = generator.to_visualization(heatmaps)
            vis_name = Path(img_name).stem + "_heatmap.png"
            cv2.imwrite(str(vis_path / vis_name), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    print(f"\nDone! Saved heatmaps to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate pose heatmaps from annotations")
    
    parser.add_argument(
        "--annotations", "-a",
        default="../input_stickman_video/keypoint_annotations/annotations.json",
        help="Path to annotations.json file"
    )
    parser.add_argument(
        "--image_dir", "-i",
        default="../input_stickman_video/all_bw_images_480p",
        help="Directory containing original images"
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="../input_stickman_video/pose_heatmaps",
        help="Output directory for heatmaps"
    )
    parser.add_argument(
        "--height", "-H", type=int, default=80,
        help="Output heatmap height"
    )
    parser.add_argument(
        "--width", "-W", type=int, default=60,
        help="Output heatmap width"
    )
    parser.add_argument(
        "--sigma", "-s", type=float, default=2.0,
        help="Gaussian sigma for heatmaps"
    )
    parser.add_argument(
        "--include_limbs", action="store_true",
        help="Include limb heatmaps"
    )
    parser.add_argument(
        "--no_vis", action="store_true",
        help="Don't save visualizations"
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
    
    if not annotations_path.exists():
        print(f"Error: Annotations file not found: {annotations_path}")
        return 1
    
    process_annotations_to_heatmaps(
        annotation_file=str(annotations_path),
        image_dir=str(image_dir),
        output_dir=str(output_dir),
        output_size=(args.height, args.width),
        sigma=args.sigma,
        include_limbs=args.include_limbs,
        save_visualization=not args.no_vis,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())