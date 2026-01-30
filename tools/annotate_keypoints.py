"""
Manual Keypoint Annotation Tool for DWPose (Body Only)
Click on image to place keypoints in order.
Compatible with DWPose format (OpenPose 18 body keypoints, no hands/face)
"""

import cv2
import json
import os
import glob
import numpy as np
import shutil
from pathlib import Path
from datetime import datetime

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

# OpenPose skeleton connections for visualization
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

class KeypointAnnotator:
    def __init__(self, image_dir, output_dir):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up image directories for deletion
        self.project_dir = self.image_dir.parent
        self.all_image_dirs = [
            self.project_dir / "all_bw_images",
            self.project_dir / "all_bw_images_480p",
            self.project_dir / "all_colored_images",
            self.project_dir / "skeleton_images"
        ]
        
        # Load all images
        self.image_files = sorted(glob.glob(str(self.image_dir / "*.jpg")))
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")
        
        self.current_idx = 0
        self.keypoints = {}  # {image_name: [(x, y, visible), ...]}
        self.current_keypoint_idx = 0
        self.temp_keypoints = []
        
        # Display settings
        self.window_name = "DWPose Keypoint Annotation"
        self.display_img = None
        self.original_img = None
        self.zoom_level = 1.0
        
        # Load existing annotations
        self.load_annotations()
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
    def load_annotations(self):
        """Load existing annotation file if it exists"""
        anno_file = self.output_dir / "annotations.json"
        if anno_file.exists():
            with open(anno_file, 'r') as f:
                data = json.load(f)
                # Convert list format to dict
                for img_name, kpts in data.items():
                    self.keypoints[img_name] = [tuple(k) for k in kpts]
            print(f"Loaded existing annotations for {len(self.keypoints)} images")
    
    def save_annotations(self):
        """Save annotations to JSON file"""
        anno_file = self.output_dir / "annotations.json"
        # Convert to serializable format
        data = {k: [[float(x), float(y), int(v)] for x, y, v in v_list] 
                for k, v_list in self.keypoints.items()}
        with open(anno_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved annotations to {anno_file}")
    
    def get_current_image_name(self):
        return os.path.basename(self.image_files[self.current_idx])
    
    def load_current_image(self):
        """Load and display current image"""
        img_path = self.image_files[self.current_idx]
        self.original_img = cv2.imread(img_path)
        
        img_name = self.get_current_image_name()
        
        # Load existing keypoints for this image
        if img_name in self.keypoints:
            self.temp_keypoints = list(self.keypoints[img_name])
            self.current_keypoint_idx = len(self.temp_keypoints)
        else:
            self.temp_keypoints = []
            self.current_keypoint_idx = 0
        
        self.update_display()
    
    def update_display(self):
        """Update the display image with keypoints and skeleton"""
        self.display_img = self.original_img.copy()
        
        # Draw skeleton connections
        for i, j in SKELETON:
            if i < len(self.temp_keypoints) and j < len(self.temp_keypoints):
                pt1 = self.temp_keypoints[i]
                pt2 = self.temp_keypoints[j]
                if pt1[2] > 0 and pt2[2] > 0:  # Both visible
                    cv2.line(self.display_img, (int(pt1[0]), int(pt1[1])), 
                            (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)
        
        # Draw existing keypoints
        for idx, (x, y, visible) in enumerate(self.temp_keypoints):
            if visible > 0:
                color = (0, 255, 0) if idx < self.current_keypoint_idx - 1 else (0, 255, 255)
                cv2.circle(self.display_img, (int(x), int(y)), 5, color, -1)
                cv2.putText(self.display_img, str(idx), (int(x) + 7, int(y) - 7),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add instruction text
        img_name = self.get_current_image_name()
        progress = f"Image {self.current_idx + 1}/{len(self.image_files)}: {img_name}"
        next_kpt = BODY_KEYPOINTS[self.current_keypoint_idx] if self.current_keypoint_idx < len(BODY_KEYPOINTS) else "COMPLETE"
        
        cv2.putText(self.display_img, progress, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.display_img, f"Next: {next_kpt} ({self.current_keypoint_idx}/{len(BODY_KEYPOINTS)})", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(self.display_img, "Left-click: place | Right-click: skip | U: undo | C: copy | S: save | X: DELETE", 
                   (10, self.display_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(self.window_name, self.display_img)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add keypoint
            if self.current_keypoint_idx < len(BODY_KEYPOINTS):
                self.temp_keypoints.append((float(x), float(y), 2))  # 2 = visible
                self.current_keypoint_idx += 1
                
                # Auto-save when complete
                if self.current_keypoint_idx == len(BODY_KEYPOINTS):
                    self.save_current_image()
                
                self.update_display()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Skip keypoint (mark as not visible)
            if self.current_keypoint_idx < len(BODY_KEYPOINTS):
                self.temp_keypoints.append((0.0, 0.0, 0))  # 0 = not visible
                self.current_keypoint_idx += 1
                
                # Auto-save when complete
                if self.current_keypoint_idx == len(BODY_KEYPOINTS):
                    self.save_current_image()
                
                self.update_display()
    
    def undo_last_keypoint(self):
        """Remove the last placed keypoint"""
        if self.temp_keypoints:
            self.temp_keypoints.pop()
            self.current_keypoint_idx = len(self.temp_keypoints)
            self.update_display()
    
    def copy_from_previous(self):
        """Copy keypoints from previous image (useful for video sequences)"""
        if self.current_idx > 0:
            prev_img_name = os.path.basename(self.image_files[self.current_idx - 1])
            if prev_img_name in self.keypoints:
                self.temp_keypoints = list(self.keypoints[prev_img_name])
                self.current_keypoint_idx = len(self.temp_keypoints)
                self.update_display()
                print(f"Copied keypoints from {prev_img_name}")
    
    def save_current_image(self):
        """Save current image annotations"""
        img_name = self.get_current_image_name()
        # Ensure we have all 17 keypoints
        while len(self.temp_keypoints) < len(BODY_KEYPOINTS):
            self.temp_keypoints.append((0.0, 0.0, 0))
        
        self.keypoints[img_name] = self.temp_keypoints[:len(BODY_KEYPOINTS)]
        self.save_annotations()
        print(f"Saved annotation for {img_name}")
    
    def delete_current_image(self):
        """Delete current image from all directories and remove annotation"""
        img_name = self.get_current_image_name()
        
        print(f"\n{'='*60}")
        print(f"DELETE: {img_name}")
        print(f"{'='*60}")
        
        deleted_count = 0
        
        # Delete from all image directories
        for img_dir in self.all_image_dirs:
            img_path = img_dir / img_name
            if img_path.exists():
                try:
                    img_path.unlink()
                    print(f"  ✓ Deleted from {img_dir.name}/")
                    deleted_count += 1
                except Exception as e:
                    print(f"  ✗ Error deleting from {img_dir.name}/: {e}")
        
        # Remove from annotation
        if img_name in self.keypoints:
            # Backup annotations before deletion
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.output_dir / f"annotations_backup_{timestamp}.json"
            anno_file = self.output_dir / "annotations.json"
            if anno_file.exists():
                shutil.copy2(anno_file, backup_file)
            
            del self.keypoints[img_name]
            self.save_annotations()
            print(f"  ✓ Removed annotation")
            print(f"  ✓ Created backup: {backup_file.name}")
        
        # Remove from image file list
        current_path = self.image_files[self.current_idx]
        self.image_files.pop(self.current_idx)
        
        print(f"\nDeleted {deleted_count} file(s). {len(self.image_files)} images remaining.")
        print(f"{'='*60}\n")
        
        # Move to next image or previous if at end
        if len(self.image_files) == 0:
            print("No more images to annotate!")
            cv2.destroyAllWindows()
            exit(0)
        elif self.current_idx >= len(self.image_files):
            self.current_idx = len(self.image_files) - 1
        
        self.load_current_image()
    
    def next_image(self):
        """Move to next image"""
        if self.current_keypoint_idx > 0:
            self.save_current_image()
        
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self.load_current_image()
    
    def prev_image(self):
        """Move to previous image"""
        if self.current_keypoint_idx > 0:
            self.save_current_image()
        
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current_image()
    
    def run(self):
        """Main annotation loop"""
        self.load_current_image()
        
        print("\n=== DWPose Keypoint Annotation Tool ===")
        print(f"Annotating {len(self.image_files)} images")
        print("\nControls:")
        print("  Left-click:  Place keypoint")
        print("  Right-click: Skip keypoint (mark as not visible)")
        print("  U: Undo last keypoint")
        print("  C: Copy keypoints from previous image")
        print("  S: Save current annotation")
        print("  X: DELETE current image and annotation (creates backup)")
        print("  N/D/→: Next image")
        print("  P/A/←: Previous image")
        print("  Q/ESC: Quit")
        print("\nKeypoint order:", ", ".join(BODY_KEYPOINTS))
        print("=" * 50 + "\n")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                if self.current_keypoint_idx > 0:
                    self.save_current_image()
                break
            elif key == ord('n') or key == ord('d') or key == 83:  # N, D, or right arrow
                self.next_image()
            elif key == ord('p') or key == ord('a') or key == 81:  # P, A, or left arrow
                self.prev_image()
            elif key == ord('u'):  # Undo
                self.undo_last_keypoint()
            elif key == ord('c'):  # Copy from previous
                self.copy_from_previous()
            elif key == ord('s'):  # Save
                self.save_current_image()
            elif key == ord('x'):  # Delete
                self.delete_current_image()
        
        cv2.destroyAllWindows()
        print("\nAnnotation session complete!")
        print(f"Annotated {len(self.keypoints)} images")

def main():
    import sys
    
    # Set paths - relative to script location
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    image_dir = project_dir / "input_stickman_video" / "all_bw_images_480p"
    output_dir = project_dir / "input_stickman_video" / "keypoint_annotations"
    
    # Allow override from command line
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    annotator = KeypointAnnotator(image_dir, output_dir)
    annotator.run()

if __name__ == "__main__":
    main()
