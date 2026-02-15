"""
Group similar images and provide GUI for selection
Uses pose-aware features to find similar stickman poses
Allows user to select which images to keep from each group
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from collections import defaultdict
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


class ImageSimilarityGrouper:
    def __init__(self, image_dir, feature_type='pose_aware'):
        self.image_dir = Path(image_dir)
        self.feature_type = feature_type  # 'pose_aware', 'deep', or 'hash'
        self.image_files = []
        self.features = []
        
        # Load model only if using deep features
        if feature_type == 'deep':
            print("Loading ResNet18 for feature extraction...")
            self.model = models.resnet18(pretrained=True)
            self.model.eval()
            # Remove final classification layer to get features
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            print(f"Using {feature_type} feature extraction...")
    
    def load_images(self):
        """Load all image files from directory"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for ext in extensions:
            self.image_files.extend(self.image_dir.glob(f'*{ext}'))
            self.image_files.extend(self.image_dir.glob(f'*{ext.upper()}'))
        
        self.image_files = sorted(set(self.image_files))
        print(f"Found {len(self.image_files)} images")
    
    def compute_perceptual_hash(self, image_path, hash_size=16):
        """Compute perceptual hash for an image"""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Resize to hash_size x hash_size
        img_resized = cv2.resize(img, (hash_size, hash_size))
        
        # Compute DCT
        dct = cv2.dct(np.float32(img_resized))
        
        # Get top-left 8x8 corner (low frequencies)
        dct_low = dct[:8, :8]
        
        # Compute median
        median = np.median(dct_low)
        
        # Create binary hash
        hash_vec = (dct_low > median).flatten().astype(np.float32)
        return hash_vec
    
    def compute_deep_features(self, image_path):
        """Compute deep features using ResNet"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)
            
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            with torch.no_grad():
                features = self.model(img_tensor)
            
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def compute_pose_aware_features(self, image_path, grid_size=8):
        """
        Compute pose-aware features using HOG + spatial features
        Better for detecting pose/position differences in stickman images
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize to standard size
            gray = cv2.resize(gray, (256, 256))
            
            # 1. Edge detection (captures pose structure)
            edges = cv2.Canny(gray, 50, 150)
            
            # 2. Spatial grid features - divide image into grid and compute pixel density
            h, w = edges.shape
            grid_h, grid_w = h // grid_size, w // grid_size
            spatial_features = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    cell = edges[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                    # Compute density of edge pixels in this cell
                    density = np.sum(cell > 0) / (grid_h * grid_w)
                    spatial_features.append(density)
            
            # 3. Compute HOG (Histogram of Oriented Gradients)
            # More sensitive to pose orientation
            win_size = (256, 256)
            block_size = (32, 32)
            block_stride = (16, 16)
            cell_size = (16, 16)
            nbins = 9
            
            hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
            hog_features = hog.compute(gray).flatten()
            
            # 4. Moment features (captures overall shape/position)
            moments = cv2.moments(edges)
            moment_features = [
                moments['m00'],  # Area
                moments['m10'] / (moments['m00'] + 1e-5),  # Center X
                moments['m01'] / (moments['m00'] + 1e-5),  # Center Y
                moments['mu20'],  # Variance X
                moments['mu02'],  # Variance Y
                moments['mu11'],  # Covariance
            ]
            
            # Combine all features
            combined = np.concatenate([
                np.array(spatial_features),  # Spatial grid
                hog_features[::10],  # Downsample HOG (too many features)
                np.array(moment_features)
            ])
            
            # Normalize
            combined = combined / (np.linalg.norm(combined) + 1e-8)
            
            return combined
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def extract_features(self):
        """Extract features from all images"""
        print("\nExtracting features...")
        self.features = []
        valid_files = []
        
        for img_path in tqdm(self.image_files):
            if self.feature_type == 'deep':
                feat = self.compute_deep_features(img_path)
            elif self.feature_type == 'pose_aware':
                feat = self.compute_pose_aware_features(img_path)
            else:  # 'hash'
                feat = self.compute_perceptual_hash(img_path)
            
            if feat is not None:
                self.features.append(feat)
                valid_files.append(img_path)
        
        self.image_files = valid_files
        self.features = np.array(self.features)
        print(f"Extracted features for {len(self.features)} images")
        print(f"Feature dimension: {self.features.shape[1]}")
    
    def find_groups(self, similarity_threshold=0.85, min_group_size=2):
        """
        Group similar images using clustering
        
        Args:
            similarity_threshold: Cosine similarity threshold (0-1)
            min_group_size: Minimum images in a group to be considered
        
        Returns:
            List of groups, each group is a list of image indices
        """
        print(f"\nFinding similar groups (threshold: {similarity_threshold})...")
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(self.features)
        
        # Convert similarity to distance for DBSCAN
        # Clip to ensure non-negative values (due to numerical precision)
        distance_matrix = np.clip(1 - similarity_matrix, 0, 2)
        
        # Use DBSCAN clustering
        epsilon = 1 - similarity_threshold
        clustering = DBSCAN(eps=epsilon, min_samples=min_group_size, metric='precomputed')
        labels = clustering.fit_predict(distance_matrix)
        
        # Group images by cluster label
        groups_dict = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:  # -1 means noise/outlier
                groups_dict[label].append(idx)
        
        # Convert to list of groups
        groups = [indices for indices in groups_dict.values() if len(indices) >= min_group_size]
        
        print(f"Found {len(groups)} groups of similar images")
        for i, group in enumerate(groups):
            print(f"  Group {i+1}: {len(group)} images")
        
        return groups
    
    def save_groups_info(self, groups, output_file='similar_groups.json'):
        """Save group information to JSON"""
        groups_info = {}
        for i, group in enumerate(groups):
            groups_info[f"group_{i+1}"] = [str(self.image_files[idx].name) for idx in group]
        
        with open(output_file, 'w') as f:
            json.dump(groups_info, f, indent=2)
        
        print(f"\nSaved group information to {output_file}")


class GroupSelectorGUI:
    def __init__(self, image_dir, groups, image_files):
        self.image_dir = Path(image_dir)
        self.groups = groups
        self.image_files = image_files
        self.current_group_idx = 0
        self.current_page = 0  # For pagination within groups
        self.images_per_page = 9  # 3x3 grid max
        self.selections = {}  # group_idx -> set of selected indices
        self.deleted_files = []
        
        # For click detection
        self.image_positions = []  # List of (x, y, width, height, position_in_group)
        self.header_height = 120
        self.footer_height = 50
        self.img_display_size = 430  # 400 + 15*2 border
        
        # Initialize selections - all images selected by default
        for i, group in enumerate(groups):
            self.selections[i] = set(group)
        
        self.window_name = "Similar Image Group Selector"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1600, 1000)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks on images"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is within an image
            for img_x, img_y, img_w, img_h, position_in_group in self.image_positions:
                if img_x <= x < img_x + img_w and img_y <= y < img_y + img_h:
                    # Toggle selection for this image
                    group = self.groups[self.current_group_idx]
                    if position_in_group < len(group):
                        idx = group[position_in_group]
                        if idx in self.selections[self.current_group_idx]:
                            self.selections[self.current_group_idx].remove(idx)
                        else:
                            self.selections[self.current_group_idx].add(idx)
                        
                        # Redraw
                        display = self.create_group_display(self.current_group_idx)
                        cv2.imshow(self.window_name, display)
                    break
    
    def create_group_display(self, group_idx):
        """Create a grid display of images in a group with pagination"""
        group = self.groups[group_idx]
        selected = self.selections[group_idx]
        
        # Clear image positions for click detection
        self.image_positions = []
        
        # Calculate pagination
        total_images = len(group)
        total_pages = (total_images + self.images_per_page - 1) // self.images_per_page
        
        # Ensure current page is valid
        if self.current_page >= total_pages:
            self.current_page = 0
        
        # Get images for current page
        start_idx = self.current_page * self.images_per_page
        end_idx = min(start_idx + self.images_per_page, total_images)
        page_indices = range(start_idx, end_idx)
        
        # Load images for this page
        images = []
        for i in page_indices:
            idx = group[i]
            img_path = self.image_files[idx]
            img = cv2.imread(str(self.image_dir / img_path.name))
            if img is not None:
                images.append((img, idx, i))
            else:
                images.append((np.zeros((100, 100, 3), dtype=np.uint8), idx, i))
        
        # Calculate grid dimensions (max 3x3)
        n_images = len(images)
        grid_cols = min(3, n_images)
        grid_rows = min(3, (n_images + grid_cols - 1) // grid_cols)
        
        # Larger image size for better visibility
        img_size = 400
        resized_images = []
        
        for img, idx_in_full_group, position_in_group in images:
            # Resize maintaining aspect ratio
            h, w = img.shape[:2]
            scale = img_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h))
            
            # Pad to square
            pad_h = (img_size - new_h) // 2
            pad_w = (img_size - new_w) // 2
            padded = cv2.copyMakeBorder(resized, pad_h, img_size-new_h-pad_h, 
                                       pad_w, img_size-new_w-pad_w, 
                                       cv2.BORDER_CONSTANT, value=(240, 240, 240))
            
            # Add green border if selected, red if not
            border_color = (0, 255, 0) if idx_in_full_group in selected else (0, 0, 255)
            padded = cv2.copyMakeBorder(padded, 15, 15, 15, 15, 
                                       cv2.BORDER_CONSTANT, value=border_color)
            
            # Add image number and status
            status = 'KEEP' if idx_in_full_group in selected else 'DELETE'
            key_num = (position_in_group % self.images_per_page) + 1
            text = f"[{key_num}] #{position_in_group+1}: {status}"
            cv2.putText(padded, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, border_color, 3)
            
            # Add filename (smaller)
            filename = self.image_files[idx_in_full_group].name
            if len(filename) > 30:
                filename = filename[:27] + "..."
            cv2.putText(padded, filename, (20, img_size + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            resized_images.append(padded)
        
        # Pad to fill grid if needed
        while len(resized_images) < grid_cols * grid_rows:
            resized_images.append(np.ones_like(resized_images[0]) * 250)
        
        # Create grid and track positions for click detection
        rows = []
        for r in range(grid_rows):
            start = r * grid_cols
            end = min(start + grid_cols, len(resized_images))
            row_images = resized_images[start:end]
            
            # Store image positions (before padding)
            for c, img_idx in enumerate(range(start, end)):
                if img_idx < len(images):  # Only store real images, not padding
                    _, _, position_in_group = images[img_idx]
                    img_x = c * self.img_display_size
                    img_y = self.header_height + r * self.img_display_size
                    self.image_positions.append((img_x, img_y, self.img_display_size, self.img_display_size, position_in_group))
            
            # Pad row if needed
            while len(row_images) < grid_cols:
                row_images.append(np.ones_like(resized_images[0]) * 250)
            
            rows.append(np.hstack(row_images))
        
        grid = np.vstack(rows)
        
        # Add header
        header = np.ones((self.header_height, grid.shape[1], 3), dtype=np.uint8) * 255
        
        # Title
        title = f"Group {group_idx + 1}/{len(self.groups)} - {len(group)} images total"
        cv2.putText(header, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2)
        
        # Page info
        page_info = f"Page {self.current_page + 1}/{total_pages} (showing {start_idx+1}-{end_idx})"
        cv2.putText(header, page_info, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 50), 2)
        
        # Instructions line 1
        instructions1 = "CLICK image to toggle | Keys 1-9 also work | SPACE/]: Next page | [: Prev page"
        cv2.putText(header, instructions1, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 1)
        
        # Add footer
        footer = np.ones((self.footer_height, grid.shape[1], 3), dtype=np.uint8) * 255
        
        # Instructions line 2
        instructions2 = "N: Next group | P: Prev group | D: Done & Delete | Q: Quit"
        cv2.putText(footer, instructions2, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
        
        result = np.vstack([header, grid, footer])
        return result
    
    def run(self):
        """Run the interactive GUI"""
        print("\n" + "="*70)
        print("SIMILAR IMAGE GROUP SELECTOR")
        print("="*70)
        print("\nControls:")
        print("  CLICK on image : Toggle KEEP/DELETE")
        print("  Number keys 1-9 : Also toggle KEEP/DELETE")
        print("  SPACE or ] : Next page (within group)")
        print("  [ : Previous page (within group)")
        print("  N or → : Next group")
        print("  P or ← : Previous group")
        print("  D : Done (apply deletions)")
        print("  Q : Quit without saving")
        print("\nGreen border = KEEP, Red border = DELETE")
        print("="*70)
        
        while True:
            display = self.create_group_display(self.current_group_idx)
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(0) & 0xFF
            
            # Number keys 1-9 to toggle selection
            if ord('1') <= key <= ord('9'):
                img_num = key - ord('1')  # 0-indexed relative to current page
                group = self.groups[self.current_group_idx]
                
                # Calculate actual index in group
                actual_idx_in_group = self.current_page * self.images_per_page + img_num
                
                if actual_idx_in_group < len(group):
                    idx = group[actual_idx_in_group]
                    if idx in self.selections[self.current_group_idx]:
                        self.selections[self.current_group_idx].remove(idx)
                    else:
                        self.selections[self.current_group_idx].add(idx)
            
            # Next page
            elif key == ord(' ') or key == ord(']'):  # Space or ]
                group = self.groups[self.current_group_idx]
                total_pages = (len(group) + self.images_per_page - 1) // self.images_per_page
                self.current_page = (self.current_page + 1) % total_pages
            
            # Previous page
            elif key == ord('['):
                group = self.groups[self.current_group_idx]
                total_pages = (len(group) + self.images_per_page - 1) // self.images_per_page
                self.current_page = (self.current_page - 1) % total_pages
            
            # Next group
            elif key == ord('n') or key == ord('N') or key == 83:  # Right arrow
                self.current_group_idx = (self.current_group_idx + 1) % len(self.groups)
                self.current_page = 0  # Reset to first page of new group
            
            # Previous group
            elif key == ord('p') or key == ord('P') or key == 81:  # Left arrow
                self.current_group_idx = (self.current_group_idx - 1) % len(self.groups)
                self.current_page = 0  # Reset to first page of new group
            
            # Done
            elif key == ord('d') or key == ord('D'):
                self.apply_deletions()
                break
            
            # Quit
            elif key == ord('q') or key == ord('Q') or key == 27:  # ESC
                print("\nQuitting without changes.")
                break
        
        cv2.destroyAllWindows()
    
    def apply_deletions(self):
        """Delete unselected images"""
        print("\n" + "="*60)
        print("APPLYING DELETIONS")
        print("="*60)
        
        # Collect all images to delete
        to_delete = set()
        for group_idx, group in enumerate(self.groups):
            selected = self.selections[group_idx]
            for img_idx in group:
                if img_idx not in selected:
                    to_delete.add(img_idx)
        
        if len(to_delete) == 0:
            print("No images to delete.")
            return
        
        print(f"\nWill delete {len(to_delete)} images:")
        for idx in sorted(to_delete):
            print(f"  - {self.image_files[idx].name}")
        
        response = input(f"\nConfirm deletion of {len(to_delete)} images? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return
        
        # Delete files
        print("\nDeleting files...")
        for idx in tqdm(sorted(to_delete)):
            img_path = self.image_dir / self.image_files[idx].name
            try:
                img_path.unlink()
                self.deleted_files.append(str(img_path.name))
            except Exception as e:
                print(f"Error deleting {img_path}: {e}")
        
        print(f"\nDeleted {len(self.deleted_files)} images successfully")
        
        # Save deletion log
        log_file = self.image_dir / "deleted_images_log.json"
        with open(log_file, 'w') as f:
            json.dump({
                "deleted_count": len(self.deleted_files),
                "deleted_files": self.deleted_files
            }, f, indent=2)
        print(f"Deletion log saved to {log_file}")


def main():
    parser = argparse.ArgumentParser(description='Group similar images and select which to keep')
    
    parser.add_argument('--input_dir', type=str,
                        default='input_stickman_video/lora_training_colored_images',
                        help='Directory with images to analyze')
    parser.add_argument('--similarity_threshold', type=float, default=0.90,
                        help='Similarity threshold (0-1, default: 0.90)')
    parser.add_argument('--min_group_size', type=int, default=2,
                        help='Minimum images in a group (default: 2)')
    parser.add_argument('--feature_type', type=str, default='pose_aware',
                        choices=['pose_aware', 'deep', 'hash'],
                        help='Feature extraction method: pose_aware (best for poses), deep (ResNet), hash (fastest)')
    parser.add_argument('--save_groups_only', action='store_true',
                        help='Only save groups to JSON, do not open GUI')
    
    args = parser.parse_args()
    
    # Initialize grouper
    grouper = ImageSimilarityGrouper(
        args.input_dir,
        feature_type=args.feature_type
    )
    
    # Load and process images
    grouper.load_images()
    
    if len(grouper.image_files) == 0:
        print("No images found!")
        return
    
    grouper.extract_features()
    groups = grouper.find_groups(args.similarity_threshold, args.min_group_size)
    
    if len(groups) == 0:
        print("\nNo similar groups found with current settings.")
        print("Try lowering --similarity_threshold")
        return
    
    # Save groups info
    grouper.save_groups_info(groups, 
                            Path(args.input_dir) / 'similar_groups.json')
    
    if args.save_groups_only:
        print("\nGroups saved. Use without --save_groups_only to open GUI.")
        return
    
    # Run GUI for selection
    gui = GroupSelectorGUI(args.input_dir, groups, grouper.image_files)
    gui.run()


if __name__ == '__main__':
    main()
