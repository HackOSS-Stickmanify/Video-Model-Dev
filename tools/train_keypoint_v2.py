"""
Improved keypoint detector with heavy data augmentation
Designed to work with limited training data (50-100 samples)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# OpenPose keypoint pairs for horizontal flip (left <-> right)
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

class KeypointDatasetAugmented(Dataset):
    """Dataset with heavy augmentation for limited data"""
    def __init__(self, image_dir, annotation_file, img_size=(256, 256), augment=True):
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        self.augment = augment
        
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.image_names = list(self.annotations.keys())
        print(f"Loaded {len(self.image_names)} annotated images")
        
    def __len__(self):
        # Virtual augmentation - each image appears multiple times
        return len(self.image_names) * (8 if self.augment else 1)
    
    def horizontal_flip_keypoints(self, keypoints, img_width):
        """Flip keypoints horizontally and swap left/right pairs"""
        flipped = keypoints.copy()
        # Flip x coordinates
        flipped[:, 0] = img_width - flipped[:, 0]
        # Swap left/right pairs
        for left, right in FLIP_PAIRS:
            flipped[left], flipped[right] = flipped[right].copy(), flipped[left].copy()
        return flipped
    
    def __getitem__(self, idx):
        # Map virtual index to real image
        real_idx = idx % len(self.image_names)
        aug_idx = idx // len(self.image_names)
        
        img_name = self.image_names[real_idx]
        img_path = self.image_dir / img_name
        
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load: {img_path}")
        
        orig_h, orig_w = img.shape[:2]
        
        # Load keypoints
        keypoints = np.array(self.annotations[img_name], dtype=np.float32)
        
        # Apply augmentations
        if self.augment:
            # Horizontal flip (50% chance based on aug_idx)
            if aug_idx % 2 == 1:
                img = cv2.flip(img, 1)
                keypoints = self.horizontal_flip_keypoints(keypoints, orig_w)
            
            # Random rotation (-15 to 15 degrees)
            if aug_idx >= 2:
                angle = random.uniform(-15, 15)
                center = (orig_w // 2, orig_h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (orig_w, orig_h))
                
                # Rotate keypoints
                for i in range(len(keypoints)):
                    if keypoints[i, 2] > 0:  # Only rotate visible keypoints
                        x, y = keypoints[i, 0], keypoints[i, 1]
                        new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
                        new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
                        keypoints[i, 0] = new_x
                        keypoints[i, 1] = new_y
            
            # Random scale (0.9 to 1.1)
            if aug_idx >= 4:
                scale = random.uniform(0.9, 1.1)
                new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                img_scaled = cv2.resize(img, (new_w, new_h))
                
                # Pad or crop to original size
                if scale > 1:
                    start_x = (new_w - orig_w) // 2
                    start_y = (new_h - orig_h) // 2
                    img = img_scaled[start_y:start_y+orig_h, start_x:start_x+orig_w]
                    keypoints[:, 0] = keypoints[:, 0] * scale - start_x
                    keypoints[:, 1] = keypoints[:, 1] * scale - start_y
                else:
                    pad_x = (orig_w - new_w) // 2
                    pad_y = (orig_h - new_h) // 2
                    img = cv2.copyMakeBorder(img_scaled, pad_y, orig_h-new_h-pad_y, 
                                            pad_x, orig_w-new_w-pad_x, cv2.BORDER_CONSTANT)
                    keypoints[:, 0] = keypoints[:, 0] * scale + pad_x
                    keypoints[:, 1] = keypoints[:, 1] * scale + pad_y
            
            # Color jitter
            if aug_idx >= 6:
                # Random brightness
                img = img.astype(np.float32)
                img = img * random.uniform(0.8, 1.2)
                img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Resize
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Scale keypoints to resized image
        keypoints[:, 0] = keypoints[:, 0] * (self.img_size[0] / orig_w)
        keypoints[:, 1] = keypoints[:, 1] * (self.img_size[1] / orig_h)
        
        # Normalize coordinates to [0, 1]
        keypoints[:, 0] = keypoints[:, 0] / self.img_size[0]
        keypoints[:, 1] = keypoints[:, 1] / self.img_size[1]
        
        # Clip to valid range
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, 1)
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, 1)
        
        # Normalize visibility to [0, 1] (was 0/2)
        keypoints[:, 2] = np.clip(keypoints[:, 2] / 2.0, 0, 1)
        
        # Convert image to tensor
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = (img - 0.5) / 0.5
        
        keypoints = torch.from_numpy(keypoints)
        
        return img, keypoints


class KeypointRegressor(nn.Module):
    """
    Direct regression model for keypoint detection
    Predicts normalized (x, y, visibility) for each keypoint
    """
    def __init__(self, num_keypoints=18):
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # Use ResNet18 backbone (smaller, easier to train)
        from torchvision.models import resnet18, ResNet18_Weights
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_keypoints * 3)  # x, y, visibility for each
        )
        
    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        # Reshape to (batch, num_keypoints, 3)
        out = out.view(-1, self.num_keypoints, 3)
        # Sigmoid for x, y (normalized coords) and visibility
        out = torch.sigmoid(out)
        return out


def wing_loss(pred, target, w=10, epsilon=2):
    """
    Wing loss - better for small errors (facial/keypoint detection)
    """
    c = w - w * np.log(1 + w / epsilon)
    diff = torch.abs(pred - target)
    loss = torch.where(
        diff < w,
        w * torch.log(1 + diff / epsilon),
        diff - c
    )
    return loss.mean()


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, keypoints in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        keypoints = keypoints.to(device)
        
        pred = model(images)
        
        # Separate losses for coordinates and visibility
        coord_loss = wing_loss(pred[:, :, :2], keypoints[:, :, :2])
        vis_loss = nn.functional.binary_cross_entropy(pred[:, :, 2], keypoints[:, :, 2])
        
        loss = coord_loss + 0.1 * vis_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_pixel_error = 0
    
    with torch.no_grad():
        for images, keypoints in dataloader:
            images = images.to(device)
            keypoints = keypoints.to(device)
            
            pred = model(images)
            
            # Coordinate loss
            coord_loss = wing_loss(pred[:, :, :2], keypoints[:, :, :2])
            vis_loss = nn.functional.binary_cross_entropy(pred[:, :, 2], keypoints[:, :, 2])
            loss = coord_loss + 0.1 * vis_loss
            
            total_loss += loss.item()
            
            # Calculate pixel error (denormalize to 256x256)
            pred_coords = pred[:, :, :2] * 256
            target_coords = keypoints[:, :, :2] * 256
            visible = keypoints[:, :, 2] > 0.5
            
            if visible.sum() > 0:
                pixel_error = torch.sqrt(((pred_coords - target_coords) ** 2).sum(dim=-1))
                pixel_error = pixel_error[visible].mean()
                total_pixel_error += pixel_error.item()
    
    return total_loss / len(dataloader), total_pixel_error / len(dataloader)


def main():
    # Configuration - relative to script location
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    image_dir = project_dir / "input_stickman_video" / "all_bw_images_480p"
    annotation_file = project_dir / "input_stickman_video" / "keypoint_annotations" / "annotations.json"
    output_dir = script_dir / "checkpoints"
    output_dir.mkdir(exist_ok=True)
    
    # Hyperparameters
    batch_size = 16
    num_epochs = 200
    learning_rate = 1e-4
    img_size = (256, 256)
    val_split = 0.15
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset with augmentation
    full_dataset = KeypointDatasetAugmented(image_dir, annotation_file, img_size=img_size, augment=True)
    val_dataset = KeypointDatasetAugmented(image_dir, annotation_file, img_size=img_size, augment=False)
    
    # Split based on original image count
    num_original = len(full_dataset.image_names)
    val_size = int(num_original * val_split)
    train_size = num_original - val_size
    
    # Create index splits on original images
    indices = list(range(num_original))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets (train uses augmented virtual indices)
    train_virtual_indices = []
    for i in train_indices:
        for aug in range(8):  # 8 augmentations per image
            train_virtual_indices.append(i + aug * num_original)
    
    train_subset = torch.utils.data.Subset(full_dataset, train_virtual_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    print(f"Train samples: {len(train_subset)} (with augmentation)")
    print(f"Val samples: {len(val_subset)}")
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = KeypointRegressor(num_keypoints=18).to(device)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_val_loss = float('inf')
    best_pixel_error = float('inf')
    patience = 0
    max_patience = 30
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, pixel_error = validate(model, val_loader, device)
        
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
              f"Pixel Error: {pixel_error:.1f}px, LR: {lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_pixel_error = pixel_error
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'pixel_error': pixel_error,
            }, output_dir / 'best_keypoint_regressor.pt')
            print(f"  ✓ Saved best model (pixel error: {pixel_error:.1f}px)")
        else:
            patience += 1
        
        # Early stopping
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        # Checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best pixel error: {best_pixel_error:.1f}px")

if __name__ == "__main__":
    main()
