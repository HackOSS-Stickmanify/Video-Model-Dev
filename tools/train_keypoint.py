"""
Optimized keypoint detector training for RTX 3090 / Intel XPU
- Mixed precision (FP16) for 2x speed
- Larger batch size (64)
- More workers for data loading
- OneCycleLR for faster convergence
- ResNet34 backbone option
- Supports CUDA, Intel XPU, and CPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# XPU support detection
try:
    import intel_extension_for_pytorch as ipex
    HAS_XPU = hasattr(torch, 'xpu') and torch.xpu.is_available()
except ImportError:
    ipex = None
    HAS_XPU = False

HAS_CUDA = torch.cuda.is_available()


def get_device() -> torch.device:
    """Get the best available device (CUDA > XPU > CPU)."""
    if HAS_CUDA:
        return torch.device("cuda")
    elif HAS_XPU:
        return torch.device("xpu")
    else:
        return torch.device("cpu")


def get_device_type(device: torch.device) -> str:
    """Get device type string for autocast."""
    return device.type if device.type in ("cuda", "xpu") else "cpu"


def get_scaler(device: torch.device):
    """Get appropriate GradScaler for the device."""
    if device.type == "cuda":
        from torch.cuda.amp import GradScaler
        return GradScaler()
    elif device.type == "xpu":
        return torch.amp.GradScaler(device_type="xpu")
    return None
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import argparse

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
    def __init__(self, image_dir, annotations, img_size=(256, 256), augment=True, aug_factor=8):
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        self.augment = augment
        self.aug_factor = aug_factor if augment else 1
        self.annotations = annotations
        self.image_names = list(annotations.keys())
        print(f"Dataset: {len(self.image_names)} images, augment={augment}, factor={self.aug_factor}")
        
    def __len__(self):
        return len(self.image_names) * self.aug_factor
    
    def horizontal_flip_keypoints(self, keypoints, img_width):
        """Flip keypoints horizontally and swap left/right pairs"""
        flipped = keypoints.copy()
        flipped[:, 0] = img_width - flipped[:, 0]
        for left, right in FLIP_PAIRS:
            flipped[left], flipped[right] = flipped[right].copy(), flipped[left].copy()
        return flipped
    
    def __getitem__(self, idx):
        real_idx = idx % len(self.image_names)
        aug_idx = idx // len(self.image_names)
        
        img_name = self.image_names[real_idx]
        img_path = self.image_dir / img_name
        
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load: {img_path}")
        
        orig_h, orig_w = img.shape[:2]
        keypoints = np.array(self.annotations[img_name], dtype=np.float32)
        
        if self.augment and aug_idx > 0:
            # Horizontal flip (odd aug_idx)
            if aug_idx % 2 == 1:
                img = cv2.flip(img, 1)
                keypoints = self.horizontal_flip_keypoints(keypoints, orig_w)
            
            # Random rotation
            if aug_idx >= 2:
                angle = random.uniform(-20, 20)  # Slightly more rotation
                center = (orig_w // 2, orig_h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (orig_w, orig_h))
                
                for i in range(len(keypoints)):
                    if keypoints[i, 2] > 0:
                        x, y = keypoints[i, 0], keypoints[i, 1]
                        new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
                        new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
                        keypoints[i, 0] = new_x
                        keypoints[i, 1] = new_y
            
            # Random scale
            if aug_idx >= 4:
                scale = random.uniform(0.85, 1.15)
                new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                img_scaled = cv2.resize(img, (new_w, new_h))
                
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
            
            # Color jitter (brightness, contrast)
            if aug_idx >= 6:
                img = img.astype(np.float32)
                img = img * random.uniform(0.7, 1.3)  # Brightness
                img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Resize
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Scale keypoints
        keypoints[:, 0] = np.clip(keypoints[:, 0] * (self.img_size[0] / orig_w) / self.img_size[0], 0, 1)
        keypoints[:, 1] = np.clip(keypoints[:, 1] * (self.img_size[1] / orig_h) / self.img_size[1], 0, 1)
        keypoints[:, 2] = np.clip(keypoints[:, 2] / 2.0, 0, 1)
        
        # Convert to tensor
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        img = (img - 0.5) / 0.5
        
        return img, torch.from_numpy(keypoints)


class KeypointRegressor(nn.Module):
    """Direct regression model for keypoint detection"""
    def __init__(self, num_keypoints=18, backbone='resnet34'):
        super().__init__()
        self.num_keypoints = num_keypoints
        
        if backbone == 'resnet34':
            from torchvision.models import resnet34, ResNet34_Weights
            resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            feat_dim = 512
        elif backbone == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            feat_dim = 2048
        else:  # resnet18
            from torchvision.models import resnet18, ResNet18_Weights
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = 512
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_keypoints * 3)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features)
        out = out.view(-1, self.num_keypoints, 3)
        out = torch.sigmoid(out)
        return out


def wing_loss(pred, target, w=10, epsilon=2):
    """Wing loss - better for small errors"""
    c = w - w * np.log(1 + w / epsilon)
    diff = torch.abs(pred - target)
    loss = torch.where(
        diff < w,
        w * torch.log(1 + diff / epsilon),
        diff - c
    )
    return loss.mean()


def train_epoch(model, dataloader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, keypoints in pbar:
        images = images.to(device, non_blocking=True)
        keypoints = keypoints.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            pred = model(images)
            coord_loss = wing_loss(pred[:, :, :2], keypoints[:, :, :2])
        # BCE needs to be outside autocast or use float32
        vis_loss = nn.functional.binary_cross_entropy(pred[:, :, 2].float(), keypoints[:, :, 2].float())
        loss = coord_loss + 0.1 * vis_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device, use_amp=True):
    model.eval()
    total_loss = 0
    total_pixel_error = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, keypoints in dataloader:
            images = images.to(device, non_blocking=True)
            keypoints = keypoints.to(device, non_blocking=True)
            
            device_type = get_device_type(device)
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                pred = model(images)
                coord_loss = wing_loss(pred[:, :, :2], keypoints[:, :, :2])
            
            # BCE outside autocast
            vis_loss = nn.functional.binary_cross_entropy(pred[:, :, 2].float(), keypoints[:, :, 2].float())
            loss = coord_loss + 0.1 * vis_loss
            
            total_loss += loss.item()
            
            # Pixel error
            pred_coords = pred[:, :, :2].float() * 256
            target_coords = keypoints[:, :, :2].float() * 256
            visible = keypoints[:, :, 2] > 0.5
            
            if visible.sum() > 0:
                pixel_error = torch.sqrt(((pred_coords - target_coords) ** 2).sum(dim=-1))
                pixel_error = pixel_error[visible].mean()
                total_pixel_error += pixel_error.item()
                num_batches += 1
    
    avg_loss = total_loss / len(dataloader)
    avg_pixel_error = total_pixel_error / max(num_batches, 1)
    return avg_loss, avg_pixel_error


def main():
    parser = argparse.ArgumentParser(description="Train keypoint regression model")
    parser.add_argument('--backbone', type=str, default='resnet34',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Backbone architecture')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=256, help='Image size')
    parser.add_argument('--aug-factor', type=int, default=10, help='Augmentation factor')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loader workers')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    image_dir = project_dir / "input_stickman_video" / "all_bw_images_480p"
    annotation_file = project_dir / "input_stickman_video" / "keypoint_annotations" / "annotations_manual_backup_v2.json"
    output_dir = project_dir / "checkpoints" / "keypoint"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    device_type = get_device_type(device)
    use_amp = device.type in ("cuda", "xpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif device.type == "xpu":
        dev_name = torch.xpu.get_device_name(0) if hasattr(torch.xpu, 'get_device_name') else "Intel XPU"
        print(f"GPU: {dev_name}")
    
    # Load annotations
    with open(annotation_file, 'r') as f:
        all_annotations = json.load(f)
    print(f"Loaded {len(all_annotations)} annotations from {annotation_file.name}")
    
    # Split train/val (85/15)
    image_names = list(all_annotations.keys())
    random.seed(42)
    random.shuffle(image_names)
    
    val_size = max(int(len(image_names) * 0.15), 5)
    train_names = image_names[:-val_size]
    val_names = image_names[-val_size:]
    
    train_annotations = {k: all_annotations[k] for k in train_names}
    val_annotations = {k: all_annotations[k] for k in val_names}
    
    print(f"Train: {len(train_names)} images, Val: {len(val_names)} images")
    
    img_size = (args.img_size, args.img_size)
    train_dataset = KeypointDatasetAugmented(image_dir, train_annotations, img_size, augment=True, aug_factor=args.aug_factor)
    val_dataset = KeypointDatasetAugmented(image_dir, val_annotations, img_size, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, persistent_workers=True)
    
    print(f"\nTraining samples per epoch: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Model
    model = KeypointRegressor(num_keypoints=18, backbone=args.backbone).to(device)
    print(f"Model: KeypointRegressor with {args.backbone} backbone")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # OneCycleLR for faster convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos'
    )
    
    # Mixed precision scaler
    scaler = get_scaler(device)
    
    best_pixel_error = float('inf')
    patience = 0
    max_patience = 25
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}, Backbone: {args.backbone}")
    
    for epoch in range(args.epochs):
        train_loss = 0
        model.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, keypoints in pbar:
            images = images.to(device, non_blocking=True)
            keypoints = keypoints.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                pred = model(images)
                coord_loss = wing_loss(pred[:, :, :2], keypoints[:, :, :2])
            vis_loss = nn.functional.binary_cross_entropy(pred[:, :, 2].float(), keypoints[:, :, 2].float())
            loss = coord_loss + 0.1 * vis_loss
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        
        train_loss /= len(train_loader)
        val_loss, pixel_error = validate(model, val_loader, device, use_amp)
        
        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, PixelErr={pixel_error:.1f}px")
        
        if pixel_error < best_pixel_error:
            best_pixel_error = pixel_error
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'backbone': args.backbone,
                'val_loss': val_loss,
                'pixel_error': pixel_error,
            }, output_dir / 'best_keypoint_regressor.pt')
            print(f"  ✓ Saved best model ({pixel_error:.1f}px)")
        else:
            patience += 1
        
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best pixel error: {best_pixel_error:.1f}px")
    print(f"Model saved to: {output_dir / 'best_keypoint_regressor.pt'}")


if __name__ == "__main__":
    main()
