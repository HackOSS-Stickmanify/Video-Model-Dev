"""
Optimized keypoint prediction using trained model
- Batch processing for speed
- Mixed precision inference
- Supports CUDA, Intel XPU, and CPU
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import glob
import argparse

# XPU support detection
HAS_XPU = torch.xpu.is_available()
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


def predict_batch(model, images, device, use_amp=True):
    """Predict keypoints for a batch of images"""
    device_type = get_device_type(device)
    with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=use_amp):
        pred = model(images.to(device))
    return pred.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--only-unannotated', action='store_true', default=True)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    model_path = project_dir / "checkpoints" / "keypoint" / "best_keypoint_regressor.pt"
    image_dir = project_dir / "input_stickman_video" / "all_bw_images_480p"
    annotation_file = project_dir / "input_stickman_video" / "keypoint_annotations" / "annotations.json"
    manual_annotations = project_dir / "input_stickman_video" / "keypoint_annotations" / "annotations_manual_backup_v2.json"

    device = get_device()
    device_type = get_device_type(device)
    use_amp = device.type in ("cuda", "xpu")
    print(f"Device: {device}")

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Run train_keypoint_v3.py first")
        return

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    backbone = checkpoint.get('backbone', 'resnet34')
    print(f"Loading model with {backbone} backbone")
    print(f"Model pixel error: {checkpoint.get('pixel_error', 'N/A'):.1f}px")

    # Load model
    model = KeypointRegressor(num_keypoints=18, backbone=backbone).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load manual annotations (these are ground truth, don't overwrite)
    with open(manual_annotations, 'r') as f:
        manual_annots = json.load(f)
    print(f"Manual annotations: {len(manual_annots)}")

    # Start with manual annotations
    annotations = manual_annots.copy()

    # Get all images
    image_files = sorted(glob.glob(str(image_dir / "*.jpg")))
    print(f"Total images: {len(image_files)}")

    # Find unannotated
    to_predict = [f for f in image_files if Path(f).name not in manual_annots]
    print(f"Images to predict: {len(to_predict)}")

    if len(to_predict) == 0:
        print("All images already have manual annotations!")
        return

    # Get original image sizes
    img_sizes = {}
    for img_path in to_predict:
        img = cv2.imread(img_path)
        if img is not None:
            img_sizes[img_path] = img.shape[:2]  # (h, w)

    # Batch prediction
    img_size = (256, 256)
    batch_paths = []
    batch_tensors = []

    for img_path in tqdm(to_predict, desc="Loading images"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_resized = cv2.resize(img, img_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        img_tensor = (img_tensor - 0.5) / 0.5

        batch_paths.append(img_path)
        batch_tensors.append(img_tensor)

        # Process batch
        if len(batch_tensors) >= args.batch_size:
            batch = torch.stack(batch_tensors)
            preds = predict_batch(model, batch, device, use_amp)

            for i, (path, pred) in enumerate(zip(batch_paths, preds)):
                img_name = Path(path).name
                orig_h, orig_w = img_sizes[path]

                keypoints = []
                for j in range(18):
                    x = float(pred[j, 0] * orig_w)
                    y = float(pred[j, 1] * orig_h)
                    vis = 2 if pred[j, 2] > 0.5 else 0
                    keypoints.append([x, y, vis])

                annotations[img_name] = keypoints

            batch_paths = []
            batch_tensors = []

    # Process remaining
    if batch_tensors:
        batch = torch.stack(batch_tensors)
        preds = predict_batch(model, batch, device, use_amp)

        for i, (path, pred) in enumerate(zip(batch_paths, preds)):
            img_name = Path(path).name
            orig_h, orig_w = img_sizes[path]

            keypoints = []
            for j in range(18):
                x = float(pred[j, 0] * orig_w)
                y = float(pred[j, 1] * orig_h)
                vis = 2 if pred[j, 2] > 0.5 else 0
                keypoints.append([x, y, vis])

            annotations[img_name] = keypoints

    # Save
    with open(annotation_file, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"\nSaved {len(annotations)} annotations to {annotation_file}")
    print(f"  - Manual: {len(manual_annots)}")
    print(f"  - Predicted: {len(annotations) - len(manual_annots)}")


if __name__ == "__main__":
    main()
