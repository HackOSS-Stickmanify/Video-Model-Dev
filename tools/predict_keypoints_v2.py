"""
Predict keypoints using trained regressor model
"""

import torch
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import glob
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_keypoint_v2 import KeypointRegressor

def predict_on_images(model_path, image_dir, annotation_file, img_size=(256, 256)):
    """Run keypoint prediction on all unannotated images"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = KeypointRegressor(num_keypoints=18).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from: {model_path}")
    print(f"Model pixel error: {checkpoint.get('pixel_error', 'N/A')}")
    
    # Load existing annotations
    annotation_path = Path(annotation_file)
    if annotation_path.exists():
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations)} existing annotations")
    else:
        annotations = {}
    
    # Get all images
    image_dir = Path(image_dir)
    image_files = sorted(glob.glob(str(image_dir / "*.jpg")))
    print(f"Found {len(image_files)} total images")
    
    # Count unannotated
    unannotated = [f for f in image_files if Path(f).name not in annotations]
    print(f"Predicting on {len(unannotated)} unannotated images")
    
    # Process images
    with torch.no_grad():
        for img_path in tqdm(unannotated, desc="Predicting"):
            img_name = Path(img_path).name
            
            # Load and preprocess
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            orig_h, orig_w = img.shape[:2]
            
            # Resize and normalize
            img_resized = cv2.resize(img, img_size)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Predict
            pred = model(img_tensor)[0].cpu().numpy()
            
            # Convert normalized coords back to original image size
            keypoints = []
            for i in range(18):
                x = pred[i, 0] * orig_w
                y = pred[i, 1] * orig_h
                vis = 2 if pred[i, 2] > 0.5 else 0
                keypoints.append([float(x), float(y), int(vis)])
            
            annotations[img_name] = keypoints
    
    # Save all annotations
    with open(annotation_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\nSaved {len(annotations)} total annotations to {annotation_path}")

def main():
    # Configuration - relative to script location
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    model_path = script_dir / "checkpoints" / "best_keypoint_regressor.pt"
    image_dir = project_dir / "input_stickman_video" / "all_bw_images_480p"
    annotation_file = project_dir / "input_stickman_video" / "keypoint_annotations" / "annotations.json"
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_keypoint_v2.py")
        return
    
    predict_on_images(model_path, image_dir, annotation_file)

if __name__ == "__main__":
    main()
