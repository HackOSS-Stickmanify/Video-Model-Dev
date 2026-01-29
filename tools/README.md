# Keypoint Annotation Tools

Tools for annotating stickman images with OpenPose-style 18-keypoint body pose data, compatible with DWPose format (body only, hands/face disabled).

## Background & Context

**Goal**: Annotate ~500 stickman images in `input_stickman_video/all_bw_images_480p/` with DWPose-compatible keypoints for training a video model.

**Problem**: Manual annotation of 500+ images is tedious.

**Solution**: Semi-automated approach:
1. Manually annotate 80-150 diverse images
2. Train a lightweight CNN (ResNet34-based) on those annotations
3. Use the trained model to predict keypoints on remaining images
4. Review/correct predictions as needed

**What was tried**:
- First attempt used MobileNetV2 + heatmap prediction → poor results
- Second attempt used ResNet18 + direct coordinate regression + data augmentation → 8.7px error
- Third attempt (current) uses ResNet34 + mixed precision + OneCycleLR → **5.7px error**, best results

## Files

```
tools/
├── annotate_keypoints.py      # Manual annotation GUI (OpenCV-based)
├── train_keypoint_v2.py       # Legacy training script (ResNet18)
├── train_keypoint_v3.py       # Optimized training (ResNet34 + FP16) ← RECOMMENDED
├── predict_keypoints_v2.py    # Legacy prediction script
├── predict_keypoints_v3.py    # Batch prediction with FP16 ← RECOMMENDED
├── checkpoints/               # Saved models
│   └── best_keypoint_regressor.pt
└── README.md                  # This file

input_stickman_video/
├── all_bw_images_480p/        # Source images (509 stickman images, 480x640)
└── keypoint_annotations/
    ├── annotations.json       # Active annotations file (all 509 images)
    ├── annotations_manual_backup_v1.json  # Backup of ~80 manual annotations
    └── annotations_manual_backup_v2.json  # Backup of 126 manual annotations
```

## Architecture Details

**Model**: `KeypointRegressor` in `train_keypoint_v3.py`
- Backbone: ResNet34 (pretrained on ImageNet)
- Head: FC layers (512 → 512 → 256 → 54) with dropout
- Output: 18 keypoints × 3 values (x, y, visibility) normalized to [0,1]
- Loss: Wing loss for coordinates + BCE for visibility

**Data Augmentation** (12x virtual samples per image):
- Horizontal flip with left/right keypoint swapping
- Random rotation (±20°)
- Random scale (0.85-1.15x)
- Color jitter (brightness 0.7-1.3x)

**Training Config (v3 - RTX 3090 optimized)**:
- Image size: 256×256
- Batch size: 64
- Learning rate: 3e-4 with OneCycleLR
- Mixed precision (FP16) for 2x speed
- Early stopping: patience=25
- Validation split: 15%
- Workers: 8

## Workflow

### Step 1: Manual Annotation

```bash
python tools/annotate_keypoints.py
```

**Controls:**
| Key | Action |
|-----|--------|
| Left-click | Place keypoint |
| Right-click | Skip keypoint (not visible) |
| N / D / → | Next image |
| P / A / ← | Previous image |
| U | Undo last keypoint |
| C | Copy keypoints from previous image |
| S | Save current annotation |
| Q / ESC | Quit |

**Keypoint Order (OpenPose 18-point):**
1. nose
2. neck
3. right_shoulder
4. right_elbow
5. right_wrist
6. left_shoulder
7. left_elbow
8. left_wrist
9. right_hip
10. right_knee
11. right_ankle
12. left_hip
13. left_knee
14. left_ankle
15. right_eye
16. left_eye
17. right_ear
18. left_ear

### Step 2: Train Model

After annotating 50-100 diverse images:

```bash
# Recommended (optimized for RTX 3090)
python tools/train_keypoint_v3.py --backbone resnet34 --batch-size 64 --epochs 150

# Legacy version
python tools/train_keypoint_v2.py
```

**Features (v3):**
- ResNet34 backbone with regression head
- Heavy data augmentation (12x virtual samples)
- Wing loss for better keypoint accuracy
- OneCycleLR for faster convergence
- Mixed precision (FP16) training
- Early stopping with patience=25

**Training output:**
- Best model saved to `tools/checkpoints/best_keypoint_regressor.pt`
- Reports pixel error on validation set (~5.7px with 126 training images)

### Step 3: Predict on All Images

```bash
# Recommended (batch processing with FP16)
python tools/predict_keypoints_v3.py --batch-size 64

# Legacy version
python tools/predict_keypoints_v2.py
```

This predicts keypoints on all unannotated images and adds them to the annotations file.
Manual annotations are preserved and not overwritten.

### Step 4: Review & Correct (Optional)

Run the annotation tool again to review predictions:

```bash
python tools/annotate_keypoints.py
```

Navigate through images to spot-check and correct any bad predictions.

## Output Format

Annotations are saved to:
```
input_stickman_video/keypoint_annotations/annotations.json
```

Format:
```json
{
  "image_name.jpg": [
    [x, y, visibility],  // keypoint 0 (nose)
    [x, y, visibility],  // keypoint 1 (neck)
    ...                  // 18 keypoints total
  ]
}
```

Visibility values:
- `0` = not visible
- `2` = visible

## Model Performance

| Version | Backbone | Training Images | Pixel Error |
|---------|----------|-----------------|-------------|
| v2 | ResNet18 | 80 | 8.7px |
| v3 | ResNet34 | 126 | **5.7px** |

Current model achieves **5.7px error** on 256x256 images, suitable for stickman pose estimation.

## Tips

1. **Diverse training data**: Annotate images with different poses, not just sequential frames
2. **Copy from previous**: Use `C` key for video sequences where poses are similar
3. **Iterative improvement**: If predictions are bad, correct some and retrain
4. **More data = better**: 100+ annotations will give better results than 50

## Current State

- **126 images** manually annotated (saved in `annotations_manual_backup_v2.json`)
- **Model trained** with 5.7px validation error (ResNet34 backbone)
- **All 509 images** have keypoint annotations (stored in `annotations.json`)
  - 126 manual annotations
  - 383 model predictions

## For Future AI/Developers

If you need to modify this:

1. **Keypoint format**: OpenPose 18-point (not COCO 17-point). Key difference is "neck" keypoint at index 1.

2. **Paths are relative**: All scripts use `Path(__file__).parent` to find project root.

3. **To retrain after adding annotations**:
   ```bash
   python tools/train_keypoint_v3.py --backbone resnet34 --batch-size 64
   python tools/predict_keypoints_v3.py --batch-size 64
   ```

4. **To restore manual-only annotations**:
   ```bash
   cp input_stickman_video/keypoint_annotations/annotations_manual_backup_v2.json input_stickman_video/keypoint_annotations/annotations.json
   ```

5. **Annotation format** in JSON:
   ```json
   {
     "vid1_1.jpg": [[x, y, vis], [x, y, vis], ...],  // 18 keypoints
   }
   ```
   Where `vis`: 0=not visible, 2=visible (normalized to 0-1 during training)
