# Keypoint Annotation Tools

Tools for annotating stickman images with OpenPose-style 18-keypoint body pose data, compatible with DWPose format (body only, hands/face disabled).

## Background & Context

**Goal**: Annotate ~500 stickman images in `input_stickman_video/all_bw_images_480p/` with DWPose-compatible keypoints for training a video model.

**Problem**: Manual annotation of 500+ images is tedious.

**Solution**: Semi-automated approach:
1. Manually annotate 80-150 diverse images
2. Train a lightweight CNN (ResNet18-based) on those annotations
3. Use the trained model to predict keypoints on remaining images
4. Review/correct predictions as needed

**What was tried**:
- First attempt used MobileNetV2 + heatmap prediction → poor results
- Second attempt (current) uses ResNet18 + direct coordinate regression + heavy data augmentation → **8.7px error**, much better results

## Files

```
tools/
├── annotate_keypoints.py      # Manual annotation GUI (OpenCV-based)
├── train_keypoint_v2.py       # Train keypoint detector (ResNet18 + regression)
├── predict_keypoints_v2.py    # Predict on unannotated images
├── checkpoints/               # Saved models
│   └── best_keypoint_regressor.pt
└── README.md                  # This file

input_stickman_video/
├── all_bw_images_480p/        # Source images (509 stickman images, 480x640)
└── keypoint_annotations/
    ├── annotations.json       # Active annotations file
    └── annotations_manual_backup.json  # Backup of manual-only annotations
```

## Architecture Details

**Model**: `KeypointRegressor` in `train_keypoint_v2.py`
- Backbone: ResNet18 (pretrained on ImageNet)
- Head: FC layers (512 → 256 → 128 → 54) with dropout
- Output: 18 keypoints × 3 values (x, y, visibility) normalized to [0,1]
- Loss: Wing loss for coordinates + BCE for visibility

**Data Augmentation** (8x virtual samples per image):
- Horizontal flip with left/right keypoint swapping
- Random rotation (±15°)
- Random scale (0.9-1.1x)
- Color jitter (brightness 0.8-1.2x)

**Training Config**:
- Image size: 256×256
- Batch size: 16
- Learning rate: 1e-4 with cosine annealing
- Early stopping: patience=30
- Validation split: 15%

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
python tools/train_keypoint_v2.py
```

**Features:**
- ResNet18 backbone with regression head
- Heavy data augmentation (8x virtual samples)
- Wing loss for better keypoint accuracy
- Cosine annealing learning rate
- Early stopping with patience=30

**Training output:**
- Best model saved to `tools/checkpoints/best_keypoint_regressor.pt`
- Reports pixel error on validation set

### Step 3: Predict on All Images

```bash
python tools/predict_keypoints_v2.py
```

This predicts keypoints on all unannotated images and adds them to the annotations file.

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

With ~80 training images and data augmentation:
- **Pixel Error: ~8.7px** on 256x256 images
- Suitable for most stickman pose estimation tasks

## Tips

1. **Diverse training data**: Annotate images with different poses, not just sequential frames
2. **Copy from previous**: Use `C` key for video sequences where poses are similar
3. **Iterative improvement**: If predictions are bad, correct some and retrain
4. **More data = better**: 100+ annotations will give better results than 50

## Current State

- **80 images** manually annotated (saved in `annotations_manual_backup.json`)
- **Model trained** with ~8.7px validation error
- **All 509 images** have predictions (stored in `annotations.json`)
- User is adding more manual annotations to improve model accuracy

## For Future AI/Developers

If you need to modify this:

1. **Keypoint format**: OpenPose 18-point (not COCO 17-point). Key difference is "neck" keypoint at index 1.

2. **Paths are relative**: All scripts use `Path(__file__).parent` to find project root.

3. **To retrain after adding annotations**:
   ```bash
   python tools/train_keypoint_v2.py
   python tools/predict_keypoints_v2.py
   ```

4. **To restore manual-only annotations**:
   ```bash
   cp input_stickman_video/keypoint_annotations/annotations_manual_backup.json input_stickman_video/keypoint_annotations/annotations.json
   ```

5. **Annotation format** in JSON:
   ```json
   {
     "vid1_1.jpg": [[x, y, vis], [x, y, vis], ...],  // 18 keypoints
   }
   ```
   Where `vis`: 0=not visible, 2=visible (normalized to 0-1 during training)
