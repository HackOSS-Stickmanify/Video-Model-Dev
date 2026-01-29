# Keypoint Annotation Tools

Tools for annotating stickman images with OpenPose-style 18-keypoint body pose data.

## Overview

This toolkit allows you to:
1. **Manually annotate** a small set of images (~50-100)
2. **Train a model** on your annotations
3. **Auto-predict** keypoints on remaining images

## Files

```
tools/
├── annotate_keypoints.py      # Manual annotation GUI
├── train_keypoint_v2.py       # Train keypoint detector
├── predict_keypoints_v2.py    # Predict on unannotated images
└── checkpoints/               # Saved models
    └── best_keypoint_regressor.pt
```

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
