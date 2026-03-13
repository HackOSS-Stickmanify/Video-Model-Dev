# Video Model Dev

Fast Pose→Video Generation System (480p Grayscale)

## Project Goal

Build a **fast, simple pose-to-video generation system** that outputs **480p black-and-white video**, prioritizing motion coherence and speed over photorealism. Designed for a single RTX 3090/A5000 (24GB VRAM).

**Output style:** Stylized/lo-fi, silhouette-driven stickman animations with strong pose adherence and smooth limb motion.

---

## ✨ Recent Improvements

- **Stage 2: Image Diffusion** - Pose-conditioned latent diffusion UNet with DDIM sampling
- **LPIPS Perceptual Loss** - Optional perceptual loss for better reconstruction quality
- **Learning Rate Scheduler** - Cosine decay with linear warmup for better convergence
- **Validation & Early Stopping** - Proper train/val split with early stopping
- **EMA (Exponential Moving Average)** - Smoother model weights for inference
- **YAML Config Support** - Reproducible experiments with config files
- **Inference Utilities** - Easy-to-use API for encoding, decoding, and reconstruction
- **Pose Heatmap Generation** - Ready for Stage 2 diffusion conditioning
- **Modern Packaging** - `pyproject.toml` with proper dependencies

---

## Architecture Overview

### Training Pipeline (3 Stages)

```
Stage 1: VAE Training (✓ Complete)
    Input: 1×640×480 grayscale images
    Output: Frozen VAE encoder/decoder
    Latent: 4×80×60 (×8 downsample)
    Features: EMA, LPIPS, LR scheduling, validation

Stage 2: Image Diffusion (✓ Complete)
    Train pose-conditioned latent diffusion UNet on single frames
    Conditioning: Pose heatmaps concatenated with latent input
    Sampling: DDIM/DPM-Solver for fast inference (50 steps)

Stage 3: Video Fine-tuning  
    Add temporal conv blocks to UNet
    Train on short clips (24-48 frames, 2-4 seconds)
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Color | **Grayscale (1-ch) or RGB (3-ch)** | Grayscale for speed, RGB for color preservation |
| Resolution | **480×640 (portrait)** | Matches source stickman videos |
| VAE type | **Spatial/convolutional** | Preserves spatial structure for diffusion |
| Normalization | **GroupNorm** | Works with any batch size |
| VAE loss | **L1 + KL + LPIPS** | Sharp reconstructions with perceptual quality |
| Temporal modeling | **1D temporal convs only** | Lightweight, at mid+bottleneck blocks only |

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd Video-Model-Dev

# Install with pip (recommended)
pip install -e .

# Or install dependencies directly
pip install -r vae/requirements.txt
```

### Dependencies

- Python >= 3.10
- PyTorch >= 2.0
- torchvision >= 0.15
- OpenCV, Pillow, numpy
- TensorBoard, tqdm
- LPIPS (for perceptual loss)
- PyYAML (for config files)

---

## Directory Structure

```
Video-Model-Dev/
├── README.md                          # This file
├── pyproject.toml                     # Modern Python packaging
├── .gitignore
│
├── configs/                           # Configuration files
│   ├── vae_default.yaml               # Default VAE training config
│   ├── vae_color.yaml                 # Color VAE training config
│   ├── diffusion_default.yaml         # Default diffusion config (grayscale)
│   └── diffusion_color.yaml           # Color diffusion config
│
├── vae/                               # VAE module (Stage 1)
│   ├── __init__.py
│   ├── model.py                       # VAE, EMA, VAELoss
│   ├── dataset.py                     # Dataset with augmentation
│   ├── train_vae.py                   # Training with scheduler, EMA, validation
│   ├── inference.py                   # Inference utilities
│   ├── evaluate.py                    # Model evaluation & visualization
│   └── preprocess_images.py           # Resize/pad to 480×640
│
├── diffusion/                         # Diffusion module (Stage 2)
│   ├── __init__.py
│   ├── unet.py                        # UNet with pose conditioning
│   ├── scheduler.py                   # DDPM/DDIM noise scheduler
│   ├── dataset.py                     # Latent + pose heatmap dataset
│   ├── train_diffusion.py             # Training script
│   └── inference.py                   # Sampling utilities
│
├── tools/                             # Keypoint & pose tools
│   ├── __init__.py                    # Constants and utilities
│   ├── annotate_keypoints.py          # Manual annotation GUI
│   ├── train_keypoint.py              # Train keypoint prediction model
│   ├── predict_keypoints.py           # Batch predict keypoints
│   ├── render_openpose_skeleton.py    # Render color-coded skeletons
│   ├── pose_heatmaps.py               # Generate pose heatmaps (Stage 2)
│   ├── replace_background_white.py    # Background replacement utility
│   ├── group_similar_images.py        # Image similarity grouping
│   └── README.md                      # Tools documentation
│
├── tests/                             # Unit tests
│   ├── __init__.py
│   └── test_vae.py                    # VAE model tests
│
├── input_stickman_video/              # Dataset
│   ├── all_bw_images/                 # Original B&W frames
│   ├── all_bw_images_480p/            # Preprocessed to 480×640
│   ├── all_colored_images/            # Color versions
│   ├── skeleton_images/               # Rendered skeletons
│   ├── keypoint_annotations/          # Pose keypoint data
│   └── README.md
│
└── checkpoints/                       # All model checkpoints
    ├── vae/                           # VAE checkpoints (by run)
    ├── diffusion/                     # Diffusion checkpoints (by run)
    └── keypoint/                      # Keypoint model checkpoints
```

---

## Stage 1: VAE Training

### Quick Start

```bash
# Using default config (grayscale)
cd vae
python train_vae.py --data_dir ../input_stickman_video/all_bw_images_480p --preprocessed

# Using config file (recommended)
python train_vae.py --config ../configs/vae_default.yaml

# With LPIPS perceptual loss
python train_vae.py --config ../configs/vae_default.yaml --use_lpips

# Train color VAE (RGB, 3 channels)
python train_vae.py --config ../configs/vae_color.yaml
```

### Model Architecture

The VAE supports both grayscale (1 channel) and color (3 channel RGB) images:

```python
# Grayscale VAE (default)
# Encoder: 1×640×480 → 4×80×60
model = VAE(in_channels=1, z_channels=4)

# Color VAE
# Encoder: 3×640×480 → 8×80×60
model = VAE(in_channels=3, z_channels=8)

# Common settings
channels: [32, 64, 128]  # base_channels=32, mults=(1,2,4)
downsample: ×8 (3 downsamples)
activation: SiLU
normalization: GroupNorm(8)
```

### Training Features

| Feature | Description |
|---------|-------------|
| **Mixed Precision** | FP16 training with GradScaler for 2x speed |
| **LR Scheduler** | Cosine decay with linear warmup |
| **Validation** | Automatic train/val split with early stopping |
| **EMA** | Exponential Moving Average for better inference |
| **LPIPS Loss** | Optional perceptual loss for quality |
| **Gradient Clipping** | Stability with configurable clip value |
| **Config Files** | YAML-based reproducible experiments |

### Training Config

Two config files are provided:

| Config | Use Case |
|--------|----------|
| `configs/vae_default.yaml` | Grayscale images (1 channel) |
| `configs/vae_color.yaml` | RGB color images (3 channels) |

Create a custom config by copying the appropriate template:

```yaml
# Key settings
data_dir: "../input_stickman_video/all_bw_images_480p"
color: false  # Set to true for RGB (3 channels)
batch_size: 8  # Reduce to 4 for color
lr: 0.0001
num_steps: 50000  # Increase to 75000 for color

# Model (larger latent for color)
z_channels: 4  # Use 8 for color images

# Loss
kl_weight: 0.01
use_lpips: false  # Recommended true for color

# EMA
use_ema: true
ema_decay: 0.999

# Early stopping
early_stopping: true
patience: 10
```

#### Color VAE Tips

- Use `z_channels: 8` (larger latent space for 3x more input channels)
- Enable `use_lpips: true` for better perceptual quality
- Reduce `batch_size` to 4 (3x more memory per image)
- Increase `num_steps` to 75000 (more to learn)
- Use `grad_accum: 2` to simulate larger effective batch size

### Inference

```python
from vae.inference import VAEInference

# Load trained model
vae = VAEInference.from_checkpoint("checkpoints/vae_best.pt")

# Reconstruct an image
recon = vae.reconstruct("image.jpg")
recon.save("reconstruction.png")

# Encode to latent space
latent = vae.encode("image.jpg")  # Shape: (1, 4, 80, 60)

# Interpolate between images
frames = vae.interpolate("image1.jpg", "image2.jpg", steps=10)

# Process entire directory
vae.process_directory("input/", "output/", save_latents=True)
```

Command-line inference:

```bash
# Reconstruct images
python vae/inference.py reconstruct input.jpg -c checkpoints/vae/run_name/vae_best.pt -o output/

# Interpolate between two images
python vae/inference.py interpolate img1.jpg img2.jpg -c checkpoints/vae/run_name/vae_best.pt --steps 20

# Sample random images from prior
python vae/inference.py sample -c checkpoints/vae/run_name/vae_best.pt -n 10 -o samples/

# Evaluate model (auto-detects grayscale vs color from checkpoint)
python vae/evaluate.py reconstruct -c checkpoints/vae/run_name/vae_best.pt --num_samples 10

# Force grayscale or color mode
python vae/evaluate.py reconstruct -c checkpoint.pt --grayscale
python vae/evaluate.py reconstruct -c checkpoint.pt --color

# Interpolate between images
python vae/evaluate.py interpolate img1.jpg img2.jpg -c checkpoints/vae/run_name/vae_best.pt
```

### Monitoring

```bash
tensorboard --logdir checkpoints/vae/run_name/logs --port 6006
```

---

## Stage 2: Image Diffusion

### Quick Start

```bash
# Step 1: Pre-encode images to latents (one-time)
cd diffusion
python dataset.py \
    --vae_checkpoint ../checkpoints/vae/run_name/vae_best.pt \
    --image_dir ../input_stickman_video/all_bw_images_480p \
    --annotations ../input_stickman_video/keypoint_annotations/annotations.json \
    --output_dir ../input_stickman_video/latents

# Step 2: Train diffusion model
python train_diffusion.py \
    --latent_dir ../input_stickman_video/latents \
    --annotations ../input_stickman_video/keypoint_annotations/annotations.json \
    --z_channels 4

# Or use config file (recommended)
python train_diffusion.py --config ../configs/diffusion_default.yaml \
    --vae_checkpoint ../checkpoints/vae/run_name/vae_best.pt

# For color VAE
python train_diffusion.py --config ../configs/diffusion_color.yaml \
    --vae_checkpoint ../checkpoints/vae/color_run/vae_best.pt
```

### Model Architecture

```python
# UNet for latent diffusion
from diffusion.unet import UNetSimple

model = UNetSimple(
    in_channels=4,        # Latent channels (4 grayscale, 8 color)
    base_channels=128,    # Base feature channels
    channel_mults=(1,2,4),# Progressive channel scaling
    num_res_blocks=2,     # ResBlocks per level
    attention_levels=(1,2),# Self-attention at levels 1 & 2
    pose_channels=35,     # 18 keypoints + 17 limbs
)
# Input: 4×80×60 noisy latent + 35×80×60 pose heatmap
# Output: 4×80×60 predicted noise
```

### Training Features

| Feature | Description |
|---------|-------------|
| **DDPM Training** | Predict noise at random timesteps |
| **Cosine Schedule** | Better noise distribution than linear |
| **Mixed Precision** | FP16 training with GradScaler |
| **EMA** | Exponential Moving Average for inference |
| **DDIM Sampling** | Fast 50-step inference (vs 1000 DDPM) |
| **Pose Conditioning** | Concatenate pose heatmaps with input |

### Inference

```python
from diffusion.inference import DiffusionInference

# Load trained models
inference = DiffusionInference.from_checkpoints(
    diffusion_checkpoint='checkpoints/diffusion/run_name/diffusion_best.pt',
    vae_checkpoint='checkpoints/vae/run_name/vae_best.pt',
)

# Generate from keypoints
keypoints = [[x, y, vis], ...]  # 18 OpenPose keypoints
image = inference.generate_from_keypoints(keypoints, num_steps=50)
image.save('generated.png')

# Generate from pose heatmap
heatmap = torch.randn(35, 80, 60)  # Or load from file
image = inference.generate_from_heatmap(heatmap)

# Batch generate from annotations
inference.generate_batch_from_annotations(
    'annotations.json',
    'output/',
    num_steps=50
)
```

Command-line inference:

```bash
# Generate images from annotations
python diffusion/inference.py sample \
    -d checkpoints/diffusion/run_name/diffusion_best.pt \
    -v checkpoints/vae/run_name/vae_best.pt \
    -a input_stickman_video/keypoint_annotations/annotations.json \
    -o generated_images/
```

### Pose Heatmap Generation

Generate conditioning heatmaps for diffusion training:

```bash
# Generate heatmaps from annotations
python tools/pose_heatmaps.py \
    --annotations ../input_stickman_video/keypoint_annotations/annotations.json \
    --output_dir ../input_stickman_video/pose_heatmaps \
    --height 80 --width 60 \
    --sigma 2.0 \
    --include_limbs
```

Python API:

```python
from tools.pose_heatmaps import PoseHeatmapGenerator

# Create generator (matches VAE latent size)
generator = PoseHeatmapGenerator(output_size=(80, 60), sigma=2.0)

# Generate heatmaps from keypoints
keypoints = [[x, y, vis], ...]  # 18 OpenPose keypoints
heatmaps = generator.generate(
    keypoints,
    original_size=(480, 640),
    include_limbs=True
)
# Shape: (35, 80, 60) - 18 keypoints + 17 limbs
```

---

## Tools & Utilities

### Keypoint Annotation

```bash
# Manual annotation GUI
python tools/annotate_keypoints.py
```

**Controls:**
| Key | Action |
|-----|--------|
| Left-click | Place keypoint |
| Right-click | Skip (not visible) |
| U | Undo |
| C | Copy from previous |
| S | Save |
| X | Delete image |
| N/→ | Next |
| P/← | Previous |

### Keypoint Model Training

```bash
# Train on manual annotations
python tools/train_keypoint.py --backbone resnet34 --batch-size 64

# Predict on all images
python tools/predict_keypoints.py --batch-size 64
```

### Skeleton Rendering

```bash
# Render OpenPose-style skeletons
python tools/render_openpose_skeleton.py
```

---

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=vae --cov=tools --cov-report=html
```

---

## Dataset Info

- **509 grayscale stickman frames** from 21 video clips
- Original: 405×720 (portrait, 9:16)
- Preprocessed: 480×640 (padded with black bars)
- All 509 images annotated with 18 OpenPose keypoints

### Pose Annotations

| Metric | Value |
|--------|-------|
| Total images | 509 |
| Manual annotations | 126 |
| Model predictions | 383 |
| Keypoints per image | 18 (OpenPose body) |
| Prediction error | ~5.7px on 256×256 |

---

## Hardware Requirements

- **GPU:** RTX 3090 or A5000 (24GB VRAM)
- **Batch size:** 8 (with AMP)
- **Training time:** ~4.5 hours for 50k VAE steps

---

## Quick Commands

```bash
# === Setup ===
pip install -e .

# === VAE Training (Grayscale) ===
cd vae
python train_vae.py --config ../configs/vae_default.yaml

# With LPIPS (better quality, slower)
python train_vae.py --config ../configs/vae_default.yaml --use_lpips

# === VAE Training (Color) ===
python train_vae.py --config ../configs/vae_color.yaml

# Resume training
python train_vae.py --config ../configs/vae_default.yaml --resume ../checkpoints/vae/run_name/vae_best.pt

# === Inference ===
python inference.py reconstruct ../input_stickman_video/all_bw_images_480p/ \
    -c ../checkpoints/vae/run_name/vae_best.pt \
    -o ../output/reconstructions/

# === Evaluation (auto-detects color mode) ===
python evaluate.py reconstruct -c ../checkpoints/vae/run_name/vae_best.pt

# === Keypoint Tools ===
cd tools
python annotate_keypoints.py  # Manual annotation
python train_keypoint.py      # Train model
python predict_keypoints.py   # Batch prediction
python render_openpose_skeleton.py  # Render skeletons

# === Pose Heatmaps ===
python pose_heatmaps.py --include_limbs

# === Stage 2: Diffusion Training ===
cd diffusion

# Pre-encode images to latents
python dataset.py \
    --vae_checkpoint ../checkpoints/vae/run_name/vae_best.pt \
    --image_dir ../input_stickman_video/all_bw_images_480p \
    --annotations ../input_stickman_video/keypoint_annotations/annotations.json \
    --output_dir ../input_stickman_video/latents

# Train diffusion (grayscale)
python train_diffusion.py --config ../configs/diffusion_default.yaml \
    --vae_checkpoint ../checkpoints/vae/run_name/vae_best.pt

# Train diffusion (color)
python train_diffusion.py --config ../configs/diffusion_color.yaml \
    --vae_checkpoint ../checkpoints/vae/color_run/vae_best.pt

# Generate images from poses
python inference.py sample \
    -d ../checkpoints/diffusion/run_name/diffusion_best.pt \
    -v ../checkpoints/vae/run_name/vae_best.pt \
    -a ../input_stickman_video/keypoint_annotations/annotations.json \
    -o ../generated_images/

# === Testing ===
pytest tests/ -v
```

---

## Roadmap

- [x] **Stage 1:** VAE Training
  - [x] Basic VAE with L1 + KL loss
  - [x] LPIPS perceptual loss
  - [x] EMA for better inference
  - [x] LR scheduler with warmup
  - [x] Validation with early stopping
  - [x] Config file support
  - [x] Inference utilities

- [x] **Stage 2:** Image Diffusion
  - [x] Pose heatmap generation
  - [x] Latent diffusion UNet
  - [x] Pose conditioning (concatenation)
  - [x] DDPM training loop
  - [x] DDIM sampling (50 steps)
  - [x] EMA for better inference
  - [x] Config file support
  - [x] Inference utilities

- [ ] **Stage 3:** Video Fine-tuning
  - [ ] Temporal conv blocks
  - [ ] Clip training
  - [ ] Video inference pipeline

---

## License

MIT License

---

## Acknowledgments

- OpenPose for keypoint format
- Stable Diffusion for VAE architecture inspiration
- LPIPS for perceptual loss