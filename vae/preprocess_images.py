"""
Preprocess images: resize to 480×640 portrait with padding.
Run this once before training for faster data loading.
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm


def resize_and_pad(
    img: Image.Image,
    target_size: tuple[int, int] = (480, 640),  # (width, height)
    pad_value: int = 255,  # White padding (255=white, 0=black)
) -> Image.Image:
    """
    Resize image to fit target height, then pad width to target.
    
    Args:
        img: Input PIL Image (grayscale)
        target_size: (width, height) tuple
        pad_value: Padding color (255=white, 0=black)
    
    Returns:
        Resized and padded image
    """
    target_w, target_h = target_size
    orig_w, orig_h = img.size
    
    # Scale to match target height
    scale = target_h / orig_h
    new_w = int(orig_w * scale)
    new_h = target_h
    
    img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Pad width if needed (center padding)
    if new_w < target_w:
        pad_left = (target_w - new_w) // 2
        padded = Image.new('L', (target_w, target_h), pad_value)
        padded.paste(img, (pad_left, 0))
        img = padded
    elif new_w > target_w:
        # Center crop width
        left = (new_w - target_w) // 2
        img = img.crop((left, 0, left + target_w, target_h))
    
    return img


def process_image(args):
    """Process a single image."""
    src_path, dst_path, target_size = args
    
    try:
        img = Image.open(src_path).convert('L')
        img = resize_and_pad(img, target_size)
        img.save(dst_path, quality=95)
        return True, src_path.name
    except Exception as e:
        return False, f"{src_path.name}: {e}"


def main():
    parser = argparse.ArgumentParser(description='Preprocess images for VAE training')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory with original images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed images')
    parser.add_argument('--width', type=int, default=480,
                        help='Target width')
    parser.add_argument('--height', type=int, default=640,
                        help='Target height')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of worker threads')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_size = (args.width, args.height)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect images
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = []
    for ext in extensions:
        image_paths.extend(input_dir.glob(f'*{ext}'))
        image_paths.extend(input_dir.glob(f'*{ext.upper()}'))
    
    image_paths = sorted(set(image_paths))
    print(f"Found {len(image_paths)} images")
    print(f"Target size: {target_size[0]}×{target_size[1]} (W×H)")
    
    # Prepare tasks
    tasks = []
    for src in image_paths:
        dst = output_dir / src.name
        tasks.append((src, dst, target_size))
    
    # Process in parallel
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_image, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing'):
            success, msg = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"Failed: {msg}")
    
    print(f"\nDone! Processed {success_count} images, {fail_count} failed")
    print(f"Output saved to: {output_dir}")


if __name__ == '__main__':
    main()
