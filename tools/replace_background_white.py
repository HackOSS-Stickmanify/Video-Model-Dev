"""
Replace background color with white in stickman images
Processes images from lora_training_colored_images folder
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def remove_background_to_white(image, threshold=240, kernel_size=5, use_color_detection=True):
    """
    Replace background with white using color detection or brightness-based masking.
    
    Args:
        image: BGR image
        threshold: Brightness threshold (0-255), used as fallback
        kernel_size: Morphological operation kernel size for cleanup
        use_color_detection: If True, detect background color from edges
    
    Returns:
        Image with white background
    """
    h, w = image.shape[:2]
    
    if use_color_detection:
        # Method 1: Detect background color from image edges
        # Sample pixels from the border (outer 5% of image)
        border_size = max(int(h * 0.05), int(w * 0.05), 5)
        
        # Get border pixels
        top_border = image[:border_size, :].reshape(-1, 3)
        bottom_border = image[-border_size:, :].reshape(-1, 3)
        left_border = image[:, :border_size].reshape(-1, 3)
        right_border = image[:, -border_size:].reshape(-1, 3)
        
        border_pixels = np.vstack([top_border, bottom_border, left_border, right_border])
        
        # Get the most common color (mode) from border
        # Use median as a robust estimate of background color
        bg_color = np.median(border_pixels, axis=0).astype(np.uint8)
        
        # Create mask based on color similarity to background
        color_diff = np.sqrt(np.sum((image.astype(np.float32) - bg_color.astype(np.float32))**2, axis=2))
        
        # Adaptive threshold based on image variation
        threshold_color = np.percentile(color_diff, 25)  # Bottom 25% similar to background
        mask = color_diff < max(threshold_color, 30)  # At least 30 color distance
        
    else:
        # Method 2: Brightness-based (original method)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = gray > threshold
    
    # Morphological operations to clean up mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Create output image
    result = image.copy()
    result[mask > 0] = [255, 255, 255]  # Set background to white
    
    return result


def process_images(input_dir, output_dir=None, threshold=240, kernel_size=5, preview=False, use_color_detection=True):
    """
    Process all images in directory to replace background with white.
    
    Args:
        input_dir: Input directory with images
        output_dir: Output directory (if None, overwrites original)
        threshold: Brightness threshold for background detection
        kernel_size: Morphological kernel size
        preview: If True, show before/after for first image
        use_color_detection: If True, use color detection method
    """
    input_path = Path(input_dir)
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path
    
    # Find all image files
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    image_files = sorted(set(image_files))
    
    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Method: {'Color Detection' if use_color_detection else 'Brightness Threshold'}")
    print(f"Found {len(image_files)} images")
    print(f"Threshold: {threshold}, Kernel size: {kernel_size}")
    print(f"Output: {'Overwriting originals' if not output_dir else output_path}")
    
    # Preview first image if requested
    if preview and len(image_files) > 0:
        print("\nPreviewing first image...")
        first_img = cv2.imread(str(image_files[0]))
        result = remove_background_to_white(first_img, threshold, kernel_size, use_color_detection)
        
        # Show side by side
        combined = np.hstack([first_img, result])
        cv2.imshow('Before (left) vs After (right) - Press any key to continue', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        response = input("\nContinue with these settings? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Process all images
    print("\nProcessing images...")
    for img_path in tqdm(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Failed to load {img_path}")
            continue
        result = remove_background_to_white(img, threshold, kernel_size, use_color_detection)
        
        # Save to output
        output_file = output_path / img_path.name
        cv2.imwrite(str(output_file), result)
    
    print(f"\nDone! Processed {len(image_files)} images")


def main():
    parser = argparse.ArgumentParser(description='Replace background with white in stickman images')
    
    parser.add_argument('--input_dir', type=str, 
                        default='input_stickman_video/lora_training_colored_images',
                        help='Input directory with images')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (if not set, overwrites original images)')
    parser.add_argument('--threshold', type=int, default=240,
                        help='Brightness threshold for background detection (0-255, default: 240)')
    parser.add_argument('--kernel_size', type=int, default=5,
                        help='Morphological kernel size for cleanup (default: 5)')
    parser.add_argument('--use_brightness', action='store_true',
                        help='Use brightness method instead of color detection')
    parser.add_argument('--preview', action='store_true',
                        help='Preview first image before processing all')
    
    args = parser.parse_args()
    
    process_images(
        args.input_dir,
        args.output_dir,
        args.threshold,
        args.kernel_size,
        args.preview,
        use_color_detection=not args.use_brightness
    )


if __name__ == '__main__':
    main()
