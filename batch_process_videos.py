import cv2
import subprocess
import os

def crop_9_16_center(image):
    h, w = image.shape[:2]
    target_ratio = 9 / 16

    if w / h > target_ratio:
        new_w = int(h * target_ratio)
        new_h = h
    else:
        new_w = w
        new_h = int(w / target_ratio)

    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2

    return image[start_y:start_y+new_h, start_x:start_x+new_w]

def process_video(video_num):
    print(f"\n{'='*60}")
    print(f"Processing vid_{video_num}.mp4")
    print(f"{'='*60}")
    
    base_dir = "input_stickman_video"
    video_path = f"{base_dir}/vid_{video_num}.mp4"
    frames_dir = f"{base_dir}/frames_vid{video_num}"
    prepro_dir = f"{base_dir}/prepro_vid{video_num}"
    colored_prepro_dir = f"{base_dir}/colored_prepro_vid{video_num}"
    
    # Create directories
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(prepro_dir, exist_ok=True)
    os.makedirs(colored_prepro_dir, exist_ok=True)
    
    # Extract frames at 3fps
    print(f"Extracting frames at 3fps from {video_path}...")
    cmd = f'ffmpeg -i {video_path} -vf fps=3 {frames_dir}/%06d.jpg -y'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error extracting frames: {result.stderr}")
        return
    
    print(f"Frames extracted to {frames_dir}")
    
    # Count frames
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    total_frames = len(frame_files)
    print(f"Total frames: {total_frames}")
    
    # Process frames
    print("Processing frames...")
    x = 1
    for i in range(1, total_frames + 1):
        img_path = f"{frames_dir}/{i:06d}.jpg"
        
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if bw[1, 1] == 0:
            continue
        cropped_bw = crop_9_16_center(bw)
        if cv2.countNonZero(cropped_bw) > 265000:
            continue
        colored_crop = crop_9_16_center(img)
        cv2.imwrite(f"{prepro_dir}/{x}.jpg", cropped_bw)
        cv2.imwrite(f"{colored_prepro_dir}/{x}.jpg", colored_crop)
        x += 1
    
    print(f"Processed {x-1} frames for vid_{video_num}")
    print(f"Output saved to {prepro_dir} and {colored_prepro_dir}")

if __name__ == "__main__":
    for video_num in range(3, 22):
        try:
            process_video(video_num)
        except Exception as e:
            print(f"Error processing vid_{video_num}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("All videos processed!")
    print(f"{'='*60}")
