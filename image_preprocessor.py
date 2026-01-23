import cv2

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

for i in range(1, 203):
    img = cv2.imread(f"input_stickman_video/frames_vid1/000{i:>3}.jpg".replace(' ', '0'))
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
    cv2.imwrite(f"input_stickman_video/prepro_vid1/{i}.jpg", cropped_bw)