import cv2

def overlay_mask_on_gray(gray_img, mask, color=(255, 0, 0), alpha=0.45):
    rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    overlay = rgb.copy()
    overlay[mask > 0] = color
    blended = cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)
    return blended