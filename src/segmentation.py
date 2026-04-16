import cv2
import numpy as np

def morphological_cleaning(binary_img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return cleaned

def keep_largest_contour(binary_img):
    contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary_img)

    if len(contours) == 0:
        return mask, None

    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return mask, largest_contour

def apply_mask(gray_img, mask):
    return cv2.bitwise_and(gray_img, gray_img, mask=mask)