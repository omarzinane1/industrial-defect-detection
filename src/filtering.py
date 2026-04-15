import cv2

def apply_gaussian_filter(img, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(img, kernel_size, sigma)

def apply_median_filter(img, kernel_size=5):
    return cv2.medianBlur(img, kernel_size)