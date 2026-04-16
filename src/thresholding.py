import cv2

def apply_binary_threshold(img, thresh_value=127):
    _, binary = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)
    return binary

def apply_binary_inverse_threshold(img, thresh_value=127):
    _, binary_inv = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY_INV)
    return binary_inv

def apply_otsu_threshold(img):
    thresh_value, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu, thresh_value

def apply_adaptive_mean_threshold(img, block_size=11, c=2):
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c
    )

def apply_adaptive_gaussian_threshold(img, block_size=11, c=2):
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )