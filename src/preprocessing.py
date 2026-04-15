import cv2

def preprocess_image(img, target_size=(256, 256)):
    img_resized = cv2.resize(img, target_size)
    img_normalized = cv2.normalize(img_resized, None, 0, 255, cv2.NORM_MINMAX)
    return img_normalized