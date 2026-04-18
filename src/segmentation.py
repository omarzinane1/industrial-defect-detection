import cv2
import numpy as np

def detect_piece_circle(img_prepared, dp=1.2, min_dist=80, param1=100, param2=30,
                        min_radius=70, max_radius=120):
    blurred = cv2.GaussianBlur(img_prepared, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    H, W = img_prepared.shape
    img_cx, img_cy = W / 2, H / 2

    best_circle = None
    best_score = -1

    for x, y, r in circles:
        dist = np.sqrt((x - img_cx) ** 2 + (y - img_cy) ** 2)
        score = r - 0.5 * dist
        if score > best_score:
            best_score = score
            best_circle = (x, y, r)

    return best_circle

def segment_main_object_circle(img_prepared):
    circle = detect_piece_circle(img_prepared)

    if circle is None:
        H, W = img_prepared.shape
        x, y = W // 2, H // 2
        r = int(0.34 * min(H, W))
        mask = np.zeros_like(img_prepared)
        cv2.circle(mask, (x, y), r, 255, thickness=-1)

        return {
            "mask": mask,
            "center": (x, y),
            "radius": r,
            "selected_mode": "FALLBACK_CIRCLE"
        }

    x, y, r = circle
    mask = np.zeros_like(img_prepared)
    cv2.circle(mask, (x, y), r, 255, thickness=-1)

    return {
        "mask": mask,
        "center": (x, y),
        "radius": r,
        "selected_mode": "HOUGH_CIRCLE"
    }

def apply_mask(gray_img, mask):
    return cv2.bitwise_and(gray_img, gray_img, mask=mask)