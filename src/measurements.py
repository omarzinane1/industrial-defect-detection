import cv2
import numpy as np

def summarize_contours(contours):
    if len(contours) == 0:
        return {
            "num_contours": 0,
            "total_area": 0,
            "max_area": 0,
            "mean_area": 0,
            "total_perimeter": 0
        }

    areas = [cv2.contourArea(c) for c in contours]
    perimeters = [cv2.arcLength(c, True) for c in contours]

    return {
        "num_contours": len(contours),
        "total_area": float(np.sum(areas)),
        "max_area": float(np.max(areas)),
        "mean_area": float(np.mean(areas)),
        "total_perimeter": float(np.sum(perimeters))
    }

def suspicious_area_ratio(piece_mask, suspicious_mask):
    piece_area = np.sum(piece_mask == 255)
    suspicious_area = np.sum(suspicious_mask == 255)

    if piece_area == 0:
        return 0

    return suspicious_area / piece_area