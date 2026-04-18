"""Seuillage, contours et segmentation simple des defauts."""

from __future__ import annotations

import cv2
import numpy as np

from .filtering import apply_clahe, gaussian_blur


def otsu_threshold(image: np.ndarray, invert: bool = False) -> tuple[float, np.ndarray]:
    """Applique le seuillage d'Otsu et retourne le seuil et le masque."""
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    threshold_value, binary = cv2.threshold(image, 0, 255, mode + cv2.THRESH_OTSU)
    return threshold_value, binary


def adaptive_threshold(
    image: np.ndarray,
    block_size: int = 31,
    c: int = 3,
    invert: bool = True,
) -> np.ndarray:
    """Seuillage adaptatif utile lorsque l'eclairage n'est pas uniforme."""
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type,
        block_size,
        c,
    )


def canny_edges(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> np.ndarray:
    """Detecte les contours principaux avec Canny."""
    return cv2.Canny(image, low_threshold, high_threshold)


def morphological_clean(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """Nettoie un masque binaire par ouverture puis fermeture."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed


def remove_border_components(mask: np.ndarray) -> np.ndarray:
    """Supprime les regions connectees au bord de l'image."""
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape
    cleaned = np.zeros_like(binary, dtype=np.uint8)

    for label in range(1, num_labels):
        x, y, w, h, _ = stats[label]
        touches_border = x == 0 or y == 0 or (x + w) >= width or (y + h) >= height
        if not touches_border:
            cleaned[labels == label] = 255

    return cleaned


def remove_small_components(mask: np.ndarray, min_area: int = 25) -> np.ndarray:
    """Supprime les petites regions probablement dues au bruit."""
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary, dtype=np.uint8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def find_external_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Retourne les contours externes d'un masque binaire."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def contour_measurements(contours: list[np.ndarray]) -> list[dict[str, float]]:
    """Extrait des mesures simples pour chaque contour."""
    measurements: list[dict[str, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        x, y, w, h = cv2.boundingRect(contour)
        circularity = 0.0
        if perimeter > 0:
            circularity = float(4 * np.pi * area / (perimeter**2))
        measurements.append(
            {
                "area": area,
                "perimeter": perimeter,
                "x": float(x),
                "y": float(y),
                "width": float(w),
                "height": float(h),
                "circularity": circularity,
            }
        )
    return measurements


def segment_dark_defects(
    image: np.ndarray,
    min_area: int = 30,
    kernel_size: int = 3,
) -> np.ndarray:
    """Segmente simplement les zones sombres pouvant correspondre a des defauts.

    Les defauts de fonderie apparaissent souvent comme des regions plus sombres.
    On renforce donc le contraste, on applique Otsu inverse, puis on nettoie le
    masque pour enlever le fond et les petites regions isolees.
    """
    blurred = gaussian_blur(image, kernel_size=(5, 5))
    enhanced = apply_clahe(blurred)
    _, mask = otsu_threshold(enhanced, invert=True)
    mask = remove_border_components(mask)
    mask = morphological_clean(mask, kernel_size=kernel_size)
    mask = remove_small_components(mask, min_area=min_area)
    return mask


def mask_area_ratio(mask: np.ndarray) -> float:
    """Calcule le ratio de pixels blancs dans un masque binaire."""
    return float(np.count_nonzero(mask) / mask.size)


def region_summary(mask: np.ndarray) -> dict[str, float]:
    """Resume les regions segmentees: nombre, aire totale et plus grande aire."""
    binary = (mask > 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = [float(stats[label, cv2.CC_STAT_AREA]) for label in range(1, num_labels)]
    image_area = float(mask.shape[0] * mask.shape[1])

    return {
        "defect_count": float(len(areas)),
        "defect_area": float(sum(areas)),
        "defect_area_ratio": float(sum(areas) / image_area if image_area else 0.0),
        "largest_defect_area": float(max(areas) if areas else 0.0),
        "largest_defect_area_ratio": float(max(areas) / image_area if areas else 0.0),
    }
