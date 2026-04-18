"""Filtres et amelioration du contraste."""

from __future__ import annotations

import cv2
import numpy as np


def gaussian_blur(
    image: np.ndarray,
    kernel_size: tuple[int, int] = (5, 5),
    sigma: float = 0,
) -> np.ndarray:
    """Reduit le bruit avec un filtre gaussien."""
    return cv2.GaussianBlur(image, kernel_size, sigma)


def median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Reduit le bruit impulsionnel avec un filtre median."""
    return cv2.medianBlur(image, kernel_size)


def bilateral_denoise(
    image: np.ndarray,
    diameter: int = 7,
    sigma_color: float = 50,
    sigma_space: float = 50,
) -> np.ndarray:
    """Lisse l'image tout en preservant mieux les contours."""
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """Applique une egalisation globale d'histogramme."""
    return cv2.equalizeHist(image)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Ameliore localement le contraste avec CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def sharpen_image(image: np.ndarray) -> np.ndarray:
    """Renforce legerement les details de l'image."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, ddepth=-1, kernel=kernel)

