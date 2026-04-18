"""Chargement et pretraitement des images en niveaux de gris."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .filtering import apply_clahe, gaussian_blur


def read_grayscale_image(path: str | Path) -> np.ndarray:
    """Charge une image en niveaux de gris au format uint8."""
    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image introuvable ou illisible: {image_path}")
    return image


def resize_image(image: np.ndarray, size: tuple[int, int] = (300, 300)) -> np.ndarray:
    """Redimensionne une image avec interpolation adaptee."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Convertit une image uint8 en float32 entre 0 et 1."""
    return image.astype(np.float32) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convertit proprement une image en uint8."""
    if image.dtype == np.uint8:
        return image

    image = np.asarray(image)
    if image.max() <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def preprocess_image(
    image: np.ndarray,
    size: tuple[int, int] = (300, 300),
    use_clahe: bool = True,
    blur_kernel: tuple[int, int] = (5, 5),
) -> np.ndarray:
    """Pipeline simple: resize, leger flou et amelioration de contraste."""
    resized = resize_image(to_uint8(image), size=size)
    blurred = gaussian_blur(resized, kernel_size=blur_kernel)
    if use_clahe:
        return apply_clahe(blurred)
    return blurred


def load_and_preprocess(
    path: str | Path,
    size: tuple[int, int] = (300, 300),
    use_clahe: bool = True,
) -> np.ndarray:
    """Charge une image puis applique le pretraitement principal."""
    image = read_grayscale_image(path)
    return preprocess_image(image, size=size, use_clahe=use_clahe)

