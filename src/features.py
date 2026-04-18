"""Extraction de caracteristiques pour les regles et le SVM."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
except ImportError:  # pragma: no cover - fallback utile si scikit-image n'est pas installe.
    graycomatrix = None
    graycoprops = None
    local_binary_pattern = None

from .preprocessing import preprocess_image, read_grayscale_image
from .segmentation import region_summary, segment_dark_defects


def extract_intensity_features(image: np.ndarray) -> dict[str, float]:
    """Statistiques simples sur les niveaux de gris."""
    pixels = image.astype(np.float32).ravel()
    return {
        "intensity_mean": float(np.mean(pixels)),
        "intensity_std": float(np.std(pixels)),
        "intensity_min": float(np.min(pixels)),
        "intensity_max": float(np.max(pixels)),
        "intensity_median": float(np.median(pixels)),
        "intensity_p10": float(np.percentile(pixels, 10)),
        "intensity_p90": float(np.percentile(pixels, 90)),
    }


def extract_histogram_features(image: np.ndarray, bins: int = 16) -> dict[str, float]:
    """Histogramme normalise des niveaux de gris."""
    hist = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    return {f"hist_{i:02d}": float(value) for i, value in enumerate(hist)}


def extract_dark_bright_features(image: np.ndarray) -> dict[str, float]:
    """Ratios simples de pixels tres sombres et tres clairs."""
    return {
        "dark_pixel_ratio": float(np.mean(image < 48)),
        "bright_pixel_ratio": float(np.mean(image > 208)),
    }


def _fallback_glcm_props(quantized: np.ndarray, levels: int = 8) -> dict[str, float]:
    """Calcule des proprietes GLCM simples sans scikit-image."""
    offsets = [(0, 1), (1, 1), (1, 0), (1, -1), (0, 2), (2, 2), (2, 0), (2, -2)]
    matrices: list[np.ndarray] = []
    height, width = quantized.shape

    for dy, dx in offsets:
        y_start = max(0, -dy)
        y_end = min(height, height - dy)
        x_start = max(0, -dx)
        x_end = min(width, width - dx)

        first = quantized[y_start:y_end, x_start:x_end].ravel()
        second = quantized[y_start + dy : y_end + dy, x_start + dx : x_end + dx].ravel()
        matrix = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(matrix, (first, second), 1)
        matrix = matrix + matrix.T
        matrix = matrix / (matrix.sum() + 1e-12)
        matrices.append(matrix)

    i, j = np.indices((levels, levels))
    props = {
        "contrast": [],
        "dissimilarity": [],
        "homogeneity": [],
        "energy": [],
        "correlation": [],
    }

    for matrix in matrices:
        diff = i - j
        props["contrast"].append(float(np.sum(matrix * diff**2)))
        props["dissimilarity"].append(float(np.sum(matrix * np.abs(diff))))
        props["homogeneity"].append(float(np.sum(matrix / (1.0 + diff**2))))
        props["energy"].append(float(np.sqrt(np.sum(matrix**2))))

        mean_i = float(np.sum(i * matrix))
        mean_j = float(np.sum(j * matrix))
        std_i = float(np.sqrt(np.sum(((i - mean_i) ** 2) * matrix)))
        std_j = float(np.sqrt(np.sum(((j - mean_j) ** 2) * matrix)))
        if std_i > 0 and std_j > 0:
            correlation = float(np.sum((i - mean_i) * (j - mean_j) * matrix) / (std_i * std_j))
        else:
            correlation = 0.0
        props["correlation"].append(correlation)

    features: dict[str, float] = {}
    for prop, values in props.items():
        features[f"glcm_{prop}_mean"] = float(np.mean(values))
        features[f"glcm_{prop}_std"] = float(np.std(values))
    return features


def extract_glcm_features(image: np.ndarray) -> dict[str, float]:
    """Caracteristiques de texture GLCM sur une image quantifiee."""
    resized = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    quantized = (resized / 32).astype(np.uint8)

    if graycomatrix is None or graycoprops is None:
        return _fallback_glcm_props(quantized)

    glcm = graycomatrix(
        quantized,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=8,
        symmetric=True,
        normed=True,
    )

    features: dict[str, float] = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        values = graycoprops(glcm, prop)
        features[f"glcm_{prop}_mean"] = float(np.mean(values))
        features[f"glcm_{prop}_std"] = float(np.std(values))
    return features


def _fallback_uniform_lbp(image: np.ndarray) -> np.ndarray:
    """Approximation LBP uniforme pour P=8 et R=1 sans scikit-image."""
    center = image[1:-1, 1:-1]
    neighbors = [
        image[:-2, :-2],
        image[:-2, 1:-1],
        image[:-2, 2:],
        image[1:-1, 2:],
        image[2:, 2:],
        image[2:, 1:-1],
        image[2:, :-2],
        image[1:-1, :-2],
    ]
    bits = np.stack([(neighbor >= center).astype(np.uint8) for neighbor in neighbors], axis=0)
    transitions = np.sum(bits != np.roll(bits, shift=1, axis=0), axis=0)
    ones = np.sum(bits, axis=0)
    return np.where(transitions <= 2, ones, 9).astype(np.float32)


def extract_lbp_features(image: np.ndarray, points: int = 8, radius: int = 1) -> dict[str, float]:
    """Histogramme LBP pour decrire la texture locale."""
    if local_binary_pattern is None or points != 8 or radius != 1:
        lbp = _fallback_uniform_lbp(image)
    else:
        lbp = local_binary_pattern(image, P=points, R=radius, method="uniform")
    bins = points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
    return {f"lbp_{i:02d}": float(value) for i, value in enumerate(hist)}


def extract_segmentation_features(image: np.ndarray) -> dict[str, float]:
    """Caracteristiques geometriques issues du masque de defauts."""
    mask = segment_dark_defects(image)
    return region_summary(mask)


def extract_image_features(
    image: np.ndarray,
    size: tuple[int, int] = (300, 300),
) -> dict[str, float]:
    """Extrait toutes les caracteristiques numeriques d'une image."""
    processed = preprocess_image(image, size=size)
    features: dict[str, float] = {}
    features.update(extract_intensity_features(processed))
    features.update(extract_histogram_features(processed))
    features.update(extract_dark_bright_features(processed))
    features.update(extract_glcm_features(processed))
    features.update(extract_lbp_features(processed))
    features.update(extract_segmentation_features(processed))
    return features


def extract_features_from_path(
    path: str | Path,
    size: tuple[int, int] = (300, 300),
) -> dict[str, float]:
    """Charge une image et extrait ses caracteristiques."""
    image = read_grayscale_image(path)
    return extract_image_features(image, size=size)


def build_features_dataframe(
    index_df: pd.DataFrame,
    size: tuple[int, int] = (300, 300),
    max_images: int | None = None,
) -> pd.DataFrame:
    """Construit le dataset tabulaire a partir de l'index des images."""
    if max_images is not None:
        index_df = index_df.head(max_images).copy()

    rows: list[dict[str, object]] = []
    total = len(index_df)
    for i, row in index_df.reset_index(drop=True).iterrows():
        path = row["path"]
        feature_row = extract_features_from_path(path, size=size)
        feature_row.update(
            {
                "path": path,
                "filename": row["filename"],
                "split": row["split"],
                "class_folder": row["class_folder"],
                "label": int(row["label"]),
                "label_name": row["label_name"],
            }
        )
        rows.append(feature_row)

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"Extraction des caracteristiques: {i + 1}/{total}")

    return pd.DataFrame(rows)


def get_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Liste les colonnes numeriques utilisees pour le modele."""
    excluded = {"path", "filename", "split", "class_folder", "label", "label_name"}
    return [col for col in features_df.columns if col not in excluded]
