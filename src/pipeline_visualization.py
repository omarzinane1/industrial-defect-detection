"""Visualisations intermediaires du pipeline de traitement d'image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .features import extract_image_features
from .preprocessing import preprocess_image, resize_image, to_uint8
from .segmentation import (
    canny_edges,
    find_external_contours,
    otsu_threshold,
    region_summary,
    segment_dark_defects,
)


VIEW_ORIGINAL = "original"
VIEW_CONTRAST = "contrast"
VIEW_THRESHOLD = "threshold"
VIEW_SEGMENTATION = "segmentation"
VIEW_CONTOURS = "contours"
VIEW_SUSPECT_ZONES = "suspect_zones"
VIEW_FEATURES = "features"

VIEW_TITLES = {
    VIEW_ORIGINAL: "Vue originale",
    VIEW_CONTRAST: "Contraste",
    VIEW_THRESHOLD: "Seuillage",
    VIEW_SEGMENTATION: "Segmentation",
    VIEW_CONTOURS: "Contours",
    VIEW_SUSPECT_ZONES: "Zones suspectes",
    VIEW_FEATURES: "Statistiques / Features",
}


@dataclass
class PipelineVisualization:
    """Sorties visuelles et statistiques produites par le pipeline reel."""

    images: dict[str, np.ndarray]
    stats: dict[str, Any]


def build_pipeline_visualization(image: np.ndarray, include_stats: bool = True) -> PipelineVisualization:
    """Construit les principales vues intermediaires du pipeline.

    Les vues sont volontairement basees sur les memes briques que le reste du
    projet : pretraitement, contraste, seuillage, segmentation et contours.
    """
    original = resize_image(to_uint8(image), size=(300, 300))
    contrast = preprocess_image(image)
    _, threshold = otsu_threshold(contrast, invert=True)
    segmentation = segment_dark_defects(contrast)
    edges = canny_edges(contrast)
    contours = find_external_contours(segmentation)

    contour_view = cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_view, contours, contourIdx=-1, color=(0, 220, 120), thickness=1)

    zones_view = cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)
    overlay = zones_view.copy()
    overlay[segmentation > 0] = (30, 80, 230)
    zones_view = cv2.addWeighted(overlay, 0.45, zones_view, 0.55, 0)
    cv2.drawContours(zones_view, contours, contourIdx=-1, color=(0, 230, 140), thickness=1)

    stats = build_pipeline_statistics(image, segmentation=segmentation, edges=edges) if include_stats else {}

    return PipelineVisualization(
        images={
            VIEW_ORIGINAL: original,
            VIEW_CONTRAST: contrast,
            VIEW_THRESHOLD: threshold,
            VIEW_SEGMENTATION: segmentation,
            VIEW_CONTOURS: contour_view,
            VIEW_SUSPECT_ZONES: zones_view,
        },
        stats=stats,
    )


def build_pipeline_statistics(
    image: np.ndarray,
    segmentation: np.ndarray | None = None,
    edges: np.ndarray | None = None,
) -> dict[str, Any]:
    """Retourne un sous-ensemble lisible des caracteristiques utiles."""
    processed = preprocess_image(image)
    mask = segmentation if segmentation is not None else segment_dark_defects(processed)
    edge_image = edges if edges is not None else canny_edges(processed)

    features = extract_image_features(image)
    features.update(region_summary(mask))
    features["edge_density"] = float(np.count_nonzero(edge_image) / edge_image.size)

    return {
        "moyenne niveaux de gris": features.get("intensity_mean"),
        "ecart-type niveaux de gris": features.get("intensity_std"),
        "minimum niveaux de gris": features.get("intensity_min"),
        "maximum niveaux de gris": features.get("intensity_max"),
        "percentile 10": features.get("intensity_p10"),
        "percentile 90": features.get("intensity_p90"),
        "ratio pixels sombres": features.get("dark_pixel_ratio"),
        "ratio pixels clairs": features.get("bright_pixel_ratio"),
        "densite contours": features.get("edge_density"),
        "nombre regions suspectes": features.get("defect_count"),
        "aire suspecte": features.get("defect_area"),
        "ratio aire suspecte": features.get("defect_area_ratio"),
        "plus grande region suspecte": features.get("largest_defect_area"),
        "texture contraste": features.get("glcm_contrast_mean"),
        "texture dissimilarite": features.get("glcm_dissimilarity_mean"),
        "texture homogeneite": features.get("glcm_homogeneity_mean"),
    }


def format_pipeline_statistics(stats: dict[str, Any]) -> str:
    """Formate les statistiques pour une fenetre Tkinter."""
    if not stats:
        return "Aucune statistique disponible."

    lines = ["Statistiques du pipeline", ""]
    for key, value in stats.items():
        if value is None:
            continue
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.6f}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)
