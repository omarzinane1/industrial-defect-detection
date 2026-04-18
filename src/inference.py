"""Pipelines de prediction reutilisables par l'interface graphique."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .features import extract_image_features
from .preprocessing import preprocess_image, read_grayscale_image
from .rules import RuleConfig, calibrate_rules_from_ok_samples, explain_rule_decision, predict_one_by_rules, rule_score
from .segmentation import canny_edges, region_summary, segment_dark_defects
from .utils import FEATURES_DIR, LABEL_TO_NAME, MODELS_DIR


METHOD_RULES = "Traitement d'image + règles"
METHOD_SVM = "Traitement d'image + SVM"


@dataclass
class PredictionResult:
    """Resultat normalise pour l'affichage dans l'interface."""

    method: str
    predicted_label: int
    predicted_name: str
    summary: str
    stats: dict[str, Any]
    confidence_label: str | None = None


@lru_cache(maxsize=1)
def _load_rule_config(features_path: Path = FEATURES_DIR / "casting_features.csv") -> RuleConfig:
    """Calibre les regles depuis les caracteristiques sauvegardees si disponibles."""
    if not features_path.exists():
        return RuleConfig()

    features_df = pd.read_csv(features_path)
    return calibrate_rules_from_ok_samples(features_df)


def _edge_density(processed_image: np.ndarray) -> float:
    """Calcule un ratio simple de pixels de contours."""
    edges = canny_edges(processed_image)
    return float(np.count_nonzero(edges) / edges.size)


def _basic_stats_from_features(features: dict[str, float]) -> dict[str, Any]:
    """Selectionne les statistiques les plus utiles pour l'affichage."""
    return {
        "moyenne niveaux de gris": features.get("intensity_mean"),
        "ecart-type niveaux de gris": features.get("intensity_std"),
        "ratio pixels sombres": features.get("dark_pixel_ratio"),
        "ratio pixels clairs": features.get("bright_pixel_ratio"),
        "nombre regions suspectes": features.get("defect_count"),
        "aire suspecte": features.get("defect_area"),
        "ratio aire suspecte": features.get("defect_area_ratio"),
        "plus grande region suspecte": features.get("largest_defect_area"),
        "texture dissimilarite": features.get("glcm_dissimilarity_mean"),
    }


def _extract_all_for_image(image: np.ndarray) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    """Prepare l'image et extrait les caracteristiques communes aux deux methodes."""
    processed = preprocess_image(image)
    features = extract_image_features(image)
    mask = segment_dark_defects(processed)
    segmentation_stats = region_summary(mask)

    # On synchronise explicitement les statistiques de segmentation visibles.
    features.update(segmentation_stats)
    features["edge_density"] = _edge_density(processed)
    return processed, features, _basic_stats_from_features(features)


def predict_with_rules(
    image: np.ndarray,
    config: RuleConfig | None = None,
) -> PredictionResult:
    """Applique uniquement l'approche traitement d'image + regles."""
    _, features, stats = _extract_all_for_image(image)
    config = config or _load_rule_config()
    predicted_label = predict_one_by_rules(features, config)
    score, reasons = rule_score(features, config)
    summary = explain_rule_decision(features, config)

    stats.update(
        {
            "score suspicion": score,
            "seuil score": config.min_rule_score,
            "densite contours": features.get("edge_density"),
            "details decision": ", ".join(reasons) if reasons else "mesures sous les seuils",
        }
    )

    return PredictionResult(
        method=METHOD_RULES,
        predicted_label=predicted_label,
        predicted_name=LABEL_TO_NAME[predicted_label],
        summary=summary,
        stats=stats,
        confidence_label=f"Score règles: {score:.1f}/{config.min_rule_score:.1f}",
    )


@lru_cache(maxsize=1)
def _load_svm_model(model_path: Path = MODELS_DIR / "svm_model.joblib") -> Any:
    """Charge le modele SVM en important scikit-learn seulement si necessaire."""
    if not model_path.exists():
        raise FileNotFoundError(f"Modele SVM introuvable: {model_path}")

    try:
        import joblib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("joblib n'est pas installe. Lancez: pip install -r requirements.txt") from exc

    try:
        return joblib.load(model_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("scikit-learn n'est pas installe. Lancez: pip install -r requirements.txt") from exc


def _expected_feature_columns(model: Any) -> list[str] | None:
    """Recupere l'ordre des caracteristiques attendu par le pipeline SVM."""
    for step_name in ["scaler", "svm"]:
        step = getattr(model, "named_steps", {}).get(step_name) if hasattr(model, "named_steps") else None
        feature_names = getattr(step, "feature_names_in_", None)
        if feature_names is not None:
            return list(feature_names)
    return None


def _feature_columns_from_saved_csv(features_path: Path = FEATURES_DIR / "casting_features.csv") -> list[str] | None:
    """Fallback pour retrouver les colonnes du SVM depuis le CSV de features."""
    if not features_path.exists():
        return None

    excluded = {"path", "filename", "split", "class_folder", "label", "label_name"}
    columns = pd.read_csv(features_path, nrows=1).columns
    return [column for column in columns if column not in excluded]


def _prepare_svm_input(features: dict[str, float], model: Any) -> pd.DataFrame:
    """Aligne les caracteristiques sur celles attendues par le modele sauvegarde."""
    expected_columns = _expected_feature_columns(model) or _feature_columns_from_saved_csv()
    if expected_columns is None:
        return pd.DataFrame([features])

    aligned = {column: float(features.get(column, 0.0)) for column in expected_columns}
    return pd.DataFrame([aligned], columns=expected_columns)


def _svm_score(model: Any, X_one: pd.DataFrame, predicted_label: int) -> str | None:
    """Retourne une information de confiance disponible pour le SVM."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_one)[0]
        return f"Probabilité estimée: {float(probabilities[predicted_label]):.3f}"

    if hasattr(model, "decision_function"):
        distance = model.decision_function(X_one)
        value = float(np.ravel(distance)[0])
        return f"Distance au classifieur: {value:.3f}"

    return None


def predict_with_svm(image: np.ndarray, model: Any | None = None) -> PredictionResult:
    """Applique uniquement l'approche traitement d'image + SVM."""
    _, features, stats = _extract_all_for_image(image)
    model = model or _load_svm_model()
    X_one = _prepare_svm_input(features, model)
    predicted_label = int(model.predict(X_one)[0])
    confidence = _svm_score(model, X_one, predicted_label)

    stats.update(
        {
            "modele": str(MODELS_DIR / "svm_model.joblib"),
            "nombre caracteristiques utilisees": int(X_one.shape[1]),
            "densite contours": features.get("edge_density"),
        }
    )

    summary = f"Prediction SVM: {LABEL_TO_NAME[predicted_label]}"
    if confidence:
        summary = f"{summary} ({confidence})"

    return PredictionResult(
        method=METHOD_SVM,
        predicted_label=predicted_label,
        predicted_name=LABEL_TO_NAME[predicted_label],
        summary=summary,
        stats=stats,
        confidence_label=confidence,
    )


def predict_image(image: np.ndarray, method: str) -> PredictionResult:
    """Point d'entree unique pour predire une image deja chargee."""
    if method == METHOD_RULES:
        return predict_with_rules(image)
    if method == METHOD_SVM:
        return predict_with_svm(image)
    raise ValueError(f"Methode inconnue: {method}")


def predict_image_path(path: str | Path, method: str) -> PredictionResult:
    """Charge une image depuis un chemin puis applique la methode choisie."""
    image = read_grayscale_image(path)
    return predict_image(image, method)
