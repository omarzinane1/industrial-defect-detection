"""Decision par regles pour classer une image OK ou Defective."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .utils import LABEL_TO_NAME


@dataclass
class RuleConfig:
    """Seuils utilises par l'approche classique amelioree."""

    max_defect_area_ratio: float = 0.40
    max_largest_defect_area_ratio: float = 0.25
    max_defect_count: float = 13.0
    max_intensity_mean: float = 154.0
    max_intensity_p90: float = 214.0
    min_dark_pixel_ratio: float = 0.118
    min_hist_01_ratio: float = 0.048
    min_defect_count: float = 9.0
    min_texture_dissimilarity: float = 0.419
    min_rule_score: float = 3.0


def calibrate_rules_from_ok_samples(
    features_df: pd.DataFrame,
    quantile: float = 0.95,
    safety_factor: float = 1.2,
    brightness_quantile: float = 0.35,
    dark_quantile: float = 0.75,
    count_quantile: float = 0.80,
    texture_quantile: float = 0.75,
    min_rule_score: float = 3.0,
) -> RuleConfig:
    """Calibre les seuils a partir des images OK du jeu d'entrainement.

    La premiere version etait tres stricte: elle attendait des zones segmentees
    tres grandes avant de predire un defaut. Ici, on garde une logique par
    regles, mais on cumule plusieurs indices simples pour mieux detecter les
    defauts discrets.
    """
    working_df = features_df.copy()
    if "dark_pixel_ratio" not in working_df.columns and {"hist_00", "hist_01", "hist_02"}.issubset(
        working_df.columns
    ):
        working_df["dark_pixel_ratio"] = working_df[["hist_00", "hist_01", "hist_02"]].sum(axis=1)

    ok_train = working_df[(working_df["split"] == "train") & (working_df["label"] == 0)]
    if ok_train.empty:
        return RuleConfig()

    def quantile_or_default(column: str, q: float, default: float) -> float:
        if column not in ok_train.columns:
            return default
        return float(ok_train[column].quantile(q))

    return RuleConfig(
        max_defect_area_ratio=quantile_or_default(
            "defect_area_ratio", quantile, RuleConfig.max_defect_area_ratio
        )
        * safety_factor,
        max_largest_defect_area_ratio=quantile_or_default(
            "largest_defect_area_ratio", quantile, RuleConfig.max_largest_defect_area_ratio
        )
        * safety_factor,
        max_defect_count=quantile_or_default("defect_count", quantile, RuleConfig.max_defect_count)
        * safety_factor,
        max_intensity_mean=quantile_or_default(
            "intensity_mean", brightness_quantile, RuleConfig.max_intensity_mean
        ),
        max_intensity_p90=quantile_or_default("intensity_p90", brightness_quantile, RuleConfig.max_intensity_p90),
        min_dark_pixel_ratio=quantile_or_default(
            "dark_pixel_ratio", dark_quantile, RuleConfig.min_dark_pixel_ratio
        ),
        min_hist_01_ratio=quantile_or_default("hist_01", dark_quantile, RuleConfig.min_hist_01_ratio),
        min_defect_count=quantile_or_default("defect_count", count_quantile, RuleConfig.min_defect_count),
        min_texture_dissimilarity=quantile_or_default(
            "glcm_dissimilarity_mean", texture_quantile, RuleConfig.min_texture_dissimilarity
        ),
        min_rule_score=min_rule_score,
    )


def _get_value(row: pd.Series | dict, key: str, default: float = 0.0) -> float:
    """Lit une valeur numerique depuis une ligne Pandas ou un dictionnaire."""
    if key not in row:
        return default
    value = row[key]
    if pd.isna(value):
        return default
    return float(value)


def _dark_pixel_ratio(row: pd.Series | dict) -> float:
    """Retourne le ratio de pixels sombres, avec compatibilite anciens CSV."""
    if "dark_pixel_ratio" in row and not pd.isna(row["dark_pixel_ratio"]):
        return float(row["dark_pixel_ratio"])
    return sum(_get_value(row, column) for column in ["hist_00", "hist_01", "hist_02"])


def rule_score(row: pd.Series | dict, config: RuleConfig | None = None) -> tuple[float, list[str]]:
    """Calcule un score de suspicion interpretable."""
    config = config or RuleConfig()
    score = 0.0
    reasons: list[str] = []

    if _get_value(row, "intensity_mean") <= config.max_intensity_mean:
        score += 2
        reasons.append("image globalement sombre")
    if _get_value(row, "intensity_p90") <= config.max_intensity_p90:
        score += 1
        reasons.append("peu de pixels tres clairs")

    if _dark_pixel_ratio(row) >= config.min_dark_pixel_ratio:
        score += 2
        reasons.append("ratio de pixels sombres eleve")
    if _get_value(row, "hist_01") >= config.min_hist_01_ratio:
        score += 1
        reasons.append("histogramme concentre dans les faibles intensites")

    if _get_value(row, "defect_count") >= config.min_defect_count:
        score += 1
        reasons.append("plusieurs regions suspectes detectees")
    if _get_value(row, "glcm_dissimilarity_mean") >= config.min_texture_dissimilarity:
        score += 1
        reasons.append("texture plus irreguliere")

    return score, reasons


def predict_one_by_rules(row: pd.Series | dict, config: RuleConfig | None = None) -> int:
    """Retourne 0 pour OK et 1 pour Defective selon un score de regles."""
    config = config or RuleConfig()
    score, _ = rule_score(row, config)
    return int(score >= config.min_rule_score)


def predict_dataframe_by_rules(
    features_df: pd.DataFrame,
    config: RuleConfig | None = None,
) -> pd.Series:
    """Applique la decision par regles sur un DataFrame de caracteristiques."""
    return features_df.apply(lambda row: predict_one_by_rules(row, config), axis=1)


def explain_rule_decision(row: pd.Series | dict, config: RuleConfig | None = None) -> str:
    """Produit une explication courte de la decision."""
    config = config or RuleConfig()
    score, reasons = rule_score(row, config)
    prediction = int(score >= config.min_rule_score)

    if not reasons:
        reasons.append("mesures sous les seuils")

    return (
        f"Prediction: {LABEL_TO_NAME[prediction]} "
        f"(score={score:.1f}/{config.min_rule_score:.1f}, {', '.join(reasons)})"
    )

