"""Preparation des donnees et entrainement du modele SVM."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import get_feature_columns


def prepare_xy(
    features_df: pd.DataFrame,
    split: str,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare X et y pour un split donne."""
    subset = features_df[features_df["split"] == split].copy()
    if subset.empty:
        raise ValueError(f"Aucune donnee trouvee pour le split '{split}'.")

    feature_columns = feature_columns or get_feature_columns(features_df)
    X = subset[feature_columns]
    y = subset["label"].astype(int)
    return X, y, feature_columns


def create_svm_pipeline(
    kernel: str = "rbf",
    c: float = 10.0,
    gamma: str | float = "scale",
) -> Pipeline:
    """Cree un pipeline StandardScaler + SVM."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel=kernel, C=c, gamma=gamma, probability=True, random_state=42)),
        ]
    )


def train_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kernel: str = "rbf",
    c: float = 10.0,
    gamma: str | float = "scale",
) -> Pipeline:
    """Entraine un SVM classique sur les caracteristiques extraites."""
    model = create_svm_pipeline(kernel=kernel, c=c, gamma=gamma)
    model.fit(X_train, y_train)
    return model


def save_model(model: Pipeline, path: str | Path) -> None:
    """Sauvegarde un modele avec joblib."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(path: str | Path) -> Pipeline:
    """Charge un modele sauvegarde avec joblib."""
    return joblib.load(path)

