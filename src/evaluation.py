"""Evaluation, matrices de confusion et sauvegarde des resultats."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
except ImportError:  # pragma: no cover - fallback utile pour la partie regles sans sklearn.
    ConfusionMatrixDisplay = None
    classification_report = None
    confusion_matrix = None

from .utils import LABEL_TO_NAME


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Calcule les metriques principales en binaire."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Transforme plusieurs dictionnaires de metriques en tableau."""
    rows = []
    for method, metrics in results.items():
        row = {"method": method}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def classification_report_dataframe(y_true, y_pred) -> pd.DataFrame:
    """Retourne le rapport de classification sous forme de DataFrame."""
    if classification_report is not None:
        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[LABEL_TO_NAME[0], LABEL_TO_NAME[1]],
            output_dict=True,
            zero_division=0,
        )
        return pd.DataFrame(report).transpose()

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    rows: dict[str, dict[str, float]] = {}
    for label, name in LABEL_TO_NAME.items():
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        support = int((y_true == label).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows[name] = {"precision": precision, "recall": recall, "f1-score": f1, "support": support}

    return pd.DataFrame(rows).transpose()


def save_metrics(metrics: dict, path: str | Path) -> None:
    """Sauvegarde des metriques dans un fichier JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Sauvegarde un tableau de resultats en CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=True)


def _manual_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Construit une matrice de confusion 2x2 sans sklearn."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return np.array(
        [
            [int(((y_true == 0) & (y_pred == 0)).sum()), int(((y_true == 0) & (y_pred == 1)).sum())],
            [int(((y_true == 1) & (y_pred == 0)).sum()), int(((y_true == 1) & (y_pred == 1)).sum())],
        ]
    )


def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str,
    output_path: str | Path | None = None,
) -> None:
    """Affiche et sauvegarde une matrice de confusion."""
    if confusion_matrix is not None and ConfusionMatrixDisplay is not None:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[LABEL_TO_NAME[0], LABEL_TO_NAME[1]],
        )
        display.plot(cmap="Blues", values_format="d")
    else:
        cm = _manual_confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], labels=[LABEL_TO_NAME[0], LABEL_TO_NAME[1]])
        ax.set_yticks([0, 1], labels=[LABEL_TO_NAME[0], LABEL_TO_NAME[1]])
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Vraie classe")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.title(title)
    plt.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.show()
