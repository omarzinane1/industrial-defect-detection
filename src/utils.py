"""Fonctions utilitaires pour le projet.

Ce module centralise les chemins, la lecture de l'index du dataset et quelques
fonctions d'affichage utilisees dans les notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "casting_data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = RESULTS_DIR / "models"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_FOLDER_TO_LABEL = {"ok_front": 0, "def_front": 1}
LABEL_TO_NAME = {0: "OK", 1: "Defective"}


def ensure_project_directories() -> None:
    """Cree les dossiers de sortie si necessaire."""
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FEATURES_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        MODELS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def get_image_files(directory: Path) -> list[Path]:
    """Retourne les fichiers image d'un dossier, tries par nom."""
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_image_index(
    dataset_dir: Path = RAW_DATA_DIR,
    splits: Iterable[str] = ("train", "test"),
) -> pd.DataFrame:
    """Construit un tableau avec les chemins, classes et labels du dataset."""
    rows: list[dict[str, object]] = []

    for split in splits:
        split_dir = dataset_dir / split
        for class_folder, label in CLASS_FOLDER_TO_LABEL.items():
            class_dir = split_dir / class_folder
            for image_path in get_image_files(class_dir):
                rows.append(
                    {
                        "path": str(image_path),
                        "filename": image_path.name,
                        "split": split,
                        "class_folder": class_folder,
                        "label": label,
                        "label_name": LABEL_TO_NAME[label],
                    }
                )

    return pd.DataFrame(rows)


def dataset_summary(index_df: pd.DataFrame) -> pd.DataFrame:
    """Resume le nombre d'images par split et par classe."""
    if index_df.empty:
        return pd.DataFrame(columns=["split", "label_name", "count"])

    return (
        index_df.groupby(["split", "label_name"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["split", "label_name"])
        .reset_index(drop=True)
    )


def sample_by_class(
    index_df: pd.DataFrame,
    split: str = "train",
    n_per_class: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """Selectionne quelques images de chaque classe pour l'affichage."""
    if index_df.empty:
        return index_df.copy()

    subset = index_df[index_df["split"] == split]
    samples: list[pd.DataFrame] = []
    for _, group in subset.groupby("label_name"):
        n = min(n_per_class, len(group))
        if n > 0:
            samples.append(group.sample(n=n, random_state=random_state))

    if not samples:
        return pd.DataFrame(columns=index_df.columns)

    return pd.concat(samples).reset_index(drop=True)


def display_images(
    images: list,
    titles: list[str] | None = None,
    cmap: str = "gray",
    cols: int = 4,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """Affiche une grille d'images avec Matplotlib."""
    if not images:
        print("Aucune image a afficher.")
        return

    titles = titles or [""] * len(images)
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = list(pd.Series(axes.ravel() if hasattr(axes, "ravel") else [axes]))

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    for ax in axes[len(images) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Sauvegarde un DataFrame en CSV en creant le dossier parent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

