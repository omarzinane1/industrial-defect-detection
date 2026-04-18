"""Point d'entree minimal du projet.

Ce script charge une image du dataset, applique le pipeline classique et affiche
une prediction simple. Le modele SVM est utilise seulement s'il a deja ete
entraine avec les notebooks.
"""

from __future__ import annotations

import pandas as pd

from src.features import extract_features_from_path
from src.preprocessing import load_and_preprocess
from src.rules import RuleConfig, explain_rule_decision, predict_one_by_rules
from src.segmentation import region_summary, segment_dark_defects
from src.utils import LABEL_TO_NAME, MODELS_DIR, build_image_index, ensure_project_directories


def choose_example_image(index_df: pd.DataFrame) -> pd.Series:
    """Choisit une image de test si possible, sinon la premiere image disponible."""
    test_df = index_df[index_df["split"] == "test"]
    if not test_df.empty:
        return test_df.iloc[0]
    return index_df.iloc[0]


def main() -> None:
    ensure_project_directories()
    index_df = build_image_index()

    if index_df.empty:
        print("Aucune image trouvee.")
        print("Placez le dataset dans data/raw/casting_data/train et data/raw/casting_data/test.")
        return

    sample = choose_example_image(index_df)
    image_path = sample["path"]

    print("Image utilisee :", image_path)
    print("Classe reelle  :", sample["label_name"])

    image = load_and_preprocess(image_path)
    mask = segment_dark_defects(image)
    measurements = region_summary(mask)

    print("\nMesures principales du masque :")
    for key, value in measurements.items():
        print(f"- {key}: {value:.6f}")

    features = extract_features_from_path(image_path)
    rule_prediction = predict_one_by_rules(features, RuleConfig())

    print("\nApproche classique par regles :")
    print(explain_rule_decision(features, RuleConfig()))
    print("Prediction finale :", LABEL_TO_NAME[rule_prediction])

    model_path = MODELS_DIR / "svm_model.joblib"
    if model_path.exists():
        try:
            from src.ml_models import load_model

            model = load_model(model_path)
            X_one = pd.DataFrame([features])
            svm_prediction = int(model.predict(X_one)[0])
            print("\nApproche SVM :")
            print("Prediction finale :", LABEL_TO_NAME[svm_prediction])
        except ModuleNotFoundError:
            print("\nModele SVM trouve, mais scikit-learn n'est pas installe dans cet environnement.")
    else:
        print("\nModele SVM non trouve.")
        print("Lancez les notebooks 06 puis 07 pour extraire les caracteristiques et entrainer le SVM.")


if __name__ == "__main__":
    main()
