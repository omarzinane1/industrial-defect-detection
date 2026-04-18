# Détection de défauts industriels sur pièces de fonderie

## Contexte

Ce projet porte sur la détection automatique de défauts industriels à partir d'images de pièces de fonderie vues de dessus. Les images sont en niveaux de gris et appartiennent à deux classes :

- `OK` : pièce considérée comme correcte.
- `Defective` : pièce présentant un défaut visible.

L'objectif est de construire une démarche complète, progressive et compréhensible, en comparant deux approches sur le même dataset :

- une approche classique de traitement d'image avec décision par règles ;
- une approche de Machine Learning classique basée sur SVM.

Le projet est organisé pour être ouvert directement dans VS Code et exécuté étape par étape avec les notebooks.

## Problématique

Dans un contexte industriel, l'inspection visuelle manuelle peut être lente, répétitive et sensible à la fatigue humaine. L'idée du projet est donc de tester une méthode automatique capable d'aider à distinguer les pièces normales des pièces défectueuses.

Le but n'est pas d'utiliser une solution très complexe, mais de comprendre clairement comment passer :

1. d'une image brute ;
2. à une image prétraitée ;
3. à une segmentation des zones suspectes ;
4. à des mesures numériques ;
5. à une décision finale.

## Objectifs

Les objectifs principaux du projet sont :

- charger et visualiser le dataset ;
- appliquer des techniques simples de prétraitement ;
- filtrer les images et améliorer le contraste ;
- réaliser du seuillage, de la détection de contours et une segmentation simple ;
- extraire des mesures interprétables ;
- construire une logique de décision par règles ;
- extraire des caractéristiques pour un modèle SVM ;
- entraîner et évaluer un modèle de Machine Learning classique ;
- comparer les deux approches avec les mêmes métriques.

## Dataset

Le dataset doit être placé dans le dossier suivant :

```text
data/raw/casting_data/
├── train/
│   ├── ok_front/
│   └── def_front/
└── test/
    ├── ok_front/
    └── def_front/
```

Le code lit automatiquement cette structure.

Les labels utilisés dans le projet sont :

- `ok_front` -> `OK` -> label `0`
- `def_front` -> `Defective` -> label `1`

## Structure du projet

```text
industrial-defect-detection/
├── data/
│   ├── raw/
│   │   └── casting_data/
│   ├── processed/
│   └── features/
├── notebooks/
│   ├── 01_visualisation_dataset.ipynb
│   ├── 02_pretraitement_filtrage.ipynb
│   ├── 03_contraste_seuillage_contours.ipynb
│   ├── 04_segmentation_mesures.ipynb
│   ├── 05_decision_par_regles.ipynb
│   ├── 06_extraction_caracteristiques_ml.ipynb
│   ├── 07_modele_svm.ipynb
│   └── 08_comparaison_finale.ipynb
├── src/
│   ├── preprocessing.py
│   ├── filtering.py
│   ├── segmentation.py
│   ├── features.py
│   ├── rules.py
│   ├── ml_models.py
│   ├── evaluation.py
│   └── utils.py
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── README.md
├── requirements.txt
├── .gitignore
├── app_tkinter.py
└── main.py
```

## Méthodologie

La méthodologie suit une logique progressive.

D'abord, on commence par le traitement d'image classique. Cette étape est importante parce qu'elle permet de comprendre les images, d'observer les défauts, de voir l'effet des filtres et d'obtenir des mesures simples. Elle donne une base visuelle et interprétable au projet.

Ensuite, on ajoute un modèle SVM. Le SVM utilise les caractéristiques extraites des images pour apprendre une frontière de décision entre les pièces OK et Defective. Cette approche reste du Machine Learning classique : il n'y a pas de deep learning.

Enfin, on compare les deux méthodes. Cette comparaison est utile parce qu'elle montre ce que l'on gagne ou non en passant d'une décision manuelle par règles à un modèle entraîné sur les données.

## Approche 1 : traitement d'image classique + règles

Cette approche suit les étapes suivantes :

1. Chargement de l'image en niveaux de gris.
2. Redimensionnement à une taille commune.
3. Réduction du bruit avec un filtre gaussien.
4. Amélioration du contraste avec CLAHE.
5. Seuillage inverse pour isoler les régions sombres.
6. Nettoyage du masque par opérations morphologiques.
7. Suppression des régions connectées au bord.
8. Extraction de mesures simples :
   - nombre de régions suspectes ;
   - surface totale des régions ;
   - ratio de surface suspecte ;
   - surface de la plus grande région.
9. Décision finale avec des seuils :
   - `OK`
   - `Defective`

Cette méthode est simple et explicable. Elle est intéressante pour comprendre le problème, mais elle dépend beaucoup de la qualité du seuillage et du choix des seuils.

## Approche 2 : traitement d'image + SVM

La deuxième approche conserve le traitement d'image, mais remplace la décision fixe par un modèle SVM.

Les caractéristiques extraites sont :

- statistiques d'intensité ;
- histogrammes de niveaux de gris ;
- caractéristiques de texture GLCM ;
- caractéristiques de texture LBP ;
- mesures issues de la segmentation.

Ces caractéristiques forment un dataset tabulaire. Le pipeline SVM contient :

1. préparation de `X` et `y` ;
2. normalisation avec `StandardScaler` ;
3. entraînement d'un SVM avec noyau RBF ;
4. prédiction sur le jeu de test ;
5. évaluation avec les métriques classiques.

Cette méthode peut mieux exploiter les combinaisons de caractéristiques, mais elle est moins directement interprétable qu'une règle simple.

## Technologies utilisées

Le projet utilise uniquement des outils classiques :

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-image
- scikit-learn
- joblib
- pathlib
- Jupyter

Aucun modèle de deep learning n'est utilisé.

## Installation

Depuis la racine du projet :

```bash
python -m venv .venv
```

Sous Windows PowerShell :

```bash
.venv\Scripts\Activate.ps1
```

Puis installer les dépendances :

```bash
pip install -r requirements.txt
```

## Exécution recommandée

L'exécution la plus claire consiste à ouvrir les notebooks dans l'ordre :

1. `01_visualisation_dataset.ipynb`
2. `02_pretraitement_filtrage.ipynb`
3. `03_contraste_seuillage_contours.ipynb`
4. `04_segmentation_mesures.ipynb`
5. `05_decision_par_regles.ipynb`
6. `06_extraction_caracteristiques_ml.ipynb`
7. `07_modele_svm.ipynb`
8. `08_comparaison_finale.ipynb`

Le notebook 06 génère le fichier :

```text
data/features/casting_features.csv
```

Le notebook 07 sauvegarde le modèle :

```text
results/models/svm_model.joblib
```

Les métriques et figures sont sauvegardées dans :

```text
results/metrics/
results/figures/
```

## Exécution rapide avec main.py

Le fichier `main.py` sert de point d'entrée minimal. Il charge une image, applique la segmentation, extrait des mesures et affiche une prédiction par règles.

```bash
python main.py
```

Si le modèle SVM existe déjà, `main.py` affiche aussi la prédiction du SVM.

## Interface desktop Tkinter

Le projet contient aussi une interface graphique simple et professionnelle avec Tkinter :

```bash
python app_tkinter.py
```

L'application permet de choisir explicitement entre :

- `Traitement d'image + règles`
- `Traitement d'image + SVM`

Elle permet aussi de choisir la source :

- importer une image depuis le PC ;
- ouvrir la caméra du PC ;
- capturer une image depuis la caméra ;
- lancer une prédiction sur l'image affichée.

La zone de gauche affiche l'image importée ou le flux caméra. La zone de droite affiche la méthode sélectionnée, la source utilisée, la classe prédite, le score disponible et des statistiques utiles.

Pour l'approche par règles, l'interface affiche notamment le score de suspicion, le nombre de régions suspectes, le ratio de pixels sombres, la densité de contours, la moyenne et l'écart-type des niveaux de gris.

Pour l'approche SVM, l'interface utilise le modèle sauvegardé dans :

```text
results/models/svm_model.joblib
```

Si le modèle SVM ou `scikit-learn` n'est pas disponible, l'application affiche un message d'erreur clair. Le mode `Traitement d'image + règles` reste utilisable.

## Description des notebooks

### 01 - Visualisation du dataset

Ce notebook vérifie que les images sont bien placées, construit un index du dataset, compte les images par classe et affiche quelques exemples.

### 02 - Prétraitement et filtrage

Ce notebook montre le redimensionnement, les filtres de bruit et l'amélioration du contraste. Il compare notamment le filtre gaussien, le filtre médian, le filtre bilatéral et CLAHE.

### 03 - Contraste, seuillage et contours

Ce notebook applique le seuillage d'Otsu, le seuillage adaptatif et la détection de contours avec Canny. Il permet de voir comment les zones suspectes commencent à apparaître.

### 04 - Segmentation et mesures

Ce notebook nettoie le masque binaire et extrait des mesures simples sur les régions détectées.

### 05 - Décision par règles

Ce notebook utilise les mesures de segmentation pour construire une première classification sans Machine Learning. Les seuils sont calibrés à partir des images OK du train.

### 06 - Extraction de caractéristiques ML

Ce notebook transforme les images en tableau de caractéristiques. Ce tableau est ensuite utilisé pour entraîner le SVM.

### 07 - Modèle SVM

Ce notebook prépare `X_train`, `y_train`, `X_test` et `y_test`, entraîne un SVM, évalue les résultats et sauvegarde le modèle.

### 08 - Comparaison finale

Ce notebook compare l'approche par règles et l'approche SVM avec les mêmes métriques : accuracy, precision, recall, f1-score et matrice de confusion.

## Description des fichiers src/

### preprocessing.py

Contient les fonctions de chargement, redimensionnement, normalisation et prétraitement principal des images.

### filtering.py

Contient les filtres classiques : gaussien, médian, bilatéral, égalisation d'histogramme, CLAHE et renforcement léger.

### segmentation.py

Contient les fonctions de seuillage, contours, nettoyage morphologique, suppression du bord et segmentation des zones sombres.

### features.py

Contient l'extraction des caractéristiques numériques : intensité, histogrammes, texture, LBP et mesures de segmentation.

### rules.py

Contient la logique de décision par règles, la calibration des seuils et l'explication d'une prédiction.

### ml_models.py

Contient la préparation de `X` et `y`, la création du pipeline SVM, l'entraînement et la sauvegarde du modèle.

### evaluation.py

Contient les métriques, les rapports de classification, les matrices de confusion et les fonctions de sauvegarde des résultats.

### inference.py

Contient les fonctions de prédiction utilisées par l'interface Tkinter. Ce module évite de recopier la logique métier dans `app_tkinter.py`.

### utils.py

Contient les chemins du projet, la construction de l'index du dataset, les résumés et quelques fonctions d'affichage.

## Résultats attendus

À la fin du projet, on obtient :

- une visualisation claire du dataset ;
- un pipeline classique de traitement d'image ;
- une décision simple par règles ;
- un dataset tabulaire de caractéristiques ;
- un modèle SVM entraîné ;
- des métriques sauvegardées ;
- des matrices de confusion ;
- une comparaison finale entre les deux approches.

Les résultats exacts dépendent du dataset, de sa qualité et de l'équilibre entre les classes.

## Avantages et limites

### Traitement d'image + règles

Avantages :

- simple à comprendre ;
- rapide à exécuter ;
- facilement explicable ;
- utile pour apprendre la logique de vision par ordinateur.

Limites :

- dépend fortement des seuils ;
- sensible à l'éclairage et au contraste ;
- moins flexible si les défauts ont des formes très variées.

### Traitement d'image + SVM

Avantages :

- exploite plusieurs caractéristiques en même temps ;
- apprend à partir des exemples ;
- peut mieux généraliser qu'une règle fixe.

Limites :

- nécessite un dataset bien préparé ;
- moins interprétable qu'une décision par seuils ;
- dépend de la qualité des caractéristiques extraites.

## Pistes d'amélioration

Plusieurs améliorations peuvent être envisagées :

- ajuster les paramètres de segmentation ;
- ajouter d'autres caractéristiques de texture ;
- tester d'autres modèles classiques comme Random Forest ou Logistic Regression ;
- faire une validation croisée ;
- analyser les erreurs image par image ;
- sauvegarder des exemples de bonnes et mauvaises prédictions ;
- améliorer la sélection automatique des seuils.

## Messages de commit recommandés

Voici une proposition de commits propres pour construire l'historique Git :

```text
Initial project structure for industrial defect detection
Add image preprocessing and filtering utilities
Add segmentation and feature extraction pipeline
Implement rule-based defect classification
Add SVM training and evaluation workflow
Add pedagogical notebooks for full project pipeline
Add final comparison notebook and saved outputs structure
Write French README and project usage instructions
Add Tkinter desktop interface for image and camera prediction
```
