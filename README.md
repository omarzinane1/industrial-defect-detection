# Detection de defauts industriels sur pieces de fonderie

## Contexte

Ce projet porte sur la detection automatique de defauts sur des pieces de fonderie a partir d'images en niveaux de gris. Les images sont vues de dessus et appartiennent a deux classes :

- `OK` : piece correcte
- `Defective` : piece presentant un defaut visible

L'objectif est de construire un projet clair, progressif et exploitable, en comparant deux approches :

- une approche classique de traitement d'image avec decision par regles ;
- une approche de Machine Learning classique basee sur SVM.

Le projet contient aussi une interface Tkinter pour tester les predictions sur image importee ou via la camera du PC.

## Problematique

Dans un contexte industriel, l'inspection visuelle manuelle peut etre lente, repetitive et sensible a la fatigue. Ce projet cherche donc a automatiser une partie du controle qualite avec une logique simple et pedagogique :

1. charger les images ;
2. les pretraiter ;
3. isoler des zones suspectes ;
4. extraire des mesures utiles ;
5. prendre une decision finale.

## Objectifs

Les objectifs principaux sont :

- visualiser et comprendre le dataset ;
- construire un pipeline simple de traitement d'image ;
- segmenter les zones suspectes ;
- extraire des caracteristiques interpretable ;
- implementer une approche par regles ;
- entrainer un modele SVM ;
- comparer les deux approches ;
- proposer une interface desktop simple pour les tests ;
- ajouter un controle de stabilite camera avec Lucas-Kanade.

## Dataset

Le dataset doit etre place dans le dossier suivant :

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

Labels utilises :

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
│   ├── motion.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── pipeline_visualization.py
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

## Methodologie

Le projet suit une logique progressive.

### 1. Traitement d'image + regles

On commence par une approche simple et explicable :

1. lecture de l'image en niveaux de gris ;
2. redimensionnement ;
3. reduction du bruit ;
4. amelioration du contraste ;
5. seuillage ;
6. segmentation des zones suspectes ;
7. extraction de mesures ;
8. decision finale par regles.

Cette approche permet de comprendre visuellement le probleme et de relier chaque decision a des mesures simples.

### 2. Traitement d'image + SVM

La seconde approche reutilise le pipeline d'image, mais remplace la decision fixe par un modele SVM.

Caracteristiques principales :

- statistiques d'intensite ;
- histogrammes de niveaux de gris ;
- texture GLCM ;
- texture LBP ;
- mesures de segmentation.

Le pipeline SVM contient :

1. preparation de `X` et `y` ;
2. normalisation avec `StandardScaler` ;
3. apprentissage avec un SVM RBF ;
4. prediction sur le test ;
5. sauvegarde du modele et des metriques.

### 3. Controle de stabilite camera avec Lucas-Kanade

Une brique complementaire a ete ajoutee pour le mode camera. Elle ne remplace pas les predictions rules ou SVM. Elle sert uniquement a evaluer la stabilite de la scene avant l'analyse.

Cette brique repose sur OpenCV avec :

- `cv2.goodFeaturesToTrack`
- `cv2.calcOpticalFlowPyrLK`

Le principe est simple :

1. conserver la frame precedente en grayscale ;
2. detecter de bons points a suivre ;
3. suivre ces points sur la frame suivante ;
4. calculer les vecteurs de deplacement ;
5. produire un score global de mouvement ;
6. afficher un etat simple :
   - `Stable`
   - `En mouvement`
   - `Instable`

Cette information est utile pour :

- verifier que la piece est stable devant la camera ;
- ameliorer la qualite de capture ;
- eviter une prediction automatique quand le mouvement est trop fort.

## Technologies utilisees

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-image
- scikit-learn
- joblib
- Pillow
- pathlib
- Jupyter
- Tkinter

Aucun deep learning n'est utilise.

## Installation

Depuis la racine du projet :

```bash
python -m venv .venv
```

Sous Windows PowerShell :

```bash
.venv\Scripts\Activate.ps1
```

Puis :

```bash
pip install -r requirements.txt
```

## Execution recommandee

Ouvrir les notebooks dans l'ordre :

1. `01_visualisation_dataset.ipynb`
2. `02_pretraitement_filtrage.ipynb`
3. `03_contraste_seuillage_contours.ipynb`
4. `04_segmentation_mesures.ipynb`
5. `05_decision_par_regles.ipynb`
6. `06_extraction_caracteristiques_ml.ipynb`
7. `07_modele_svm.ipynb`
8. `08_comparaison_finale.ipynb`

Le notebook 06 genere :

```text
data/features/casting_features.csv
```

Le notebook 07 sauvegarde :

```text
results/models/svm_model.joblib
```

## Execution rapide avec main.py

```bash
python main.py
```

Ce script charge une image, applique le pipeline principal et affiche une prediction simple.

## Interface desktop Tkinter

L'application se lance avec :

```bash
python app_tkinter.py
```

Fonctionnalites principales :

- choix explicite entre `Traitement d'image + regles` et `Traitement d'image + SVM`
- import d'image depuis le PC
- ouverture de la camera du PC
- capture d'une frame camera
- prediction sur l'image courante
- affichage des statistiques utiles
- vues secondaires du pipeline
- controle de stabilite camera par Lucas-Kanade

### Informations affichees

L'interface affiche notamment :

- la methode choisie ;
- la source utilisee ;
- la decision finale ;
- le score ou la confiance si disponible ;
- le statut de validation de l'image ;
- des statistiques liees a la prediction ;
- l'etat du mouvement en mode camera ;
- le score de mouvement ;
- le nombre de points suivis ;
- le deplacement moyen ;
- un message de stabilite.

### Role de Lucas-Kanade dans l'interface

Le suivi Lucas-Kanade est actif surtout pour le flux camera.

Il sert a :

- detecter si la scene est stable ;
- informer l'utilisateur si la piece bouge trop ;
- faciliter une meilleure capture avant prediction ;
- limiter la prediction automatique en mode semi temps reel lorsque la scene est instable.

Quand le mouvement est trop fort, l'application affiche un message du type :

```text
Stabilisez la piece avant l'analyse.
```

### Tester le mouvement camera

1. lancer `python app_tkinter.py`
2. ouvrir la camera ;
3. observer le panneau de stabilite ;
4. laisser la scene immobile pour obtenir `Stable` ;
5. bouger fortement la camera ou la piece pour observer `En mouvement` ou `Instable` ;
6. revenir a une scene calme avant de lancer la prediction.

## Description des notebooks

### 01 - Visualisation du dataset

Construit l'index du dataset, resume les classes et affiche des exemples.

### 02 - Pretraitement et filtrage

Montre les filtres de bruit et l'amelioration du contraste.

### 03 - Contraste, seuillage et contours

Applique Otsu, le seuillage adaptatif et Canny.

### 04 - Segmentation et mesures

Nettoie les masques et extrait des mesures simples sur les regions detectees.

### 05 - Decision par regles

Evalue l'approche classique sans Machine Learning.

### 06 - Extraction de caracteristiques ML

Construit le dataset tabulaire des features.

### 07 - Modele SVM

Prepare les donnees, entraine le SVM et sauvegarde le modele.

### 08 - Comparaison finale

Compare l'approche par regles et l'approche SVM avec les memes metriques.

## Description des fichiers src/

### preprocessing.py

Chargement, redimensionnement, normalisation et pretraitement principal.

### filtering.py

Filtres classiques : gaussien, median, bilateral, egalisation d'histogramme, CLAHE.

### segmentation.py

Seuillage, nettoyage morphologique, contours et segmentation des zones sombres.

### features.py

Extraction des caracteristiques numeriques : intensite, histogrammes, texture et mesures de segmentation.

### rules.py

Logique de decision par regles, calibration des seuils et explication de prediction.

### ml_models.py

Preparation de `X` et `y`, creation du pipeline SVM, entrainement, chargement et sauvegarde.

### motion.py

Estimation de mouvement par Lucas-Kanade pour le flux camera :

- preparation des frames grayscale ;
- detection de points a suivre ;
- suivi des points entre deux frames ;
- calcul des deplacements ;
- score global de mouvement ;
- decision simple de stabilite.

### evaluation.py

Metriques, rapports de classification, matrices de confusion et sauvegarde des resultats.

### inference.py

Fonctions de prediction reutilisees par l'interface Tkinter.

### pipeline_visualization.py

Generation des vues intermediaires du pipeline pour l'interface.

### utils.py

Chemins du projet, index du dataset, resumes et fonctions utilitaires.

## Resultats attendus

A la fin du projet, on obtient :

- un pipeline classique de traitement d'image ;
- une decision par regles ;
- un modele SVM entraine ;
- des metriques sauvegardees ;
- une interface Tkinter exploitable ;
- un controle de stabilite camera ;
- une comparaison claire entre les deux approches.

## Avantages et limites

### Traitement d'image + regles

Avantages :

- simple a comprendre ;
- rapide ;
- interpretable.

Limites :

- sensible aux seuils ;
- moins flexible ;
- sensible aux conditions d'image.

### Traitement d'image + SVM

Avantages :

- exploite plusieurs features a la fois ;
- tres performant sur le dataset ;
- plus flexible qu'une simple regle.

Limites :

- depend de la qualite des features ;
- moins interpretable qu'une regle manuelle.

### Lucas-Kanade pour la stabilite camera

Avantages :

- coherent avec un cours de vision par ordinateur ;
- adapte au flux webcam ;
- simple a integrer avec OpenCV ;
- utile pour filtrer les captures instables.

Limites :

- depend de la presence de points d'interet ;
- moins informatif si la scene est trop uniforme ;
- ne remplace pas une vraie acquisition controlee en environnement industriel.

## Pistes d'amelioration

- ajuster les seuils de stabilite ;
- visualiser les vecteurs de mouvement dans une fenetre dediee ;
- memoriser plusieurs frames stables avant prediction ;
- comparer Lucas-Kanade a d'autres approches de flot optique ;
- tester d'autres classifieurs classiques ;
- analyser les erreurs image par image.

## Messages de commit recommandes

```text
Add Lucas-Kanade motion estimation module for camera stability
Integrate camera motion status into Tkinter interface
Gate live camera predictions with motion stability checks
Document Lucas-Kanade stability control in README
```
