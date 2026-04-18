# Détection de défauts industriels sur pièces de fonderie

## Présentation du projet

Dans ce projet, j’ai travaillé sur la **détection automatique de défauts industriels** à partir d’images de **pièces de fonderie** vues de dessus.

Les images utilisées sont en **niveaux de gris** et appartiennent à deux classes :

* **OK** : pièce correcte
* **Defective** : pièce défectueuse

L’idée principale du projet est de construire une démarche complète et claire pour savoir si une pièce est bonne ou non à partir de son image.

Je n’ai pas voulu faire directement quelque chose de trop complexe. J’ai préféré suivre une logique simple et progressive, comme dans un vrai travail d’analyse :

1. comprendre les images
2. appliquer des techniques de traitement d’image
3. extraire des mesures utiles
4. prendre une décision
5. ensuite comparer cette première approche avec une approche Machine Learning

Le projet compare donc **deux approches** :

* **Approche 1 : traitement d’image classique + décision par règles**
* **Approche 2 : traitement d’image + modèle SVM**

Le but est de voir la différence entre une méthode plus explicable, basée sur des règles, et une méthode de Machine Learning classique.

---

## Pourquoi ce projet

Dans l’industrie, le contrôle qualité est souvent très important. Une pièce défectueuse peut provoquer des pertes, des retards, ou des problèmes dans la production.

Quand l’inspection se fait manuellement, elle peut être :

* lente
* répétitive
* fatigante
* sensible à l’erreur humaine

L’objectif de ce projet est donc de tester une solution automatique capable d’aider à distinguer les pièces saines des pièces défectueuses à partir d’images.

Ce projet m’a aussi permis de travailler de manière pratique sur plusieurs notions importantes :

* traitement d’image
* segmentation
* extraction de caractéristiques
* classification
* comparaison de méthodes
* organisation propre d’un projet Python

---

## Objectif

L’objectif général est de construire un système capable de prédire si une image correspond à une pièce :

* **OK**
* ou **Defective**

Pour cela, j’ai organisé le travail en deux grandes parties :

### 1. Approche classique

Dans cette partie, je passe par les étapes classiques de vision par ordinateur :

* prétraitement
* filtrage
* amélioration du contraste
* seuillage
* contours
* segmentation
* extraction de mesures
* décision finale par règles

### 2. Approche Machine Learning

Dans cette partie, je transforme les images en **caractéristiques numériques**, puis j’utilise un modèle **SVM** pour faire la classification.

Enfin, je compare les deux approches avec les mêmes métriques.

---

## Dataset

Le dataset doit être placé dans ce dossier :

```text
data/raw/casting_data/
├── train/
│   ├── ok_front/
│   └── def_front/
└── test/
    ├── ok_front/
    └── def_front/
```

Le projet lit directement cette structure. Les labels sont interprétés comme suit :

* `ok_front` → **OK** → label `0`
* `def_front` → **Defective** → label `1`

---

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
│   ├── inference.py
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

Cette organisation me permet de séparer :

* les **données**
* les **notebooks**
* la **logique du projet**
* les **résultats**
* l’**interface utilisateur**

---

## Démarche suivie

J’ai choisi une démarche progressive.

### Première étape : comprendre les images

Avant de parler de modèle, j’ai commencé par observer le dataset pour voir :

* la forme générale des pièces
* les différences visibles entre OK et Defective
* les variations d’intensité
* les zones qui semblent suspectes

### Deuxième étape : traitement d’image classique

Ensuite, j’ai appliqué plusieurs techniques de traitement d’image pour faire ressortir les défauts :

* prétraitement
* filtrage
* contraste
* seuillage
* détection de contours
* segmentation

Le but ici était de transformer l’image brute en informations plus faciles à exploiter.

### Troisième étape : décision par règles

Après la segmentation, j’ai extrait des mesures simples puis construit une logique de décision par règles pour classer l’image.

### Quatrième étape : approche SVM

Une fois la partie classique bien comprise, j’ai construit une deuxième approche basée sur un modèle **SVM** à partir de caractéristiques extraites des images.

### Cinquième étape : comparaison finale

Enfin, j’ai comparé les deux approches avec les mêmes métriques pour voir laquelle fonctionne le mieux.

---

## Approche 1 : traitement d’image + règles

Cette première approche repose uniquement sur des techniques classiques de vision par ordinateur.

### Étapes principales

* chargement de l’image en niveaux de gris
* redimensionnement
* réduction du bruit
* amélioration du contraste
* seuillage
* nettoyage morphologique
* segmentation des zones sombres ou suspectes
* extraction de mesures simples
* décision finale par règles

### Idée

L’idée est simple : si plusieurs indicateurs visuels montrent qu’une pièce semble anormale, alors elle est classée comme **Defective**.

### Ce que cette approche apporte

Cette approche est intéressante parce qu’elle est :

* simple à comprendre
* facile à expliquer
* utile pour apprendre
* proche de la logique classique du traitement d’image

### Limite

Elle dépend beaucoup :

* de la qualité de la segmentation
* du choix des seuils
* des conditions visuelles

---

## Approche 2 : traitement d’image + SVM

Dans cette deuxième approche, je garde l’idée d’extraire des informations à partir des images, mais au lieu de décider avec des règles fixes, j’utilise un **modèle SVM**.

### Caractéristiques extraites

J’extrais par exemple :

* statistiques d’intensité
* histogrammes
* caractéristiques de texture
* mesures issues de la segmentation
* autres caractéristiques numériques utiles

### Pipeline

Le pipeline suit cette logique :

1. construire un tableau de caractéristiques
2. préparer `X` et `y`
3. normaliser les variables
4. entraîner le modèle SVM
5. tester le modèle
6. sauvegarder les métriques et le modèle

### Ce que cette approche apporte

Le SVM permet de mieux exploiter les combinaisons de caractéristiques et donne des résultats plus performants sur ce projet.

### Limite

Il est moins directement interprétable qu’une simple logique par règles.

---

## Comparaison des deux approches

Le projet compare les deux approches avec les métriques suivantes :

* accuracy
* precision
* recall
* f1-score
* matrice de confusion

### Résultats obtenus

| Méthode                     | Accuracy | Precision | Recall   | F1-score |
| --------------------------- | -------- | --------- | -------- | -------- |
| Traitement d’image + règles | 0.827972 | 0.796763  | 0.977925 | 0.878097 |
| Traitement d’image + SVM    | 0.997203 | 0.995604  | 1.000000 | 0.997797 |

### Interprétation

L’approche par règles donne déjà un bon résultat, surtout après amélioration. Elle détecte très bien les pièces défectueuses, ce qui est important dans un contexte industriel.

Mais l’approche **SVM** reste la plus performante globalement, avec des résultats presque parfaits sur ce dataset.

---

## Technologies utilisées

Dans ce projet, j’ai utilisé :

* **Python**
* **OpenCV**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **scikit-image**
* **scikit-learn**
* **joblib**
* **Jupyter Notebook**
* **Tkinter**

Je n’ai pas utilisé de Deep Learning dans ce projet. Le but était de construire une solution claire, progressive et bien maîtrisée. 

---

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

Si `pip` pose problème, on peut aussi utiliser :

```bash
py -m pip install -r requirements.txt
```

---

## Comment exécuter le projet

### Avec les notebooks

Le mieux est d’ouvrir les notebooks dans l’ordre :

1. `01_visualisation_dataset.ipynb`
2. `02_pretraitement_filtrage.ipynb`
3. `03_contraste_seuillage_contours.ipynb`
4. `04_segmentation_mesures.ipynb`
5. `05_decision_par_regles.ipynb`
6. `06_extraction_caracteristiques_ml.ipynb`
7. `07_modele_svm.ipynb`
8. `08_comparaison_finale.ipynb`

Cette suite permet de suivre toute la logique du projet étape par étape.

### Avec `main.py`

Pour un test rapide :

```bash
python main.py
```

Ce fichier applique le pipeline principal sur une image et affiche une prédiction.

---

## Interface Tkinter

J’ai aussi ajouté une interface desktop avec **Tkinter** pour utiliser le projet de manière plus simple.

### Lancer l’application

```bash
python app_tkinter.py
```

### Ce que permet l’interface

L’application permet de :

* choisir la méthode :

  * **Traitement d’image + règles**
  * **Traitement d’image + SVM**
* importer une image depuis le PC
* ouvrir la caméra du PC
* capturer une image
* lancer une prédiction
* afficher la classe prédite
* afficher des statistiques utiles selon la méthode choisie

### Ce que j’ai voulu avec cette interface

Je voulais une interface :

* claire
* simple
* propre
* professionnelle
* facile à tester

L’idée est de rendre le projet plus concret, plus visuel, et plus facile à présenter.

---

## Description rapide des notebooks

### 01 - Visualisation du dataset

Ce notebook sert à explorer le dataset, afficher des exemples, vérifier la structure des données et mieux comprendre les images.

### 02 - Prétraitement et filtrage

Dans ce notebook, je teste les premières opérations de préparation des images : resize, filtrage, réduction du bruit.

### 03 - Contraste, seuillage et contours

Ici, je travaille sur l’amélioration du contraste, le seuillage et la détection de contours.

### 04 - Segmentation et mesures

Ce notebook permet d’isoler les zones suspectes et de calculer des mesures simples.

### 05 - Décision par règles

Dans cette étape, je construis l’approche classique de classification sans Machine Learning.

### 06 - Extraction de caractéristiques ML

Je transforme les images en données tabulaires exploitables par un modèle de Machine Learning.

### 07 - Modèle SVM

Dans ce notebook, j’entraîne et j’évalue le modèle SVM.

### 08 - Comparaison finale

Ici, je compare les performances finales des deux approches.

---

## Description rapide des fichiers `src`

### `preprocessing.py`

Fonctions de chargement, redimensionnement et prétraitement.

### `filtering.py`

Filtres classiques et amélioration du contraste.

### `segmentation.py`

Seuillage, contours, opérations morphologiques et segmentation.

### `features.py`

Extraction des caractéristiques numériques à partir des images.

### `rules.py`

Logique de décision par règles.

### `ml_models.py`

Préparation des données et entraînement du SVM.

### `evaluation.py`

Calcul et sauvegarde des métriques.

### `inference.py`

Fonctions de prédiction réutilisées par l’interface Tkinter.

### `utils.py`

Fonctions utilitaires pour les chemins, l’index du dataset et quelques aides générales.

---

## Ce que ce projet montre

À travers ce projet, j’ai voulu montrer que je peux :

* organiser un projet Python proprement
* travailler sur un problème industriel réel
* utiliser le traitement d’image de manière concrète
* construire une logique de décision par règles
* entraîner un modèle de Machine Learning classique
* comparer deux approches sur le même problème
* ajouter une interface utilisateur pour rendre le projet plus exploitable

---

## Limites du projet

Même si le projet donne de bons résultats, il a aussi quelques limites :

* il dépend du dataset utilisé
* les règles restent sensibles aux seuils
* les performances peuvent changer sur d’autres images
* le SVM dépend de la qualité des caractéristiques extraites

---

## Pistes d’amélioration

Plus tard, ce projet peut être amélioré avec :

* d’autres descripteurs de texture
* d’autres modèles classiques
* une validation plus poussée
* une meilleure analyse des erreurs
* une interface encore plus avancée
* une version temps réel plus robuste
* éventuellement une approche Deep Learning pour comparaison

---

## Conclusion

Ce projet m’a permis de construire une démarche complète autour de la détection de défauts industriels.

J’ai commencé par une approche simple et explicable basée sur le traitement d’image classique, puis j’ai ajouté une approche SVM plus performante. Enfin, j’ai intégré le tout dans une interface Tkinter pour rendre le projet plus concret et plus facile à utiliser.

Le projet est donc à la fois :

* pédagogique
* technique
* pratique
* structuré
* réutilisable

Et surtout, il montre bien comment passer d’une image brute à une décision finale dans un contexte de contrôle qualité industriel.
