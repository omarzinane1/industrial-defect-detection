# Detection de defauts industriels sur pieces de fonderie

## Contexte

Ce projet porte sur la detection automatique de defauts industriels a partir d'images en niveaux de gris de pieces de fonderie vues de dessus. Les images appartiennent a deux classes :

- `OK` : piece consideree comme correcte
- `Defective` : piece presentant un defaut visible

L'objectif est de construire un projet complet, progressif et compréhensible, en comparant deux approches sur le meme dataset :

- une approche classique de traitement d'image avec decision par regles ;
- une approche de Machine Learning classique avec SVM.

Le projet contient aussi une interface Tkinter permettant de tester les predictions sur image importee ou via la camera du PC.

## Problematique

Dans un contexte industriel, l'inspection visuelle manuelle peut etre lente, repetitive et sensible a la fatigue. L'idee du projet est donc de proposer une chaine simple et pedagogique qui permet de passer :

1. d'une image brute ;
2. a une image pretraitee ;
3. a une segmentation des zones suspectes ;
4. a des mesures numeriques ;
5. a une decision finale.

L'objectif n'est pas d'utiliser une solution inutilement complexe, mais de comprendre clairement les etapes d'un pipeline de vision par ordinateur applique a un cas industriel.

## Objectifs

Les objectifs principaux sont :

- charger et visualiser le dataset ;
- appliquer des techniques simples de pretraitement ;
- segmenter les zones suspectes ;
- construire une decision par regles interpretable ;
- extraire des caracteristiques pour un modele SVM ;
- entrainer et evaluer le modele ;
- comparer les deux approches ;
- proposer une interface desktop simple pour les tests ;
- ajouter un controle de stabilite camera avec Lucas-Kanade.

## Dataset

Le dataset doit etre place dans le dossier suivant :

```text
data/raw/casting_data/
|-- train/
|   |-- ok_front/
|   `-- def_front/
`-- test/
    |-- ok_front/
    `-- def_front/
```

Le code lit automatiquement cette structure.

Labels utilises :

- `ok_front` -> `OK` -> label `0`
- `def_front` -> `Defective` -> label `1`

## Structure du projet

```text
industrial-defect-detection/
|-- data/
|   |-- raw/
|   |   `-- casting_data/
|   |-- processed/
|   `-- features/
|-- notebooks/
|   |-- 01_visualisation_dataset.ipynb
|   |-- 02_pretraitement_filtrage.ipynb
|   |-- 03_contraste_seuillage_contours.ipynb
|   |-- 04_segmentation_mesures.ipynb
|   |-- 05_decision_par_regles.ipynb
|   |-- 06_extraction_caracteristiques_ml.ipynb
|   |-- 07_modele_svm.ipynb
|   `-- 08_comparaison_finale.ipynb
|-- src/
|   |-- preprocessing.py
|   |-- filtering.py
|   |-- segmentation.py
|   |-- features.py
|   |-- rules.py
|   |-- ml_models.py
|   |-- motion.py
|   |-- evaluation.py
|   |-- inference.py
|   |-- pipeline_visualization.py
|   `-- utils.py
|-- results/
|   |-- figures/
|   |-- metrics/
|   `-- models/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- app_tkinter.py
`-- main.py
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
- mesures issues de la segmentation.

Le pipeline SVM contient :

1. preparation de `X` et `y` ;
2. normalisation avec `StandardScaler` ;
3. apprentissage avec un SVM RBF ;
4. prediction sur le jeu de test ;
5. sauvegarde du modele et des metriques.

### 3. Estimation de mouvement avec Lucas-Kanade

Une troisieme brique a ete ajoutee, non pas pour remplacer la prediction, mais pour completer le mode camera.

Cette partie sert surtout a verifier si la scene est stable avant de lancer l'analyse. En pratique, cela permet d'eviter de predire sur une image en mouvement, floue ou mal capturee.

## Pourquoi avoir ajoute l'estimation de mouvement

L'ajout de cette fonctionnalite repond a un besoin simple dans l'interface camera :

- savoir si la scene est stable ou en mouvement ;
- ameliorer la qualite de capture avant la prediction ;
- signaler a l'utilisateur quand la piece bouge trop ;
- eviter une analyse sur une frame instable.

Autrement dit, cette estimation ne sert pas a classer la piece en `OK` ou `Defective`, mais a dire si l'image camera est suffisamment fiable pour etre analysee dans de bonnes conditions.

## Pourquoi le choix de Lucas-Kanade

J'ai choisi Lucas-Kanade parce que c'est une methode classique vue en cours, simple a comprendre et tres adaptee a ce type de besoin.

Elle convient bien ici pour plusieurs raisons :

- elle est adaptee aux petits mouvements ;
- elle fonctionne sur des images successives ;
- elle est efficace sur un flux webcam ;
- elle est legere a executer ;
- elle est plus naturelle a integrer dans ce projet que des methodes plus lourdes ou plus complexes.

Dans ce projet, le but n'est pas de faire du suivi d'objet avance, mais d'obtenir une estimation robuste et lisible de la stabilite de la scene. Lucas-Kanade est donc un bon compromis entre simplicite, rapidite et coherence pedagogique.

## Ou Lucas-Kanade intervient dans le projet

L'estimation de mouvement intervient uniquement dans le mode camera de l'application Tkinter.

Elle ne remplace pas la prediction. Elle sert comme :

- information complementaire ;
- controle de stabilite ;
- aide a la capture.

La prediction finale reste faite soit par :

- `Traitement d'image + regles`
- `Traitement d'image + SVM`

Le role de Lucas-Kanade est donc de verifier si la capture est suffisamment stable avant d'autoriser ou de recommander l'analyse.

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

Aucun modele de deep learning n'est utilise.

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

### Ce que l'interface permet deja

L'application permet :

- de choisir explicitement entre `Traitement d'image + regles` et `Traitement d'image + SVM` ;
- d'importer une image depuis le PC ;
- d'ouvrir la camera du PC ;
- de capturer une frame camera ;
- de lancer une prediction sur l'image courante ;
- d'afficher la prediction finale et les statistiques utiles ;
- d'ouvrir des vues secondaires du pipeline.

### Ce que l'interface affiche maintenant avec Lucas-Kanade

En mode camera, l'interface peut maintenant afficher :

- l'etat du mouvement : `Stable`, `En mouvement`, `Instable` ;
- le score de mouvement ;
- le nombre de points suivis ;
- l'amplitude moyenne du deplacement ;
- un indicateur de qualite de capture ;
- un message indiquant si la capture est bonne ou non avant l'analyse.

### Comment cela s'integre

Le suivi Lucas-Kanade est actif surtout sur le flux camera. Il sert a :

- detecter si la scene est stable ;
- informer l'utilisateur si la piece bouge trop ;
- aider a obtenir une meilleure capture ;
- limiter la prediction automatique quand la scene est instable.

Quand le mouvement est trop fort, l'application affiche un message du type :

```text
Stabilisez la piece avant l'analyse.
```

## Comment utiliser la nouvelle partie Lucas-Kanade

1. lancer l'application :

```bash
python app_tkinter.py
```

2. choisir la methode de prediction ;
3. ouvrir la camera ;
4. observer l'etat du mouvement dans le panneau de droite ;
5. stabiliser la piece si necessaire ;
6. lancer ensuite la prediction.

En pratique, cette etape sert de controle avant analyse. Une scene stable donne une capture plus propre et donc une prediction plus fiable.

## Fichiers ajoutes ou modifies

Cette partie du projet repose surtout sur les fichiers suivants :

- `src/motion.py` : nouveau module dedie a l'estimation de mouvement par Lucas-Kanade ;
- `app_tkinter.py` : integration du suivi de mouvement dans le mode camera et affichage des informations dans l'interface ;
- `README.md` : documentation mise a jour pour expliquer cette nouvelle fonctionnalite.

Les autres fichiers de prediction ne changent pas dans leur logique metier.

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

Ce module contient l'estimation de mouvement par Lucas-Kanade pour le flux camera :

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

- coherent avec le contenu du cours ;
- adapte aux petits mouvements ;
- efficace sur un flux webcam ;
- simple a integrer avec OpenCV ;
- utile pour mieux controler la capture.

Limites :

- depend de la presence de points d'interet ;
- moins informatif si la scene est trop uniforme ;
- ne remplace pas une acquisition materielle parfaitement stable.

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
Gate live camera predictions with stability checks
Document Lucas-Kanade integration in README
```
