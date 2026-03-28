# :herb: Reconnaissance d'espèces végétales :herb:
## Description générale
Ce projet vise à développer un modèle de deep learning capable d’identifier automatiquement l’espèce végétale présente dans une image.
Il s’appuie sur plus de 58 000 images issues de Kaggle et couvre 26 espèces végétales.

Ce projet est réalisé dans le cadre de ma formation de Data Scientist avec la structure Liora.

## :bar_chart: Jeu de données :bar_chart:
Les données proviennent de :
- https://www.kaggle.com/vbookshelf/v2-plant-seedlings-dataset
- https://www.kaggle.com/abdallahalidev/plantvillage-dataset
*Le jeu de donnée n'est pas inclus dans le repository et doit être ajoutée dans le dossier '02_data/data_brute'*

Le dataset contient :
- 26 classes
- Environ 58 000 images
- Un déséquilibre significatif entre certaines espèces

## :brain: Méthodologie :brain:
- Analyse descriptive des données brutes
    → [01_Scripts/01_data_exploration.py](01_Scripts/01_data_exploration.py)
- Prétraitement des données 
    → [01_Scripts/hash_and_search.py](01_Scripts/hash_and_search.py)
    → [01_Scripts/02_data_pretreatment.py](01_Scripts/02_data_pretreatment.py)
- Modélisation
    → [01_Scripts/03_modelisation.py](01_Scripts/03_modelisation.py)
- Présentation par streamlit
    → [01_Scripts/04_streamlit.py](01_Scripts/04_streamlit.py)
    → [01_Scripts/streamlit/](01_Scripts/streamlit/)

## Résultats
Deux modèles de reconnaissance ont été crées. L'un est un modèle entrainé entièrement à partir du jeu de donnée, tandis que l'autre est issue du *Transfer Learning* de ResNet. Par soucis d'espace de stockage, aucun modèle n'est présent dans [03_model/](03_model/).

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/rcanet/projet_plantes.git
cd projet_plantes
pip install -r requirements.txt
```

## Exemple d'utilisation

