import streamlit as st
import plotly.express as px

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image

######################
### Initialisation ###
######################

# Ajouter le dossier où se trouve le script actuel (modelisation.py) au chemin de recherche
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from FUN import load_models

# Charger les modèles et les historiques des modèles
mod_lenet, mod_resnet = load_models()
df_lenet = pd.read_csv(os.path.join("03_model", "2026-03-15_training_history_lenet_256x256_color.csv"))
cr_lenet = pd.read_csv(os.path.join("03_model", "2026-03-18_cr_lenet_256x256_color.csv"))
cm_lenet = pd.read_csv(os.path.join("04_results", "2026-03-18_confusion_matrix_lenet.csv"), index_col = 0)

df_resnet = pd.read_csv(os.path.join("03_model", "2026-02-09_training_history_resnet_256x256_color.csv"))
cr_resnet = pd.read_csv(os.path.join("03_model", "2026-03-18_cr_resnet_256x256_color.csv"))
cm_resnet = pd.read_csv(os.path.join("04_results", "2026-03-18_confusion_matrix_resnet.csv"), index_col = 0)

# Récupérer les accuracy 
## Trouver la ligne avec la meilleure epoch
best_row_lenet = df_lenet[df_lenet["val_loss"] == df_lenet["val_loss"].min()]
best_row_resnet = df_resnet[df_resnet["val_loss"] == df_resnet["val_loss"].min()]

## Lenet
acc_train_lenet = best_row_lenet["accuracy"].values[0]
acc_val_lenet = best_row_lenet["val_accuracy"].values[0]
acc_test_lenet = cr_lenet.loc[cr_lenet["class"] == "accuracy", "precision"].values[0]

## Resnet
acc_train_resnet = best_row_resnet["accuracy"].values[0]
acc_val_resnet = best_row_resnet["val_accuracy"].values[0]
acc_test_resnet = cr_resnet.loc[cr_resnet["class"] == "accuracy", "precision"].values[0]

## Différence
diff_train = acc_train_resnet - acc_train_lenet
diff_val = acc_val_resnet - acc_val_lenet
diff_test = acc_test_resnet - acc_test_lenet
color = "green" if diff_train >= 0 else "red"
sign = "+" if diff_train >= 0 else ""

# Représentation de l'évolution de l'accuracy et de la valeur de la fonction de perte
def evol_loss(df_history):
    ## Extraire les valeurs de performance du modèle
    train_loss = df_history["loss"]
    val_loss = df_history["val_loss"]

    train_acc = df_history["accuracy"]
    val_acc = df_history["val_accuracy"]

    fig = plt.figure(figsize=(12, 6))

    # Tracer la perte MSE
    plt.subplot(121)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('Perte du modèle par époque ')
    plt.ylabel('Perte ')
    plt.xlabel('Époque')
    plt.legend(['Entraînement', 'Validation'], loc='best')

    # Tracer l'erreur absolue moyenne (MAE)
    plt.subplot(122)
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('Accuracy par époque')
    plt.ylabel('Accuracy')
    plt.xlabel('Époque')
    plt.legend(['Entraînement', 'Validation'], loc='best')

    return fig

#######################
### Choix du modèle ###
#######################

# Menu déroulant pour le modèle
choice_modele = st.selectbox(
    "Modèle sélectionné :",
    ["LeNet", "ResNet50"]
)

####################
### Modèle LeNet ###
####################

if choice_modele == "LeNet":
    '''
    # Modèle basé sur l'architecture LeNet.

    ## Architecture du modèle
    '''

    st.image(os.path.join("04_results", "lenet_architecture.png"), caption = "Visualisation des couches du modèle", use_container_width=True)

    '''
    - Entrée : RGB 256 x 256 pixels
    - Extraction des caractéristiques : 2 couches de convolution (30 et 16 filtres)
    - Minimiser le sur-apprentissage : Dropout + MaxPooling
    - Réduction de dimensions : GlobalAveragePooling
    - Sortie : 26 classes
    - Optimisation : Callbacks (EarlyStopping & Learning Rate)
    '''

    '''
    ## Résultats
    ### Performances

    '''

    el_lenet = evol_loss(df_lenet)

    st.pyplot(el_lenet)

    st.markdown(f'''
    **🎯 Scores d'Accuracy (au minimum de val_loss) :**
    - **Entraînement** : {acc_train_lenet:.3f}
    - **Validation** : {acc_val_lenet:.3f}
    - **Test** : {acc_test_lenet:.3f}
    ''')

    # Graphique de la précision / classe
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Trier les données
    df_sorted = cr_lenet.sort_values(by="precision", ascending=False)
    
    sns.barplot(
    data=df_sorted,
    x="class",
    y="precision",
    ax=ax,
    palette="viridis" # Un dégradé de couleur automatique pour le style
    )

    ax.set_ylim(cr_lenet["precision"].min() - 0.05, cr_lenet["precision"].max() + 0.05)  # Limite axe y
    plt.xticks(rotation=45, ha='right') # Rotation à 45° pour plus de lisibilité
    plt.title("Précision par espèce", fontsize=15) # Titre
    plt.xlabel("Espèces de plantes", fontsize=12) # Nom de l'axe x
    plt.ylabel("Précision", fontsize=12) # Nom de l'axe y
    ax.yaxis.grid(True, linestyle='--', alpha=0.7) # Ajouter la grille

    st.pyplot(fig)

    # Afficher les meilleurs F1-scores et les moins bons
    col_ok, col_notok = st.columns(2)
    with col_ok:
        st.markdown("### ✅ Top Performances")
        # Extraire les valeurs de F1-score
        f1_corn = cr_lenet.loc[cr_lenet["class"] == "Corn", "f1-score"].values[0]
        f1_orange = cr_lenet.loc[cr_lenet["class"] == "Orange", "f1-score"].values[0]
        
        st.markdown(f"""
        Les classes avec les meilleurs **F1-scores** :
        - **Corn (Maïs)** : `{f1_corn:.2f}`
        - **Orange** : `{f1_orange:.2f}`
        """)

    with col_notok:
        st.markdown("### ❗ Flop Performances")
        # Extraire les valeurs de F1-score
        f1_bg = cr_lenet.loc[cr_lenet["class"] == "Black-grass", "f1-score"].values[0]
        f1_potato = cr_lenet.loc[cr_lenet["class"] == "Potato", "f1-score"].values[0]
        
        st.markdown(f"""
        Les classes avec les moins bons **F1-scores** :
        - **Black grass** : `{f1_bg:.2f}`
        - **Potato** : `{f1_potato:.2f}`
        """)

    # Matrice de confusion interactive
    plt.figure(figsize=(10, 10))
    fig = px.imshow(
        cm_lenet,
        text_auto=".2f",                # Affiche les valeurs dans les cases (arrondies à 2 décimales)
        aspect="auto",                  # Ajuste la taille des cases automatiquement
        color_continuous_scale='RdBu_r', # Bleu pour les bons scores, Rouge pour les erreurs
        labels=dict(x="Prédiction", y="Réel", color="Taux")
    )
    st.plotly_chart(fig, use_container_width=True)

    '''
    - Bon score de précision  > 85 %  en diagonale (23/26)
    - Problèmes de confusion : 
      - Garminés : Loose Silky-bent & Black-grass (jusqu'à  20 %)
      - Adventices : Sheperds'Purse & Scentlesse Mayweed (jusqu'à 18 %)
    '''

    '''
    ### Interprétabilité
    '''
    label_inter = ["GRAD-CAM", "SHAP"]

    inter = st.radio(label = "Méthode d'interprétabilité :", 
                     options = label_inter)

    if inter == "GRAD-CAM":
        st.image(os.path.join("04_results", "lenet_grad_best.png"), width=750)
        st.markdown('''
                    - 1ère Convolution : Structure des feuilles
                    - 2nde Convolution : Contour et nervures
                    -> Les bons éléments sont utilisés pour la reconnaissance
                    ''')
    elif inter == "SHAP":
        st.image(os.path.join("04_results", "lenet_shap.png"), width=750)
        st.markdown('''
                    - Identification correcte à la première classe
                    - Pixels important majoritairement sur la feuille
                    MAIS aussi parfois du fond ("Grape")
                    -> Les bons éléments sont utilisés pour la reconnaissance
                    ''')
        
#######################
### Modèle ResNet50 ###
#######################

elif choice_modele == "ResNet50":
    st.markdown('''
    # Modèle issue de *Transfer learning* de l'architecture ResNet50.

    ## Architecture du modèle
    Modèle pré-entrainé de ResNet50 (Microsoft research - 2015) (Residual Network 50 couches) 
    
    * **Pré-entraînement** : plus d'un million d'images (ImageNet)
    * **Feature Extraction**
        * Gel des couches de base
        * Personnalisation des dernières couches
                
    * Couches ajoutées :
        - GAP : réduction des dimensions
        - Dense (256 neurones) : Apprendre les spécificités de nos plantes
        - Dropout : Minimiser l'over-fitting
        - Dense (26 neuroones) : Classification finale
    ''')
    st.info("💡 *Transfer learning* = **Gain de temps et d'efficacité**")

    '''
    ## Résultats
    ### Performances

    '''

    el_resnet = evol_loss(df_resnet)

    st.pyplot(el_resnet)

    st.markdown(f'''
    **🎯 Scores d'Accuracy (au minimum de val_loss) :**
    - **Entraînement** : {acc_train_resnet:.3f} (LeNet {acc_train_lenet:.3f}) :{color}[{sign}{diff_train*100:.2f}%]
    - **Validation** : {acc_val_resnet:.3f} (LeNet {acc_train_lenet:.3f}) :{color}[{sign}{diff_val*100:.2f}%]
    - **Test** : {acc_test_resnet:.3f} (LeNet {acc_train_lenet:.3f}) :{color}[{sign}{diff_test*100:.2f}%]
    ''')

    # Graphique de la précision / classe
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Trier les données
    df_sorted = cr_resnet.sort_values(by="precision", ascending=False)
    
    sns.barplot(
    data=df_sorted,
    x="class",
    y="precision",
    ax=ax,
    palette="viridis" # Un dégradé de couleur automatique pour le style
    )

    ## Personnaliser le graphique
    ax.set_ylim(cr_resnet["precision"].min() - 0.05, cr_resnet["precision"].max() + 0.05) # Limites axe y
    plt.xticks(rotation=45, ha='right') # Rotation à 45° pour plus de lisibilité
    plt.title("Précision par espèce", fontsize=15) # Titre 
    plt.xlabel("Espèces de plantes", fontsize=12) # Nom de l'axe x
    plt.ylabel("Précision", fontsize=12) # Nom de l'axe y
    ax.yaxis.grid(True, linestyle='--', alpha=0.7) # Ajouter la grille

    # Affichage dans Streamlit
    st.pyplot(fig)

    # Remarques de performance
    # Extraire les valeurs pour les top/flop
    ## Garder seulement les classes
    exclude = ['accuracy', 'macro avg', 'weighted avg']
    df_classes_only = cr_resnet[~cr_resnet["class"].isin(exclude)]

    # Top
    top_f1_class = df_classes_only.loc[df_classes_only["f1-score"] == 1, "class"].tolist()
     
    # Flop
    bottom_names = df_classes_only.nsmallest(3, "f1-score")["class"].tolist()
    bottom_scores = df_classes_only.nsmallest(3, "f1-score")["f1-score"].tolist()

    col_ok, col_notok = st.columns(2)
    with col_ok:
        st.markdown("### ✅ **F1-score** de 1")
        st.markdown(f"""
        - **{top_f1_class[0]}** 
        - **{top_f1_class[1]}** 
        - **{top_f1_class[2]}** 
        """)

    with col_notok:
        st.markdown("### ⚠️ Classes les plus diff.")
        st.markdown(f"""
        - **{bottom_names[0]}** : {bottom_scores[0]:.3f}
        - **{bottom_names[1]}** : {bottom_scores[1]:.3f}
        - **{bottom_names[2]}** : {bottom_scores[2]:.3f}
        """)

    # Matrice de confusion interactive
    plt.figure(figsize=(10, 10))
    fig = px.imshow(
        cm_resnet,
        text_auto=".2f",                # Affiche les valeurs dans les cases (arrondies à 2 décimales)
        aspect="auto",                  # Ajuste la taille des cases automatiquement
        color_continuous_scale='RdBu_r', # Bleu pour les bons scores, Rouge pour les erreurs
        labels=dict(x="Prédiction", y="Réel", color="Taux")
    )
    st.plotly_chart(fig, use_container_width=True)

    '''
    - **Performance Globale** : Excellente précision avec 19 classes dépassant 97 %.
    - **Analyse des erreurs (Biais morphologiques)** :
      - **Graminées** : Confusion persistante entre *Loose Silky-bent* et *Black-grass* (12 % et 27 % d'erreur).
      - **Adventices** : Légères erreurs sur *Shepherd's Purse* et *Scentless Mayweed* (11 % et 2 % d'erreur).
    - **Points Forts** : Robustesse sur les cultures principales (Maïs, Tomate, Soja).
    '''

    '''
    ### Interprétabilité
    '''

    st.image(os.path.join("04_results", "resnet_shap.png"), width=750)
    st.markdown('''
                - Identification correcte à la première classe
                - Pixels important majoritairement sur la feuille
                MAIS aussi parfois du fond ("Grape" & "Peach")
                -> Les bons éléments sont utilisés pour la reconnaissance
                ''')
    

