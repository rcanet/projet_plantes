import streamlit as st
from PIL import Image
import numpy as np
from keras.applications.resnet50 import preprocess_input
import os
import sys

# Ajouter le dossier où se trouve le script actuel (modelisation.py) au chemin de recherche
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from FUN import load_models, grad_cam, show_grad_cam_cnn, get_class_names

'''
## Démonstration
'''

# Fonction de pré-traitement de l'image téléchargée
def pre_treat_demo(image):
    img_pil = Image.open(uploaded_file)
    img_resized = img_pil.resize((256, 256))
    img_array = np.array(img_resized)
    images_batch = np.array([img_array])
    return images_batch

# Fonction de pré-traitement pour ResNet50
def prepare_for_resnet(uploaded_file):
    # 1. Charger et redimensionner
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((256, 256))
    
    # 2. Convertir en array (0-255)
    img_array = np.array(img)
    
    # 3. Ajouter la dimension batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # 4. APPLIQUER LE PRÉTRAITEMENT IDENTIQUE À L'ENTRAÎNEMENT
    # C'est ici que la magie opère (conversion BGR + centrage)
    img_preprocessed = preprocess_input(img_array)
    
    return img_preprocessed


# Charger les modèles (déjà en mémoire)
mod_lenet, mod_resnet = load_models()

# Classes
labels = get_class_names()

uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Pré-traiter l'image (redimensionner, array)
    # --- ÉTAPE 1 : Préparation de l'image de base ---
    img_pil = Image.open(uploaded_file).convert('RGB')
    img_resized = img_pil.resize((256, 256))
    img_array = np.array(img_resized)
    
    # image_raw : format (1, 256, 256, 3) | Pixels 0-255
    # C'est ce qu'on utilise pour LeNet, Grad-CAM et l'affichage st.image
    image_raw = np.expand_dims(img_array, axis=0)
    image_batch = np.expand_dims(img_array, axis=0)

    # --- ÉTAPE 2 : Préparation spécifique pour ResNet ---
    # On applique le prétraitement sur une COPIE pour ne pas corrompre l'originale
    image_resnet = preprocess_input(image_raw.copy())

    col1, col2 = st.columns(2)

    with col1:
        st.image(img_resized, caption='Image chargée', use_container_width=True)

    with col2:
        pred_test_resnet = mod_resnet.predict(image_resnet)
        idx_resnet = np.argmax(pred_test_resnet)
        confiance_resnet = np.max(pred_test_resnet) * 100

        pred_test_lenet = mod_lenet.predict(image_batch)
        idx_lenet = np.argmax(pred_test_lenet)
        confiance_lenet = np.max(pred_test_lenet) * 100
            
        '''Modèle LeNet'''
        st.metric(label="Espèce identifiée", value=labels[idx_lenet])
        st.progress(int(confiance_lenet))
        st.write(f"Indice de confiance : **{confiance_lenet:.2f}%**")

        '''Modèle ResNet50'''
        st.metric(label="Espèce identifiée", value=labels[idx_resnet])
        st.progress(int(confiance_resnet))
        st.write(f"Indice de confiance : **{confiance_resnet:.2f}%**")


    
    
    lenet_grad = show_grad_cam_cnn(image_batch, mod_lenet)
    st.pyplot(lenet_grad)
