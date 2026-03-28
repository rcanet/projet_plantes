import streamlit as st
from keras.models import load_model
import tensorflow as tf
from keras import Model
from keras.layers import Conv2D
import numpy as np
import matplotlib.pyplot as plt
import os

# Charger les modèles de LeNet et ResNet50
@st.cache_resource
def load_models():
    lenet = load_model(os.path.join("03_model", "2026-03-15_training_history_lenet_256x256_color.keras"))
    resnet = load_model(os.path.join("03_model", "2026-02-09_training_history_resnet_256x256_color.keras"))
    return lenet, resnet

# Charger le GRAD-CAM
# Définir une fonction `grad_cam` qui prend en entrée une image, un modèle entraîné, et le nom d'une couche de convolution. 
# La fonction doit renvoyer une image superposée avec la carte de chaleur générée par Grad-CAM sans l'afficher.
def grad_cam(img, model, layer_name):
    
    # Choix d'une couche de convolution
    layer = model.get_layer(layer_name)
    
    # Créer un modèle qui génère les sorties de la couche convolutive et les prédictions
    grad_model = Model(inputs = model.input, outputs = [layer.output, model.output])
    
    # Ajout d'une dimension de batch
    image = tf.expand_dims(img, axis = 0)
    
    # Calcul des gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]
    
    
    ## Etape 2 : Pondération des activations
    
    # Calcul des gradients par rapport aux activations de la couche convolutionnelle
    grads = tape.gradient(loss, conv_outputs)
    
    # Moyenne pondérée des gradients pour chaque canal
    pooled_grads = tf.reduce_mean(grads, axis = (0, 1, 2))
    
    ## Etape 3 : Construction de la carte de chaleur
    # Pondération des activations par les gradients calculés
    conv_outputs = conv_outputs[0]  # Supprimer la dimension batch
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalisation de la carte de chaleur
    heatmap = tf.maximum(heatmap, 0)  # Se concentrer uniquement sur les valeurs positives
    heatmap /= tf.math.reduce_max(heatmap)  # Normaliser entre 0 et 1
    heatmap = heatmap.numpy()  # Convertir en tableau numpy pour la visualisation
    
    # Redimensionner la carte de chaleur pour correspondre à la taille de l'image d'origine
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (image.shape[1], image.shape[2])).numpy()
    heatmap_resized = np.squeeze(heatmap_resized, axis=-1) # supprimer la dimension de taille 1 à la fin du tableau heatmap_resized

    # Colorier la carte de chaleur avec une palette (par exemple, "jet")
    heatmap_colored = plt.cm.jet(heatmap_resized)[..., :3] # Récupérer les canaux R, G, B 

    superimposed_image = heatmap_colored * 0.7 + image[0].numpy() / 255.0

    return np.clip(superimposed_image, 0, 1), predicted_class
    
# Afficher les résultats de Grad-CAM pour chaque couche de convolution. 
def show_grad_cam_cnn(images, model):
    
    number_of_images = 1 # images.shape[0]
    conv_layers = [layer.name for layer in model.layers if isinstance(layer, Conv2D)]

    fig = plt.figure(figsize=(5,5))

    for j, layer in enumerate(conv_layers):

        for i in range(number_of_images):

            subplot_index = i + 1 + j * number_of_images
            plt.subplot(len(conv_layers), number_of_images, subplot_index)

            # Obtenir l'image avec la carte de chaleur superposée
            grad_cam_image, predicted_class = grad_cam(images[i], model, layer)
            
            # Afficher l'image avec Grad-CAM
            plt.title(f'Grad-CAM {layer}')
            plt.imshow(grad_cam_image)
            plt.axis("off")

    return fig

# Récupérer les labels
def get_class_names(path_sp = "02_Data/data_spDetection_ready/val/"):
    # Chemin vers ton dossier de test ou d'entraînement
    data_dir =  path_sp
    # Liste les sous-dossiers et les trie par ordre alphabétique
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    return classes