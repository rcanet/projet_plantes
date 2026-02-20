import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Resizing
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

classes = os.listdir("02_data/data_spDetection_ready/train")
print(len(classes))  # combien de sous-dossiers
print(classes)       # noms des espèces

# path_img = "02_data/sample_data/"
path_train = "02_data/data_spDetection_ready/train"
path_test = "02_data/data_spDetection_ready/val"


train_ds = image_dataset_from_directory(path_train, 
                                        label_mode = 'categorical',
                                        image_size = (256, 256),
                                        seed=42, 
                                        batch_size=32, 
                                        color_mode = 'rgb')

val_ds = image_dataset_from_directory(path_test,
                                      label_mode = 'categorical',
                                      image_size = (256, 256),
                                      seed=42,
                                      batch_size=32, 
                                      color_mode = 'rgb',
                                      shuffle = False)

# Définition d'un callback afin d'éviter l'overfitting
earlystop = EarlyStopping(monitor = 'val_loss',                          
						  min_delta = 0,
                          patience = 10,
                          verbose = 1,
                          restore_best_weights = True)

reducelr = ReduceLROnPlateau(monitor = 'val_loss',
                                min_delta = 0.01,
                                patience = 5,
                                factor = 0.5, 
                                cooldown = 2,
                                verbose = 1)
wanted_size = 128

# Instancier une couche d'entrée qui définit les dimensions des données en entrées
inputs = Input(shape = (256, 256, 3), name = "Input")

# Couche de modification
normalization_layer = Rescaling(1./255)
resizing_1 = Resizing(wanted_size, wanted_size)

# couches de convolution/Dense
convolution_1 = Conv2D(filters = 30,
                     kernel_size = (5, 5),
                    padding = 'valid',
                    activation = 'relu')
maxpooling_1 = MaxPooling2D(pool_size = (2, 2), name = 'max_pooling_layer_1')
convolution_2 = Conv2D(filters = 16,
                     kernel_size = (3, 3),
                    padding = 'valid',
                    activation = 'relu')
maxpooling_2 = MaxPooling2D(pool_size = (2, 2), name = 'max_pooling_layer_2')
droupout_layer = Dropout(rate = 0.2)
flattening_layer = GlobalAveragePooling2D() # GlobalAverage2d
dense_layer_1 = Dense(units = 128, activation = "relu")
dense_layer_2 = Dense(units = 26, activation = "softmax")

# Extraction des caractéristiques
x = normalization_layer(inputs)
x = resizing_1(x)
x = convolution_1(x)
x = maxpooling_1(x)
x = convolution_2(x)
x = maxpooling_2(x)
x = droupout_layer(x)

# Applatissement
x = flattening_layer(x)

# Couches dense pour la prédiction 
x = dense_layer_1(x)
outputs = dense_layer_2(x)

mod_lenet = Model(inputs=inputs, outputs=outputs)

# Compile
mod_lenet.compile(loss = "categorical_crossentropy", 
                  optimizer = Adam(learning_rate=0.001),  
                  metrics = ["accuracy"])

# Entrainement du model
training_history = mod_lenet.fit(train_ds, validation_data = val_ds, 
                                 epochs = 50,
                                 callbacks = [earlystop,
                                              reducelr])

# Sauvegarde du modèle et de l'historique
today_date = datetime.now().strftime("%Y-%m-%d")
path_model = "03_model"
name_model = f"_training_history_lenet_{wanted_size}x{wanted_size}_color"
full_path = os.path.join(path_model, today_date + name_model)

mod_lenet.save(full_path + '.keras')

history_df = pd.DataFrame(training_history.history)
history_df.to_csv(full_path + ".csv", index=False)

# Chargement du modèle et de l'historique
mod_lenet = load_model(os.path.join(path_model, "2026-02-08_training_history_lenet_128x128_color.keras"))
df_lenet = pd.read_csv(os.path.join(path_model, "2026-02-08_training_history_lenet_128x128_color.csv"))

## Afficher l'évolution de la fonction de perte au fil de l'entraînement
def evol_loss(df_history):
    ## Extraire les valeurs de performance du modèle
    train_loss = df_history["loss"]
    val_loss = df_history["val_loss"]

    train_acc = df_history["accuracy"]
    val_acc = df_history["val_accuracy"]

    plt.figure(figsize=(20, 8))

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
    plt.ylabel('Accuracy absolue moyenne')
    plt.xlabel('Époque')
    plt.legend(['Entraînement', 'Validation'], loc='best')

    plt.show();

evol_loss(df_lenet)

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Afficher le Rapport de classification
## Récupération des classes de tests réelles et des prédictions 
y_class_test = np.concatenate([y for x, y in val_ds], axis=0)
y_class_test = np.argmax(y_class_test, axis=1)

pred_test = mod_lenet.predict(val_ds)
pred_class_test = np.argmax(pred_test, axis=1)
name_class = val_ds.class_names

## Affichage du rapport
print(classification_report(y_class_test, pred_class_test, target_names = name_class))

## On regarde plus en détail le tableau du CR
cr_lenet = classification_report(y_class_test, pred_class_test, target_names = name_class,
                                 output_dict=True)
df_cr_lenet = pd.DataFrame(data = cr_lenet).transpose()

### Top & Flop 3
print(f'Les trois espèces les moins reconnues sont : {df_cr_lenet.sort_values(ascending = True, by = "precision").index[0:3].tolist()}')
print(f'Les trois espèces les plus reconnues sont : {df_cr_lenet.sort_values(ascending = False, by = "precision").index[0:3].tolist()}')

### Top & Flop
df_cr_lenet.sort_values(ascending = True, by = "precision")
df_cr_lenet.sort_values(ascending = False, by = "precision")

# Afficher la matrice de confusion
plt.figure(figsize=(15, 15))
cnf_matrix = confusion_matrix(y_class_test, pred_class_test, normalize='true')
sns.heatmap(cnf_matrix, cmap='Blues', annot=True, cbar=False, fmt=".2f",
            xticklabels = name_class, 
            yticklabels = name_class)
plt.show()

#########################
### Transfer Learning ###
#########################

# 1. Initialisation 
path_model = "03_model"
if not os.path.exists(path_model):
    os.makedirs(path_model)

train_ds = image_dataset_from_directory(path_train, 
                                        label_mode = 'categorical',
                                        image_size = (256, 256),
                                        seed=42, 
                                        batch_size=32, 
                                        color_mode = 'rgb')

val_ds = image_dataset_from_directory(path_test,
                                      label_mode = 'categorical',
                                      image_size = (256, 256),
                                      seed=42,
                                      batch_size=32, 
                                      color_mode = 'rgb')


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base_model.trainable = False # Geler les couches de base pour ne pas perdre les poids ImageNet

# 1. On définit l'entrée qui correspond à la taille du dataset (256x256)
inputs = layers.Input(shape=(256, 256, 3)) # Entrée du jeu de donnée
#x = layers.Resizing(128, 128)(inputs)       # Redimensionnement
x = preprocess_input(inputs)                   # Prétraitement ResNet
x = base_model(x, training=False)         # Passage dans ResNet
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(26, activation='softmax')(x)

# 6. Création du modèle final
model_resnet = Model(inputs, outputs)

# 7. Compilation avec un Learning Rate adapté au Transfer Learning
model_resnet.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrainement du model
training_history = model_resnet.fit(train_ds, validation_data = val_ds, 
                                    epochs = 51,
                                    callbacks = [earlystop,
                                                 reducelr])

# Sauvegarde du modèle et de l'historique
today_date = datetime.now().strftime("%Y-%m-%d")
path_model = "03_model"
name_model = "_training_history_resnet_64x64"
full_path = os.path.join(path_model, today_date + name_model)

model_resnet.save(full_path + '.keras')

history_df = pd.DataFrame(training_history.history)
history_df.to_csv(full_path + ".csv", index=False)


# Chargement du modèle et de l'historique
mod_resnet = load_model(os.path.join(path_model, "2026-02-09_training_history_resnet_256x256_color.keras"))
df_resnet = pd.read_csv(os.path.join(path_model, "2026-02-09_training_history_resnet_256x256_color.csv"))

# Suivi graphique de l'accuracy et de la fonction de perte
evol_loss(df_resnet)


# Afficher le Rapport de classification
## Récupération des classes de tests réelles et des prédictions 
y_class_test = np.concatenate([y for x, y in val_ds], axis=0)
y_class_test = np.argmax(y_class_test, axis=1)

pred_test_resnet = mod_resnet.predict(val_ds)
pred_class_test_resnet = np.argmax(pred_test_resnet, axis=1)
name_class = val_ds.class_names

## Affichage du rapport
print(classification_report(y_class_test, pred_class_test_resnet, target_names = name_class))

## On regarde plus en détail le tableau du CR
cr_resnet = classification_report(y_class_test, pred_class_test_resnet, target_names = name_class,
                                 output_dict=True)
df_cr_resnet = pd.DataFrame(data = cr_resnet).transpose()

### Top & Flop 3
print(f'Les trois espèces les moins reconnues sont : {df_cr_resnet.sort_values(ascending = True, by = "precision").index[0:3].tolist()}')
print(f'Les trois espèces les plus reconnues sont : {df_cr_resnet.sort_values(ascending = False, by = "precision").index[0:3].tolist()}')

### Top & Flop
df_cr_resnet.sort_values(ascending = True, by = "precision")
df_cr_resnet.sort_values(ascending = False, by = "precision")

# Afficher la matrice de confusion
plt.figure(figsize=(15, 15))
cnf_matrix = confusion_matrix(y_class_test, pred_class_test_resnet, normalize='true')
sns.heatmap(cnf_matrix, cmap='Blues', annot=True, cbar=False, fmt=".2f",
            xticklabels = name_class, 
            yticklabels = name_class)
plt.show()

################
### GRAD-CAM ###
################

# Stocker les images et les labels
X = np.array([img.numpy() for img, _ in val_ds.take(2)][0]).astype(np.uint8)
y = np.array([label.numpy() for _, label in val_ds.take(2)][0])

# Sélection des images spécifiques
images = X[:4]
labels = y[:4]

# Définir une fonction `grad_cam` qui prend en entrée une image, un modèle entraîné, et le nom d'une couche de convolution. La fonction doit renvoyer une image superposée avec la carte de chaleur générée par Grad-CAM sans l'afficher.
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
    number_of_images = images.shape[0]
    conv_layers = [layer.name for layer in model.layers if isinstance(layer, Conv2D)]

    plt.figure(figsize=(16,16))

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

    plt.show()

show_grad_cam_cnn(images, mod_lenet)

