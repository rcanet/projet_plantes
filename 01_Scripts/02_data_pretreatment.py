import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import keras
import os
import random
from tqdm import tqdm
import time
import shutil
from datetime import datetime
from sklearn.model_selection import train_test_split
import splitfolders

##################################################################
### Structurer un dossier sous la forme : data_to_use/SP/x.jpg ###
##################################################################

def folder_architecture(path_input, path_output):
    """
    Cette fonction organise le dossier de données pour le rendre sous format sp au lieu de santé/sp

    Paramètres : 
        - path_input : Chemin des images santé/sp
        - path_output : Chemin vers les images ordonnées

    Renvoie : 
    Le jeu de données sous la forme appropriée sp/
    """    
    start_time = time.time() # heure de début

    # Extraction des classes (espèces)
    files_healthy = os.listdir(path_input + "Healthy/")
    sp_healthy = [x.split("_")[0] for x in files_healthy]

    files_diseased = os.listdir(path_input + "Diseased/")
    sp_diseased = [x.split("_")[0] for x in files_diseased]

    total_sp = list(set(sp_healthy + sp_diseased))

    # Création du nouveau dossier ainsi que des dossiers par espèce
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    for i_sp in total_sp:
        if not os.path.exists(path_output + "/" + i_sp):
            os.makedirs(path_output + "/" +  i_sp)

    # Copie des images du dossier de base vers la nouvelle arborescence
    list_path = [path_input + "Healthy/" + i for i in files_healthy]

    for health_status in ("Healthy", "Diseased"):
        files_sp = files_healthy if health_status == "Healthy" else files_diseased

        for i_file in tqdm(files_sp):
            # Extraction de l'espèce et du chemin du dossier
            i_sp = i_file.split("_")[0]
            i_path = path_input + health_status + "/" + i_file

            if health_status == "Healthy":
                prefix = "_".join([health_status, i_sp, ""]) # prefix pour ajouter l'espèce et le statut sain 
            else: 
                disease_name = i_file.split("___")[1]
                prefix = "_".join([health_status, i_sp, "", disease_name, "_"])

            for i_img in os.listdir(i_path):
                src = i_path + "/" + i_img
                dest = "/".join([path_output, i_sp, prefix + i_img])
                shutil.copy2(src, dest)

    end_time = time.time() # heure de fin 
    elapsed_time = end_time - start_time
    print(f"Temps écoulé : {elapsed_time} secondes")

path_input = "02_data/data_preliminary/"
base_path = "02_data/data_sp_detection"
folder_architecture(path_input, base_path)

########################################################################################
### Enlever les fichiers déjà augmentés pour éviter une redondance de l'augmentation ###
########################################################################################
char_augmentation = ["deg", "flip", "newPixel"]
list_files = os.listdir(base_path)
compteur = 0

for x_file in tqdm(list_files):
    folder_path = os.path.join(base_path, x_file)

    for x_img in os.listdir(folder_path):
        if any(c in x_img for c in char_augmentation):
            os.remove(os.path.join(folder_path, x_img))
            compteur += 1
print(f"Supprimé : {compteur} images")

####################################################################
### Pré-traitement des images : Rescaling et pixel normalisation ###
####################################################################

# Create an empty list to gather the information about the images and a list for unretrieved images
data_list = []

# Parameters to rescale the pictures to a 256x256 format (penser ) 254 x 254 par rapport à resnet
new_width = 256
new_height = 256

## Rescale the out of scale images and create a dataframe to resume data
for i_sp in tqdm(os.listdir(base_path)):

    ## Liste de toutes les images pour l'espèce en cours
    list_img = os.listdir(os.path.join(base_path, i_sp))

    ## Check the size of the picture
    for i_img in list_img :

        ## Retrieve the disease name (if diseased)
        disease_name = i_img.split("__")[1] if "Diseased" in i_img else np.nan

        path_img = os.path.join(base_path, i_sp, i_img)

        ## Read the image in color
        curr_img = cv2.imread(path_img, cv2.IMREAD_C)

        if curr_img is None:
            print(f"[!] Image non lue : {path_img}")
            continue

        # Filtre : Niveau de floue - basé que le quantile 10% (pic) 
        laplacian_var = cv2.Laplacian(curr_img, cv2.CV_64F).var() # Calculer le niveau de floue

        #if laplacian_var < 162.50:
        #    print(f"[!] Image inférieure au seuil de floue : {path_img}")

        ## Assess the size [height, width] & create the path to save the picture 
        size = [curr_img.shape[0], curr_img.shape[1]]
        

        if all(i_dim == 256 for i_dim in size):
            var_rescaled = 0
            #curr_img_normalised = curr_img/255 # Normalisation de l'image entre 0 et 1
            #ghost_error = cv2.imwrite(path_img, curr_img_normalised)
            ghost_error = cv2.imwrite(path_img, curr_img)
                
        else:
            var_rescaled = 1
            resized_image = cv2.resize(curr_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR) # Redimensionnalisation de l'image
            #resized_img_normalised = resized_image/255 # Normalisation de l'image entre 0 et 1
            #ghost_error = cv2.imwrite(path_img, resized_img_normalised)
            ghost_error = cv2.imwrite(path_img, resized_image)

################################
### Séparation en Train/Test ###
################################
# Sépare le dossier source en 'train' et 'test' dans un nouveau dossier 'output'
splitfolders.ratio("02_data/data_sp_detection/", output="02_data/data_spDetection_ready", seed=42, ratio=(.8, .2))


#########################
### Data Augmentation ###
#########################

# Fonction permettant de faire de la data_augmentation 
def randomFlip(img):
    # 1 = Horizontal, 0 = Vertical, -1 = Les deux
    return cv2.flip(img, 1)

def randomRotation(img):
    angle = random.uniform(-20, 20) # Rotation entre -20° et +20°
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def randomContrast(img):
    # Modification du contraste et de la luminosité (Alpha/Beta)
    alpha = random.uniform(0.8, 1.2) # Contraste
    beta = random.randint(-10, 10)   # Luminosité
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def picture_augmentation(target, path_img):
    """
    Cette fonction vérifie le nombre de photos de chaque espèce et procède à l'augmentation des images en accord avec 
    le nombre de photos désirées et le nombre de photo de chaque espèce.

    Paramètres : 
        - target : Nombre d'images désiré par classe
        - path_img : Chemin relatif des images (adapté à une architure avec un dossier pour les plantes malades et un autre pour celles qui sont saines)

    Renvoie : 
    Le jeu de données augmenté.
    """
    random_da = [randomFlip,
                 randomRotation, 
                 randomContrast]

    for x_sp in os.listdir(path_img):

        path_sp = os.path.join(path_img, x_sp)
        if not os.path.isdir(path_sp): continue # Sécurité

        # Etape 1 : Compter le nombre d'image par espèce
        list_img = os.listdir(path_sp)
        nb_img = len(list_img)

        ## Etape 2.a : Si on a plus d'image que le nombre cible, en supprimer aléatoirement jusqu'à nb. cible
        if nb_img > target:
            nb_to_remove = nb_img - target
            files_to_remove = random.sample(list_img, nb_to_remove)

            for id in files_to_remove:
                path_to_remove = os.path.join(path_sp, id)
                os.remove(path_to_remove)
            print(f"Classe {x_sp} réduite à {target}")

        ## Etape 2.b : Si on a moins d'image que le nb. cible, faire de la data augmentation        
        elif nb_img < target:
            nb_to_add = target - nb_img

            ### Calcul du nombre de transformations complètes par image originale
            ### ex. (2500 - 250)/250 = 9 transformations par image
            ratio_da = nb_to_add // nb_img
            reste_da = nb_to_add % nb_img
            print(f"Augmentation {x_sp}: {ratio_da} versions/image + {reste_da} extras.")

            ### Etape 2.b.1 : Data augmentations sur TOUTES les images d'une espèces
            for img_name in list_img:
                img_path = os.path.join(path_sp, img_name)
                img = cv2.imread(img_path)
                
                for j in range(ratio_da):
                    func = random.choice(random_da)
                    img_aug = func(img)
                    path_new_img = os.path.join(path_sp, f"DA_batch_{j}__{img_name}")
                    cv2.imwrite(path_new_img, img_aug)
            
            ### Etabe 2.b.2 : Data augmentations sur le reste pour atteindre 2500 images
            if reste_da > 0:
                images_pour_reste = random.sample(list_img, reste_da)
                for img_name in images_pour_reste:
                    img_path = os.path.join(path_sp, img_name)
                    img = cv2.imread(img_path)
                    
                    func = random.choice(random_da)
                    img_aug = func(img)
                    path_new_img = os.path.join(path_sp, f"DA_reste__{img_name}")
                    cv2.imwrite(path_new_img, img_aug)

target = 2500
path_augmentation = "02_data/data_spDetection_ready/train/"
picture_augmentation(target, path_augmentation)


# Vérification de la data augmentation
def plot_distribution(path_img):
    classes = []
    counts = []
    
    for x_sp in sorted(os.listdir(path_img)):
        path_sp = os.path.join(path_img, x_sp)
        if os.path.isdir(path_sp):
            classes.append(x_sp)
            counts.append(len(os.listdir(path_sp)))
            
    plt.figure(figsize=(12, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Nombre d\'images')
    plt.title('Distribution des classes')
    plt.tight_layout()
    plt.show()

# Utilisation
plot_distribution("02_data/data_spDetection_ready/train")