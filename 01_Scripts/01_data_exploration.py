import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import shutil
from datetime import datetime

##############################################
### Données brutes -> données à pré-traité ###
##############################################
# Créer de l'architecture pour les analyse spréliminaires
# data_preliminary/
# - Diseased/
# - Healthy/

path_plantvillage = "02_data/data_brute/plantvillage-dataset/color/"
path_seedings = "02_data/data_brute/v2-plant-seedings-dataset/"

new_path = "02_data/data_preliminary/"
if not os.path.exists(new_path):
    os.makedirs(new_path)
    os.makedirs(new_path + "Healthy/")
    os.makedirs(new_path + "Diseased/")

# Copier les images saines du dataset Seedings vers le dossier Healthy si elles n'existent pas
for x_folder in tqdm(os.listdir(path_seedings)):

    if not os.path.exists(new_path + "Healthy/" + x_folder) :
        shutil.copytree(path_seedings + x_folder, new_path + "Healthy/")
    else:
        print(f"Le dossier {x_folder} existe déjà dans {new_path + "Healthy/"}")


# Copier les images saines et malades du dataset plantvillage vers le dossier Healthy ou Diseased s'il n'existe pas
for x_folder in tqdm(os.listdir(path_plantvillage)):

    # Vérifier si l'image provient d'un dossier "Healthy"
    if "healthy" in x_folder :
        new_name = x_folder.split("___")[0]

        # Copier dans "Healthy" seulement s'il n'existe pas déjà
        if not os.path.exists(new_path + "Healthy/" + new_name):
            shutil.copytree(path_plantvillage + x_folder, new_path + "Healthy/" + new_name)

    elif "healthy" not in x_folder and not os.path.exists(new_path + "Diseased/" + x_folder):
        shutil.copytree(path_plantvillage + x_folder, new_path + "Diseased/" + x_folder)
    

###################################
### Récapitulatif des métriques ###
###################################

path_img_h = "02_data/data_preliminary/Healthy/"
path_img_d = "02_data/data_preliminary/Diseased/"
name_img = "1.png"

# img = cv2.imread("02_data/data_cleaned_25-07-30/Preparation_manuelle_du_jeu_de_donnees/Healthy/Black-grass/1.png", cv2.IMREAD_COLOR)

# plt.imshow(curr_img)
# plt.show()

# Initialiser les listes de résltat
data_list = [] # liste :  variables de taille, flou, moyenne de couleurs
list_unread = [] # liste : chemins des images non lues

# C*Extraire les chemins des espèces (Healthy & Diseased)
list_files_h = os.listdir(path_img_h)
list_files_d = os.listdir(path_img_d)
list_paths_h = [path_img_h + sp for sp in list_files_h]
list_paths_d = [path_img_d + sp for sp in list_files_d]

# -------------------------------------------------
# Parcours du dataset pré-traité (Healthy + Diseased)
# Objectif : extraire pour chaque image :
# - nom de l'espèce
# - nom de la maladie si existante
# - dimensions
# - moyenne des couleurs
# - flou (variance Laplacian)
# -------------------------------------------------
for species_path in tqdm(list_paths_h + list_paths_d):

    # Récupérer le nom des espèces
    species_name = species_path.split("/")[-1]
    species_name = species_name.split("_")[0]

    # Récupérer le nom de la maladie (si malade)
    disease_name = species_path.split("___")[-1] if "Disease" in species_path else None

    # Récupérer la liste des images pour ce dossier (sp + maladie)
    list_img = os.listdir(species_path)

    
    # Pour chaque image du dossier :
    # - lire l'image
    # - calculer le flou (variance Laplacienne)
    # - extraire les stats de couleurs et dimensions
    for image_name in (list_img) :

        path_img = os.path.join(species_path, image_name) 
        curr_img = cv2.imread(path_img, cv2.IMREAD_COLOR) # Lecture de l'image en couleurs
        laplacian_var = cv2.Laplacian(curr_img, cv2.CV_64F).var() # Calculer le niveau de floue

        if curr_img is not None:

            line = {
                "sp" : species_name,
                "name" : image_name, 
                "hauteur" : curr_img.shape[0], 
                "largeur" : curr_img.shape[1],
                "disease" : disease_name,
                "mean_blue" : np.mean(curr_img[:, :, 0]),
                "mean_green" : np.mean(curr_img[:, :, 1]),
                "mean_red" : np.mean(curr_img[:, :, 2]),
                "laplacian_var" : laplacian_var
                }
            data_list.append(line)
        else:
            print(f"[!] Image non lue : {path_img}")
            list_unread.append(path_img)

data_df = pd.DataFrame(data_list)
data_df['nb_pixel'] = data_df.hauteur*data_df.largeur
data_df.head()

# Enregistrer le Dataframe
today = datetime.today().strftime("%Y-%m-%d")
data_df.to_csv('02_data/' + today + '_data_preliminary.csv', index=None)
pd.DataFrame(list_unread).to_csv('02_data/' + today + '_unread_files.csv', index=None)

##################################
### Représentations Graphiques ###
##################################

# Niveau de floue
plt.hist(data_df[data_df["laplacian_var"] < 1000]["laplacian_var"], bins = 50);
plt.show();
print(data_df['laplacian_var'].quantile(q = [0.05, 0.1, 0.15, 0.20, 0.25]))

data_df['is_diseased'] = data_df['disease'].notna().astype(int)


# Graphic representation
## Distribution des images
order_distribution = data_df['sp'].value_counts().index
counts = data_df['sp'].value_counts().values
sns.catplot(data = data_df, y = 'sp', kind = 'count', order = order_distribution, hue = 'is_diseased')
for count, sp in zip(counts, order_distribution): # Annotation du nombre d'image pour chaque espèce
    plt.annotate(str(count), 
                 xy = ((count + 1), sp),
                 va = 'center')
plt.xlabel("Nombre d'images")
plt.ylabel("Espèces")
plt.show();


## Distribution des tailles
plt.hist(data = data_df, x = 'nb_pixel')
plt.show();
data_df.nb_pixel.value_counts(normalize = True)

## Heatmap de la largeur et la longueur
pivot_table = data_df.groupby(["hauteur", "largeur"]).size().unstack(fill_value=0)
data_df.value_counts(subset=["hauteur", "largeur"], normalize = True)  
data_df.value_counts(subset=["hauteur", "largeur"])

### Afficher la heatmap
sns.heatmap(np.log(pivot_table), cmap="viridis")
plt.show()

## Distribution des espèces  256 x 256
data256_df = data_df.loc[(data_df["hauteur"] == 256) & (data_df["largeur"] == 256)]
order_distribution = data256_df['sp'].value_counts().index
counts = data256_df['sp'].value_counts().values
sns.catplot(data = data256_df, y = 'sp', kind = 'count', order = order_distribution, hue = 'is_diseased')
plt.xlabel("Nombre d'images")
plt.ylabel("Espèces")
plt.show();

## Distribution RGB
plt.hist(data_df["mean_red"], bins=50, color="red", alpha=0.7)
plt.show()


## Et-ce qu'il y a des doublons ? 
data_df["id"] = data_df.sp.astype(str) + data_df.name.astype(str)
print(f"Il y a {data_df.id.duplicated().sum()} doublon(s).")

## Moyenne des canaux RGB
fig = plt.figure(figsize=(10,10))

plt.subplot(311)
sns.violinplot(data = data_df, y = "mean_blue", x = "sp", color = 'blue')
plt.xticks([])
plt.ylabel("Moyenne Bleu")
plt.xlabel('')

plt.subplot(312)
sns.violinplot(data = data_df, y = "mean_red", x = "sp", color = 'red')
plt.xticks([])
plt.ylabel("Moyenne Rouge")
plt.xlabel('')

plt.subplot(313)
sns.violinplot(data = data_df, y = "mean_green", x = "sp",color = 'green')
plt.xticks(rotation=45)
plt.ylabel("Moyenne Vert")
plt.show();