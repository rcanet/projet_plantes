import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from time import sleep
import pandas as pd
import seaborn as sns

path_img_h = "02_data/data_cleaned_25-07-30/Preparation_manuelle_du_jeu_de_donnees/Healthy/"
path_img_d = "02_data/data_cleaned_25-07-30/Preparation_manuelle_du_jeu_de_donnees/Diseased/"
name_img = "1.png"

# img = cv2.imread("02_data/data_cleaned_25-07-30/Preparation_manuelle_du_jeu_de_donnees/Healthy/Black-grass/1.png", cv2.IMREAD_COLOR)

# plt.imshow(curr_img)
# plt.show()

# Gather information for each photos
## Create an empty list to gather the information about the images and a list for unretrieved images
data_list = []
nrdata_list = []
list_unread = []

## List the files with photos in them in both Healthy and Diseased files
list_files_h = os.listdir(path_img_h)
list_files_d = os.listdir(path_img_d)

## List the path to the photos
list_paths_h = [path_img_h + sp for sp in list_files_h]
list_paths_d = [path_img_d + sp for sp in list_files_d]

## Search both the healthy and the diseased files
for i_sp in tqdm(list_paths_h + list_paths_d):

    ## Retrieve the species name
    sp_name = i_sp.split("/")[-1]
    sp_name = sp_name.split("_")[0]

    ## Retrieve the disease name (if diseased)
    disease_name = i_sp.split("___")[-1] if "Disease" in i_sp else None

    ## Gather the names of all photos in the current file
    list_img = os.listdir(i_sp)

    
    ## Gather the img name and the path 
    for i_img in (list_img) :
        path_img = i_sp + '/' + i_img

        ## Read the image
        curr_img = cv2.imread(path_img, cv2.IMREAD_COLOR)

        if curr_img is not None:
            ## Extract the size characteristic from the current image in the list
            line = {
                "sp" : sp_name,
                "name" : i_img, 
                "hauteur" : curr_img.shape[0], 
                "largeur" : curr_img.shape[1],
                "disease" : disease_name,
                "mean_blue" : np.mean(curr_img[:, :, 0]),
                "mean_green" : np.mean(curr_img[:, :, 1]),
                "mean_red" : np.mean(curr_img[:, :, 2])
                }
            data_list.append(line)
        else:
            print(f"[!] Image non lue : {path_img}")
            list_unread.append(path_img)
            nrdata_list.append(path_img)

data_df = pd.DataFrame(data_list)
data_df['nb_pixel'] = data_df.hauteur*data_df.largeur
data_df.head

data_df.to_csv('02_data/25-07-30_data_preliminary.csv', index=None)
pd.DataFrame(list_unread).to_csv('02_data/25-07-30_unread_files.csv', index=None)


