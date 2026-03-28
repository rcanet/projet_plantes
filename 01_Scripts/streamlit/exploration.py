import pandas as pd
import numpy as np
import os 
import random
import cv2
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

""

data_df = pd.read_csv("02_data/2026-03-16_data_preliminary.csv")
data_df["size_256"] = np.where((data_df["hauteur"] == 256) & (data_df["largeur"] == 256), 1, 0)

'''# Présentation du jeu de données'''
############################
### Echantillon d'images ###
############################
path_img = "02_data/data_preliminary/"

# Extraire les chemins des images
nb_images = 5
images = [
        os.path.join(root, f)
        for root, _, files in os.walk(path_img)
        for f in files
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

# Afficher une image lorsqu'on appuie sur le bouton
if st.button("Génération d'images", type="primary", icon="🌱", use_container_width=True):
    
    cols = st.columns(5)

    # Piocher 5 images au hasard
    sample_images = random.sample(images, 5)

    for i, img_path in enumerate(sample_images):

        # Extraire de l'espèce
        species = os.path.basename(os.path.dirname(img_path)).split('__')[0]
        
        # Afficher dans la colonne correspondante
        with cols[i]:
            st.image(img_path, caption=species, use_container_width=True)

f'''
- Source des images : Kaggle
- {len(data_df)} images de plantes pour {len(data_df["sp"].unique())} classes/espèces
- {round(data_df.size_256.value_counts(normalize = True).values[0], 3)*100} % des images font 256 x 256 pixels
'''

################################
### Distribution des classes ###
################################
'''
# Distribution dans les 26 classes 
'''
dist_sp = pd.DataFrame({"sp" : data_df["sp"].value_counts().index,
                        "nb_img" : data_df["sp"].value_counts().values})

# Bar plot
st.bar_chart(
    data=dist_sp, 
    x="sp",           # Noms des espèces
    y="nb_img",       # Valeurs
    x_label="Espèces",
    y_label="Nombre d'images",
    horizontal=True,
    sort="-nb_img",
    color="#2ecc71"
)

# Violin plot de la distribution du nombre d'images
quartiles = dist_sp["nb_img"].quantile([0.25, 0.5, 0.75]).values
iqr = quartiles[2] - quartiles[0]

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 8)) # Plus haut pour équilibrer le bar_chart
    
sns.violinplot(
    y=dist_sp["nb_img"], # On le met en Y pour qu'il soit vertical et s'aligne mieux
    ax=ax, 
    color="#2ecc71", 
    inner="quartile",
    linewidth=2
)

# Ajouter les quartiles
for q, name_q in zip(quartiles, ["Q25%", "Q50%", "Q75%"]):
    ax.text(0.55, q, f'{name_q} : {int(q)}', va='center', color="#FFFFFF", fontweight='bold')

# Rendre le fond transparent pour le Dark Mode de Streamlit
fig.patch.set_alpha(0)
ax.set_facecolor((0, 0, 0, 0))
    
ax.set_title("Dispersion des classes", fontsize=15)
ax.set_ylabel("Nombre d'images")

st.pyplot(fig)


f'''
- Un jeu de données avec un important déséquilibre des classes 
  - {round(dist_sp["nb_img"].mean())} +/- {round(dist_sp["nb_img"].std())} images par classe
  - Tomate vs Common wheat : 18 146 vs 251 

  -> **Ré-équilibrage des classes par ré-échantillonnage et data augmentation**
'''

######################
### Niveau de flou ###
######################
'''
# Niveau de flou
'''
fig, ax = plt.subplots(figsize=(10, 8))

# Calculer le seuil des 5% les images les plus floues
seuil = np.quantile(data_df["laplacian_var"], 0.05)


# Tracer l'histogramme avec la courbe de densité
sns.histplot(
    data_df[data_df["laplacian_var"] < 1000]["laplacian_var"],
    bins=50,
    kde=True
)

# Ajouter la ligne du seuil de 5%
plt.axvline(seuil, color="red", linestyle="--", label="Seuil de flou (5%)")
new_ticks = sorted([t for t in ax.get_xticks() if abs(t - seuil) > 50] + [seuil])

fig.patch.set_alpha(0)
ax.set_facecolor((0, 0, 0, 0))   
ax.legend()
ax.set_title("Niveau de flou [0, 1000]", fontsize=15)
ax.set_xlabel("Variance du Laplacien")
ax.set_ylabel("Nombre d'images")
ax.set_xticks(sorted(new_ticks))
ax.set_xlim(0, 1000)


st.pyplot(fig)

f'''
- Un pic d'images floues : 
  - 5 % des images ont un niveau de flou inférieur à {round(seuil, 2)}

  -> **Seuil à 5 %**
'''

# Récupérer les index de l'image la plus/moins floue
idx_flou = data_df["laplacian_var"].idxmin()
idx_net = data_df["laplacian_var"].idxmax()
idx_seuil = data_df[data_df["laplacian_var"].between(seuil - 0.5, seuil + 0.5)].index[2]
images_extremes = data_df.loc[[idx_flou, idx_seuil, idx_net]]

# Fonction de récupération du chemin des images
def get_img_path(row):
    status = "Healthy" if pd.isna(row["disease"]) else "Diseased"
    return os.path.join(path_img, status, row["sp"], row["name"])

images_extremes["full_path"] = images_extremes.apply(get_img_path, axis=1)

# Afficher dans Streamlit
col_flou, col_seuil, col_net = st.columns(3)

with col_flou:
    st.error(f"Image la plus floue (Var: {images_extremes.loc[idx_flou, 'laplacian_var']:.2f})")
    # Remplace 'path' par le nom de ta colonne contenant le chemin du fichier
    st.image(images_extremes.loc[idx_flou, "full_path"], use_container_width=True)

with col_seuil:
    st.warning(f"Image proche du seuil (Var: {images_extremes.loc[idx_seuil, 'laplacian_var']:.2f})")
    # Remplace 'path' par le nom de ta colonne contenant le chemin du fichier
    st.image(images_extremes.loc[idx_seuil, "full_path"], use_container_width=True)

with col_net:
    st.success(f"Image la plus nette (Var: {images_extremes.loc[idx_net, 'laplacian_var']:.2f})")
    st.image(images_extremes.loc[idx_net, "full_path"], use_container_width=True)


st.markdown(
    '''
    # Pré-traitement du jeu de données
    - Elimination des doublons (hashing)
    - Redimensionnement des images (256 x 256 pixels)
    - Retrait des images les plus floues (seuil à 5 %) 
      - 59 820 -> 56 829 images
      - Classe<sub>min</sub> = 242 images ("Common Wheat")
    - Ré-équilibrage du jeu de données (renversement, rotation, contraste) :
      - Sous-échantillonnage des Classes > Classe<sub>min</sub> x 10
      - Data augmentation des Classes < Classe<sub>min</sub> x 10''',
unsafe_allow_html=True)

##################################
### Compter le nombre d'images ###
##################################
def count_img(path_img):
    classes = []
    counts = []
    
    for x_sp in sorted(os.listdir(path_img)):
        path_sp = os.path.join(path_img, x_sp)
        if os.path.isdir(path_sp):
            classes.append(x_sp)
            counts.append(len(os.listdir(path_sp)))
            
    df = pd.DataFrame({
        "classe" : classes,
        "nb_images" : counts
    })

    return df

# Compter le nb d'images dans train/val
df_img_train = count_img("02_data/data_spDetection_ready/train")
df_img_val = count_img("02_data/data_spDetection_ready/val")

# Préparer le format du nb d'images
t_str = f"{df_img_train['nb_images'].sum():,}".replace(",", " ")
v_str = f"{df_img_val['nb_images'].sum():,}".replace(",", " ")


df_img = pd.concat([df_img_train, df_img_val])
total_images = df_img["nb_images"].sum()

######################
### Chiffres clefs ###
######################
labels = ["Nombre d'images avant pré-traitement", "Nombre de classes", "Ecart interquartile (IQR)", "Nombre d'images après pré-traitement"]
values = [len(data_df), len(data_df["sp"].unique()), round(iqr), f"{total_images:,}".replace(",", " ")]
helps = [
    "Total des images présentes dans le dataset",
    "Nombre de catégories de plantes distinctes",
    "Dispersion du nombre d'images par espèce (Q3 - Q1)",
    f"Détail : \n- Train : {t_str} \n- Test : {v_str}"
]

with st.sidebar:
    st.write("### 📊 Chiffres clés")
    
    for l, v, h in zip(labels, values, helps):
        # Ajouter un formatage f-string pour l'espace des milliers sur le total
        if isinstance(v, int) and v > 1000:
            display_val = f"{v:,}".replace(",", " ")
        else:
            display_val = v
            
        st.metric(label=l, value=display_val, help = h)
    