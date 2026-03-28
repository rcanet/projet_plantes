import streamlit as st
import os

# Page 1 : Introduction & Context 
""
st.markdown('''
# Contexte
Ce projet vise à développer un modèle de deep learning capable d’identifier automatiquement l’espèce végétale présente dans une image.  
Application multiples pour le grand public (randonneurs, ...), l'agriculture et la recherche. 
            
## Pourquoi ? 
''')

################
### Contexte ###
################

path_img = "01_Scripts/streamlit/img_intro"
col_1, col_2 = st.columns(2)

with col_1:
    st.image(os.path.join(path_img, "pexels_hiking.jpg"), use_container_width=True)
    st.markdown(''' Randonnées, grand public ''')

with col_2:
    st.image(os.path.join(path_img, "rita_wheat.jpg"), use_container_width=True)
    st.markdown(''' Agriculture, reconnaissance d'adventices ''')


################
### Sommaire ###
################

st.markdown('''
### Sommaire
1. **Exploration des données** : Exploration descriptive du jeu de données brut.
2. **Modélisation** : Architecture des modèles et résultats.
3. **Démonstration** 
4. **Conclusion** : Réussite du projet.
''')
