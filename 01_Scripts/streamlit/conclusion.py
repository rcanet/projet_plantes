import streamlit as st

def show_conclusion():
    st.title("🏁 Conclusion & Perspectives")

    # --- SECTION 1 : BILAN TECHNIQUE ---
    st.subheader("📊 Bilan du Projet")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Classes de plantes", value="26")
    with col2:
        st.metric(label="Précision (ResNet50)", value="~96%", delta="Top Performance")
    with col3:
        st.metric(label="Modèle LeNet", value="Léger", delta="- d'accuracy", delta_color="inverse")

    st.info("""
    **Ce qu'il faut retenir :**
    * Le **Transfer Learning** (ResNet50) est plus efficace qu'un modèle "simple".
    * -> Le modèle ResNet50 = le modèle à mettre en production.
    """)

    st.divider()

    # --- SECTION 2 : LIMITES ET AMÉLIORATIONS ---
    st.subheader("📋 Perspectives d'évolution")

    st.write("""
        * **Augmentation de la taille de l'échantillon :** Utiliser d'autres images en entrée => Améliorer la classification des plantes qui se ressemblent ;
        * **Segmentation :** Eviter que le modèle ne reconnaisse le fond des images plutôt que les plantes ;
        * **Maladie :** Ajouter la détection de maladie ;
        """)

show_conclusion()