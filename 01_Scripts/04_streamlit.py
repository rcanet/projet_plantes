import streamlit as st


pg = st.navigation([
    st.Page("streamlit/intro.py", title="Introduction", icon="✒️"),
    st.Page("streamlit/exploration.py", title="Exploration des données", icon="🔍"),
    st.Page("streamlit/modelisation.py", title="Modélisation", icon="🔬"),
    st.Page("streamlit/demo.py", title="Démonstration", icon="🍀"),
    st.Page("streamlit/conclusion.py", title="Conclusion", icon="🔥")
])
pg.run()

# streamlit run 01_Scripts/04_streamlit.py