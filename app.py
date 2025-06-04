import streamlit as st
from components.upload import upload_data
from components.eda import run_eda
from components.pca_visualizer import run_pca
from components.model_selector import run_model_selector
from components.prediction import run_prediction


st.set_page_config(page_title="ML Flow", layout="wide")

# Try to load custom CSS if available
try:
    with open("assests/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom styles not found. Skipping CSS.")

# Page config

#st.sidebar.image("assests/logo.png", width=150)

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Choose a Section", ["EDA", "PCA", "Model Selector", "Prediction"])

# Title and instructions
st.title("🧠 ML Flow")
st.caption("Your ML journey begins here. Upload your data to explore insights, visualize features, and build models.")

# Upload and hold dataset
df = upload_data()

# Render pages conditionally
if df is not None:
    st.success("✅ Data uploaded successfully!")

    if page == "EDA":
        run_eda(df)

    elif page == "PCA":
        run_pca(df)

    elif page == "Model Selector":
        run_model_selector(df)

    elif page == "Prediction":
        run_prediction()
else:
    st.warning("⚠️ Please upload a dataset to proceed.")
