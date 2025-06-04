import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def run_pca(df):
    st.title("🧬 PCA Feature Selection & Visualization")

    # Step 1: Standardize the Data
    st.subheader("1️⃣ Standardize Numerical Data")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not num_cols:
        st.warning("⚠️ No numerical columns found for PCA.")
        return

    st.write("📌 Columns used for PCA:", num_cols)

    X = df[num_cols].dropna()  # Handle missing values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.success("✅ Data has been standardized.")

    # Step 2: Covariance Matrix (Informational)
    st.subheader("2️⃣ Covariance Matrix (Optional View)")
    if st.checkbox("Show covariance matrix"):
        cov_matrix = np.cov(X_scaled, rowvar=False)
        st.write(pd.DataFrame(cov_matrix, index=num_cols, columns=num_cols))

    # Step 3: Compute Principal Components
    st.subheader("3️⃣ Principal Components")
    pca = PCA()
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    st.write("🔍 Explained Variance by each component:", explained_variance)

    # Scree Plot
    st.markdown("**📊 Scree Plot (Explained Variance Ratio)**")
    fig1 = px.line(
        x=np.arange(1, len(explained_variance) + 1),
        y=np.cumsum(explained_variance),
        labels={'x': 'Number of Components', 'y': 'Cumulative Explained Variance'},
        markers=True,
        title="Cumulative Explained Variance"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Step 4: Pick Top Directions
    st.subheader("4️⃣ Top Principal Components")
    n_components = st.slider("Select number of components to retain", 2, min(len(num_cols), 10), 2)
    pca_top = PCA(n_components=n_components)
    X_pca_top = pca_top.fit_transform(X_scaled)

    st.success(f"Retained {n_components} principal components.")

    # Display transformed data
    pca_df = pd.DataFrame(X_pca_top, columns=[f'PC{i+1}' for i in range(n_components)])
    st.dataframe(pca_df.head(), use_container_width=True)

    # Biplot if 2 components
    if n_components == 2:
        st.markdown("📌 PCA Biplot")
        fig2, ax = plt.subplots()
        scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA Biplot")
        st.pyplot(fig2)

    # Feature importance from PCA
    st.subheader("🧠 Feature Importance from PCA")
    loading_scores = pd.Series(
        np.abs(pca.components_[0]),
        index=num_cols
    ).sort_values(ascending=False)
    top_features = loading_scores.head(10).index.tolist()
    st.session_state["top_pca_features"] = top_features
    st.write("Top Influential Features (PC1):")
    st.write(loading_scores.head(10))

     # 🔁 Button to Clear PCA Selection
    if st.button("🔁 Clear PCA Feature Selection"):
        if "top_pca_features" in st.session_state:
            del st.session_state["top_pca_features"]
            st.success("PCA-selected features cleared.")

