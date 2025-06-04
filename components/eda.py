import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import os
import uuid
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import sweetviz as sv
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_datetime64_any_dtype

def run_eda(df):
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.markdown("### 🎬 Watch: EDA Process Explained")
    st.video("https://www.youtube.com/embed/PPEHpg2RixQ?si=XAK2GdKccaNBDmMJ")

    # Step 1: Problem Understanding
    st.subheader("1️⃣ Problem Statement and Dataset Overview")
    st.info("This step is user-driven. Understand what your data is about.")
    st.write("Shape of Dataset:", df.shape)
    st.write("Column Names:", df.columns.tolist())

    # Step 2: Data Preview
    st.subheader("2️⃣ Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Column types
    st.subheader("📌 Column Data Types")
    st.write(df.dtypes)

    # 🔁 Batch Conversion
    st.markdown("### 🔄 Batch Convert Column Data Types")
    cols_selected = st.multiselect("Select Columns", df.columns)
    conversion_type = st.selectbox("Convert Selected Columns To", ["int", "float", "object", "datetime"])

    if st.button("🔁 Batch Convert"):
        success_cols, failed_cols = [], []
        for col in cols_selected:
            try:
                if conversion_type == "datetime":
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = df[col].astype(conversion_type)
                success_cols.append(col)
            except Exception as e:
                failed_cols.append((col, str(e)))

        if success_cols:
            st.success(f"✅ Successfully converted: {', '.join(success_cols)}")
        if failed_cols:
            for col, err in failed_cols:
                st.error(f"❌ Failed to convert `{col}`: {err}")
        st.write(df.dtypes)

    # 🧠 Suggested Data Types
    st.markdown("### 🧠 Suggested Data Type Fixes")
    suggestions = []
    for col in df.columns:
        if is_string_dtype(df[col]):
            try:
                pd.to_datetime(df[col])
                suggestions.append((col, 'datetime'))
            except:
                try:
                    df[col].astype(float)
                    suggestions.append((col, 'float'))
                except:
                    pass
        elif is_numeric_dtype(df[col]) and df[col].dropna().apply(lambda x: float(x).is_integer()).all():
            suggestions.append((col, 'int'))

    if suggestions:
        st.info("🔍 Detected potential type improvements:")
        for col, suggested_type in suggestions:
            st.write(f"🔄 Column `{col}` might be better as `{suggested_type}`")

    # Step 3: Missing Values
    st.subheader("3️⃣ Missing Values")
    nulls = df.isnull().sum()
    st.write(nulls[nulls > 0])
    st.bar_chart(nulls[nulls > 0])

    # Step 4: Summary Statistics
    st.subheader("4️⃣ Data Summary Statistics")
    st.write(df.describe(include='all'))

    # Step 5: Data Transformation
    st.subheader("5️⃣ Data Transformation (Suggestions)")
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.write("📌 Categorical Columns:", cat_cols)
    st.write("📌 Numerical Columns:", num_cols)

    # Step 6: Visualizations
    st.subheader("6️⃣ Visualizations")

    if num_cols:
        col = st.selectbox("Select a column for histogram", num_cols)
        fig1 = px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}")
        st.plotly_chart(fig1, use_container_width=True)

        if len(num_cols) >= 2:
            st.markdown("**🔗 Correlation Heatmap**")
            corr = df[num_cols].corr()
            fig2, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig2)

        if len(num_cols) <= 5:
            st.markdown("**📊 Pairplot (for fewer columns)**")
            fig3 = sns.pairplot(df[num_cols])
            st.pyplot(fig3)

    # Step 7: Outlier Detection
    st.subheader("7️⃣ Outlier Detection")
    outlier_col = st.selectbox("Select a column for outlier detection", num_cols)
    fig4, ax = plt.subplots()
    sns.boxplot(x=df[outlier_col], ax=ax)
    st.pyplot(fig4)

    # Step 8: Insights
    st.subheader("8️⃣ Insights & Observations")
    st.text_area("🧠 Note down key findings from your analysis here:")

    # Step 9: EDA Tools
    st.subheader("9️⃣ Automated EDA Tools")
    eda_tool = st.selectbox("Choose EDA Tool", ["None", "Pandas Profiling", "Sweetviz"])

    if eda_tool == "Pandas Profiling":
        profile = ProfileReport(df, explorative=True)
        st_profile_report(profile)

    elif eda_tool == "Sweetviz":
        st.warning("Sweetviz report will download after generation.")
        report = sv.analyze(df)
        unique_id = str(uuid.uuid4())
        report_path = f"sweetviz_report_{unique_id}.html"
        report.show_html(filepath=report_path, open_browser=False)
        with open(report_path, 'rb') as f:
            st.download_button("📥 Download Sweetviz Report", f, file_name="sweetviz_report.html")
        os.remove(report_path)
