import streamlit as st
import pandas as pd

def upload_data():
    st.title("📥 Upload your CSV file")
    uploaded_file = st.file_uploader("Choose a CSV, JSON, or Excel file", type=["csv", "json", "xls", "xlsx"])

    if uploaded_file is not None:
        try:
            file_type = uploaded_file.name.split(".")[-1].lower()

            if file_type == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_type == "json":
                df = pd.read_json(uploaded_file)
            elif file_type in ["xls", "xlsx"]:
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Unsupported file type")

            file_details = {
                "Filename": uploaded_file.name,
                "Size (KB)": round(len(uploaded_file.getvalue()) / 1024, 2),
                "Rows": df.shape[0],
                "Columns": df.shape[1],
                "File Type": file_type
            }

            st.success("✅ File successfully uploaded!")
            st.json(file_details)

            st.subheader("📊 Preview of Data")
            st.dataframe(df.head(10), use_container_width=True)

            return df  # important: return df for downstream use

        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            return None
    else:
        st.info("📂 Upload a file to begin.")
        return None
