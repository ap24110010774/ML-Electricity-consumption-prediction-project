
import streamlit as st
import pandas as pd

def show_upload_page():

    st.title("📁 Upload Your Dataset")

    st.write(
        "Upload a CSV or Excel file to begin your forecasting project."
    )

    uploaded_file = st.file_uploader(
        "Choose your dataset",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success("Dataset uploaded successfully!")

            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

            st.subheader("Dataset Information")

            col1, col2 = st.columns(2)

            col1.metric("Total Rows", df.shape[0])
            col2.metric("Total Columns", df.shape[1])

            st.subheader("Column Names")
            st.write(list(df.columns))

            st.subheader("Missing Values")
            st.dataframe(
                df.isnull().sum().reset_index().rename(
                    columns={
                        "index": "Column",
                        0: "Missing Values"
                    }
                ),
                use_container_width=True
            )

            # Keep uploaded dataset available during this session
            st.session_state["uploaded_dataset"] = df
            st.session_state["uploaded_filename"] = uploaded_file.name

        except Exception as e:
            st.error(f"Unable to read the dataset: {e}")

