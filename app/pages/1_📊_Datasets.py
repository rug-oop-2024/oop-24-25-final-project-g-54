import streamlit as st
import pandas as pd
import time
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

st.title("Dataset Management")

st.subheader("Upload a New Dataset")
uploaded_file = st.file_uploader("Choose a file to upload", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    file_name = uploaded_file.name.split(".")[0]
    asset_path = f"dataset/{file_name}"

    new_dataset = Dataset.from_dataframe(
        data=data, name=file_name, asset_path=asset_path,
        version="1.0.0"
    )

    if st.button("Upload/Save Dataset"):
        automl.registry.register(new_dataset)
        st.success(f"Dataset '{file_name}' uploaded successfully.")
        time.sleep(1)
        st.rerun()


if datasets:
    st.subheader("Uploaded Datasets")
    dataset_list = [dataset.name for dataset in datasets]
    selected = st.selectbox("Select a dataset to view or delete", dataset_list)

    selected_dataset = next(file for file in datasets if file.name == selected)

    if st.button("View Dataset"):
        asset_path = selected_dataset.asset_path
        data = pd.read_csv(f"assets/objects/{asset_path}")
        st.write(f"Displaying contents of {selected}:")
        st.dataframe(data)

    if st.button("Delete Dataset"):
        automl.registry.delete(selected_dataset.id)
        st.success(f"Dataset '{selected}' deleted successfully.")
        time.sleep(1)
        st.rerun()
