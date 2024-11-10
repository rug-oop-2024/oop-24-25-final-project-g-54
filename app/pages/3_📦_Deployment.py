import streamlit as st
import pandas as pd
import numpy as np
import time
from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline

st.set_page_config(page_title="Pipeline Deployment", page_icon="📦")

st.write("# 📦 Pipeline Deployment")
st.write("""
    Manage, view, and deploy saved machine learning pipelines.
    Select a pipeline to explore its configuration, make predictions,
    or delete it.
""")

automl = AutoMLSystem.get_instance()
saved_pipelines = automl.registry.list(type="pipeline")

if saved_pipelines:
    pipeline_names = [pipeline.name for pipeline in saved_pipelines]
    st.subheader("Available Pipelines")
    selected_pipeline_name = st.selectbox(
        "Select a pipeline to view or delete:", pipeline_names
    )

    selected_pipeline = next(
        p for p in saved_pipelines if p.name == selected_pipeline_name
    )

    if st.button("Load Pipeline"):
        st.session_state["loaded_pipeline"] = Pipeline.load(
            selected_pipeline.asset_path
        )

    if "loaded_pipeline" in st.session_state:
        loaded_pipeline = st.session_state["loaded_pipeline"]

        dataset_name = loaded_pipeline._dataset.name if hasattr(
            loaded_pipeline._dataset, "name") else "N/A"
        model_name = type(loaded_pipeline._model).__name__
        model_type = (
            "Regression" if "Regression" in model_name else "Classification"
        )
        input_features = [f.name for f in loaded_pipeline._input_features]
        target_feature = loaded_pipeline._target_feature.name if hasattr(
            loaded_pipeline._target_feature, "name") else "N/A"
        metrics = (
            [type(m).__name__ for m in loaded_pipeline._metrics]
            if loaded_pipeline._metrics else ["None"]
            )
        split_ratio = loaded_pipeline._split

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write("#### Dataset")
            st.write(f"- **Name**: {dataset_name}")
        with col2:
            st.write("#### Model")
            st.write(f"- **Type**: {model_type}")
            st.write(f"- **Model**: {model_name}")
        with col3:
            st.write("#### Features")
            st.write(f"- **Input Features**: {', '.join(input_features)}"
                     if input_features else "None")
            st.write(f"- **Target Feature**: {target_feature}")
        with col4:
            st.write("#### Metrics")
            st.write(f"- **Selected Metrics**: {', '.join(metrics)}")

        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Split Ratio")
            st.write(f"- **Training/Test Split**: {split_ratio}")

        st.write("### Perform Predictions")
        uploaded_file = st.file_uploader(
            "Upload a CSV file with input features", type="csv")

        if hasattr(loaded_pipeline._model, '_param') and hasattr(
                loaded_pipeline._model, 'set_parameters'):
            loaded_pipeline._model.set_parameters(
                loaded_pipeline._model._param)

        if uploaded_file is not None:
            input_data = pd.read_csv(uploaded_file)
            input_feature_names = [
                f.name for f in loaded_pipeline._input_features
            ]

            if not all(
                    feature in input_data.columns
                    for feature in input_feature_names
                 ):
                st.error(
                    f"Uploaded file must contain the following columns: "
                    f"{', '.join(input_feature_names)}"
                    )
            else:
                input_data = input_data[input_feature_names]
                results = loaded_pipeline.execute()

                predictions_flat = np.ravel(results["test_predictions"])
                actual_values_flat = np.ravel(loaded_pipeline._test_y)

                predictions_df = pd.DataFrame({
                    "Predictions": predictions_flat,
                    "Actual": actual_values_flat
                })

                st.write("### Predictions Results")
                st.dataframe(predictions_df)

                csv_data = predictions_df.to_csv(index=False)
                st.download_button("Download Predictions as CSV",
                                   data=csv_data, file_name="predictions.csv",
                                   mime="text/csv")

    if st.button("Delete Pipeline"):
        automl.registry.delete(selected_pipeline.id)
        st.success(
            f"Pipeline '{selected_pipeline_name}' deleted successfully."
        )
        time.sleep(1)
        st.rerun()

else:
    st.warning("No pipelines are available. Please save a pipeline first.")
