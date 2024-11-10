import streamlit as st
import pandas as pd
import numpy as np
import os

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import get_metric
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types

from autoop.core.ml.model.regression import (
    Lasso, MultipleLinearRegression, GradientBoostingR
)
from autoop.core.ml.model.classification import (
    KNN, Neural_network_classifier, Random_forest
)

selected_model_name = None

st.set_page_config(page_title="Modelling", page_icon="📈")

st.write("# ⚙ Modelling")

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

regression_models = {
    "Lasso": Lasso,
    "Multiple Linear Regression": MultipleLinearRegression,
    "Gradient Boosting Regression": GradientBoostingR
}
classification_models = {
    "K Nearest Neighbors": KNN,
    "Neural Network Classifier": Neural_network_classifier,
    "Random Forest Classification": Random_forest
}

regression_metrics = {
    "Mean Squared Error": "mean_squared_error",
    "Mean Absolute Error": "mean_absolute_error",
    "R-Squared": "r-squared"
}
classification_metrics = {
    "Accuracy": "accuracy",
    "Macro Precision": "macro_precision",
    "Macro Recall": "macro_recall"
}

if "regression_selected" not in st.session_state:
    st.session_state.regression_selected = False
if "classification_selected" not in st.session_state:
    st.session_state.classification_selected = False

if datasets:

    dataset_list = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox(
        "Please select your dataset", dataset_list
    )

    if selected_dataset_name:
        selected_dataset = next(
            ds for ds in datasets if ds.name == selected_dataset_name
        )

        features_data = Dataset.from_artifact(selected_dataset)
        features = detect_feature_types(features_data)
        data = features_data.to_dataframe()

        continuous_columns = [
            f.name for f in features if f.type == "numerical"
        ]
        categorical_columns = [
            f.name for f in features if f.type == "categorical"
        ]

        st.write("### Select model type:")
        col1, col2 = st.columns(2)
        if col1.button("Regression Models"):
            st.session_state.regression_selected = True
            st.session_state.classification_selected = False
        if col2.button("Classification Models"):
            st.session_state.classification_selected = True
            st.session_state.regression_selected = False

        selected_metrics = []
        if st.session_state.regression_selected:
            selected_model_name = st.selectbox(
                "Please select a regression model", list(
                    regression_models.keys()
                    )
                )
            st.write(f"Selected Model: {selected_model_name}")
            st.write("### Select Metrics for Regression")
            for metric in regression_metrics:
                if st.checkbox(metric):
                    selected_metrics.append(metric)
        elif st.session_state.classification_selected:
            selected_model_name = st.selectbox(
                "Please select a classification model", list(
                    classification_models.keys()))
            st.write(f"Selected Model: {selected_model_name}")
            st.write("### Select Metrics for Classification")
            for metric in classification_metrics:
                if st.checkbox(metric):
                    selected_metrics.append(metric)

            if selected_model_name == "K Nearest Neighbors":
                k_value = st.number_input(
                    "Enter the number of neighbors (k) for KNN:",
                    min_value=1, value=3)

        st.write("### Select Features for Modelling")
        input_feature_names = st.multiselect("Select input features", [
            f.name for f in features])

        input_features = [f for f in features if f.name in input_feature_names]

        if st.session_state.regression_selected:
            target_columns = [
                col for col in continuous_columns
                if col not in input_feature_names
            ]
        else:
            target_columns = [
                col for col in categorical_columns
                if col not in input_feature_names
            ]

        target_feature_name = st.selectbox("Select target feature",
                                           target_columns)
        target_feature = next((f for f in features
                               if f.name == target_feature_name), None)

        st.write("### Dataset Split Selection")
        split_ratio = st.slider(
            "Select the training data split ratio (0.0 - 1.0)",
            min_value=0.1, max_value=0.9, value=0.8)
        st.write(f"Selected Split Ratio: {split_ratio}")

        st.write("# 📋 Pipeline Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("#### Dataset")
            st.write(f"- **Name**: {selected_dataset_name}")
            st.write("#### Model")
            model_text = 'Regression' \
                if st.session_state.regression_selected \
                else 'Classification'
            st.write(f"- **Type**: {model_text}")
            st.write(f"- **Model**: {selected_model_name}")
        with col2:
            st.write("#### Features")
            st.write(f"- **Input Features**: {', '.join(input_feature_names)}")
            st.write(f"- **Target Feature**: {target_feature_name}")
        with col3:
            st.write("#### Metrics")
            st.write(
                f"- **Selected Metrics**: {', '.join(selected_metrics)
                                           if selected_metrics
                                           else 'None'}"
                                           )
            st.write("#### Split Ratio")
            st.write(f"- **Training/Test Split**: {split_ratio}")

        if st.button("Run Pipeline"):
            if selected_model_name == "K Nearest Neighbors":
                selected_model = (
                    classification_models[selected_model_name](k=k_value)
                )
            else:
                selected_model = (
                    regression_models[selected_model_name]()
                    if st.session_state.regression_selected
                    else classification_models[selected_model_name]()
                )

            metrics_instances = [
                get_metric(
                    regression_metrics[m] if m in regression_metrics
                    else classification_metrics[m]
                )
                for m in selected_metrics
            ]

            pipeline = Pipeline(
                metrics=metrics_instances,
                dataset=features_data,
                model=selected_model,
                input_features=input_features,
                target_feature=target_feature,
                split=split_ratio
            )

            st.session_state["pipeline"] = pipeline

            results = pipeline.execute()

            metrics_data = {
                metric.__class__.__name__: [result] for metric,
                result in results["train_metrics"]
            }

            metrics_df = pd.DataFrame(metrics_data)

            predictions_flat = np.ravel(results["test_predictions"])
            actual_values_flat = np.ravel(pipeline._test_y)

            predictions_df = pd.DataFrame({
                "Predictions": predictions_flat,
                "Actual": actual_values_flat
            })

            st.write("### Metrics Results")
            st.dataframe(metrics_df)

            st.write("### Predictions Results")
            st.dataframe(predictions_df)

        with st.form("Save Pipeline"):
            st.write("### Save Pipeline")
            pipeline_name = st.text_input(
                "Enter Pipeline Name", value=selected_dataset_name
            )
            pipeline_version = st.text_input(
                "Enter Pipeline Version", value="1.0"
            )
            save_button = st.form_submit_button("Confirm")
            print(st.session_state)

        if save_button:
            if "pipeline" in st.session_state:
                st.write("Save button pressed and pipeline exists")
                st.session_state["pipeline_name"] = pipeline_name
                st.session_state["pipeline_version"] = pipeline_version

                pipeline_dir = "assets/pipeline"
                if not os.path.exists(pipeline_dir):
                    os.makedirs(pipeline_dir, exist_ok=True)

                asset_path = os.path.join(
                    pipeline_dir, f"{pipeline_name}.pkl")

                pipeline = st.session_state["pipeline"]
                artifact = pipeline.save(
                    name=pipeline_name, version=pipeline_version,
                    save_path=asset_path)

                automl.registry.register(artifact)
                st.success(
                    f"Pipeline '{pipeline_name}' saved successfully!"
                )
            else:
                st.error("Pipeline not found. Please run the pipeline first.")
