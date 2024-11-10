from typing import List
import pandas as pd
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """

    df = dataset.read()
    features = []

    for column in df.columns:
        column_data = df[column]

        if isinstance(column_data.dtype, pd.CategoricalDtype):
            feature_type = "categorical"
        elif pd.api.types.is_numeric_dtype(column_data):
            feature_type = "numerical"
        else:
            feature_type = "categorical"

        features.append(Feature(name=column, type=feature_type))

    return features

    # raise NotImplementedError("This should be implemented by you.")
