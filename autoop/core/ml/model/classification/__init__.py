"""
This package contains the classification models.
"""

from autoop.core.ml.model.classification.k_nearest_nerighbors import KNN
from autoop.core.ml.model.classification.mlpclassifier import (
    Neural_network_classifier
)
from autoop.core.ml.model.classification.random_forest import (
    Random_forest
)

__all__ = [
    "KNN",
    "Neural_network_classifier",
    "Random_forest"
]
