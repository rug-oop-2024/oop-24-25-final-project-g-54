
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.classification.k_nearest_nerighbors import KNN


REGRESSION_MODELS = [
] # add your models as str here

CLASSIFICATION_MODELS = [
    "KNN",
    "Logistic_Regression",
    "GradientBoosting"
    
] # add your models as str here

def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    raise NotImplementedError("To be implemented.")