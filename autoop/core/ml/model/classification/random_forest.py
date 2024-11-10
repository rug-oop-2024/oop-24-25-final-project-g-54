import numpy as np
from pydantic import PrivateAttr
from sklearn.ensemble import RandomForestClassifier as RFC

from autoop.core.ml.model import Model


class Random_forest(Model):
    """
    A wrapper that implements the random forest classification algorithm.

    This model uses an ensemble of decision trees to classify observations
    based on patterns learned from the training data.
    """

    _rfc: RFC = PrivateAttr(default=None)

    def __init__(self) -> None:
        """
        Initializes the random forest classifier.
        """
        super().__init__()
        self._rfc = RFC()
        self._type = "classification"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """
        Trains the random forest model on the provided dataset.

        Arg:
            observations (ndarray): The input data, a matrix where each
            row (n) is an observation. Each column (p) is a feature.
            ground_truths (ndarray): An array containing the true
            labels for each observation.
        """
        self._rfc.fit(observations, ground_truths)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the label for each observation using
        the trained random forest.

        Arg:
            observations (ndarray): The input data, a matrix where each
            row (n) is an observation. Each column (p) is a feature.

        Returns:
            A list of predicted labels for each observation in the input data.
        """
        return self._rfc.predict(observations)
