import numpy as np
from pydantic import PrivateAttr
from sklearn.linear_model import LassoCV as ls

from autoop.core.ml.model import Model


class Lasso(Model):
    """
    A wrapper that implements the Lasso
    regreesion model from Scikit-learn package.

    This model fits input and forms predictions based on learned parameters.
    """

    _ls: ls = PrivateAttr(default=None)

    def __init__(self) -> None:
        """
        Initializes the Lasso regression model.
        """
        super().__init__()
        self._ls = ls()
        self._type = "regression"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """
        Trains the Lasso model on the provided dataset.

        Args:
            observations (ndarray): The input data, where each row (n)
            represents an observation and each column (p) represents a feature.
            ground_truths (ndarray): An array of true values corresponding to
            each observation in the input data.
        """
        self._ls.fit(observations, ground_truths)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Predicts the value for a given
        observation using the trained Lasso model.

        Args:
            observation (ndarray): A matrix where each row (n) is a single
            observation, and each column (p) is a feature.

        Returns:
            A list of predicted values for each observation in the input data.
        """
        return self._ls.predict(observation)
