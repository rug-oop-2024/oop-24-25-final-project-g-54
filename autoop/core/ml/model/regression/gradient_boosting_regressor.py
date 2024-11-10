import numpy as np
from pydantic import PrivateAttr
from sklearn.ensemble import GradientBoostingRegressor as GBR

from autoop.core.ml.model import Model


class GradientBoostingR(Model):
    """
    A wrapper that implements the
    Gradient Boosting Regressor from Scikit-learn..

    This model uses an ensemble of weak learners (decision trees)
    to predict a continuous value trained using gradient boosting
    which minimizes prediction errors through iterative updates.
    """

    _gbr: GBR = PrivateAttr(default=None)

    def __init__(self) -> None:
        """
        Initializes the gradient boosting regressor.
        """
        super().__init__()
        self._gbr = GBR()
        self._type = "regression"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """
        Trains the gradient boosting model on the provided dataset.

        Arg:
            observations (ndarray): The input data, a matrix where each
            row (n) is an observation. Each column (p) is a feature.
            ground_truths (ndarray): An array containing the true values for
            each observation.
        """
        self._gbr.fit(observations, ground_truths)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        Predicts the value for a given observation
        using the trained gradient boosting model.

        Arg:
            observation (ndarray): A matrix where each row (n)
            is a single observation and each column (p) is a feature.

        Returns:
            A list of predicted values for each observation in the input data.
        """
        return self._gbr.predict(observation)
