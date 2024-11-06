import numpy as np
from pydantic import PrivateAttr
from sklearn.linear_model import LogisticRegression as logr

from autoop.core.ml.model import Model


class Logistic_Regression(Model):

    _logr: logr = PrivateAttr(default=None)

    def __init__(self) -> None:
        super().__init__()
        self._logr = logr()
        self._type = "classification"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        self._logr.fit(observations, ground_truths)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self._logr.predict(observations)
