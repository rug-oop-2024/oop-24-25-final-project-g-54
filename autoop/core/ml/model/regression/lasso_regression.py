import numpy as np
from pydantic import PrivateAttr
from sklearn.linear_model import Lasso as ls

from autoop.core.ml.model import Model


class Lasso(Model):

    _ls: ls = PrivateAttr(default=None)

    def __init__(self) -> None:

        super().__init__()
        self._ls = ls()
        self._type = "regression"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        self._ls.fit(observations, ground_truths)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        return self._ls.predict(observation)
