import numpy as np
from pydantic import PrivateAttr
from sklearn.ensemble import GradientBoostingRegressor as GBR

from autoop.core.ml.model import Model


class GradientBoosting(Model):

    _gbr: GBR = PrivateAttr(default=None)

    def __init__(self) -> None:
        super().__init__()
        self._gbr = GBR()
        self._type = "regression"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        self._gbr.fit(observations, ground_truths)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        return self._gbr.predict(observation)
