import numpy as np
from pydantic import PrivateAttr
from sklearn.ensemble import RandomForestClassifier as RFC

from autoop.core.ml.model import Model


class Random_forest(Model):

    _rfc: RFC = PrivateAttr(deffault=None)

    def __init__(self) -> None:
        super().__init__()
        self._rfc = RFC()
        self._type = "classification"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        self._rfc.fit(observations, ground_truths)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self._rfc.predict(observations)
