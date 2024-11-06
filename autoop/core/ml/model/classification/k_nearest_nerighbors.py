import numpy as np
from pydantic import PrivateAttr
from sklearn.neighbors import KNeighborsClassifier as knn

from autoop.core.ml.model import Model


class KNN(Model):

    _knn: knn = PrivateAttr(default=None)

    def __init__(self, k: int = 3) -> None:
        super().__init__()
        self._knn = knn()
        self._type = "classification"
        self.k = k

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        self._knn.fit(observations, ground_truths)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self._knn.predict(observations)
