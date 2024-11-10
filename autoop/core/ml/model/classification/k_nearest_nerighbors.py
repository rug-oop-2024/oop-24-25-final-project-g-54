import numpy as np
from pydantic import PrivateAttr
from sklearn.neighbors import KNeighborsClassifier as knn

from autoop.core.ml.model import Model


class KNN(Model):
    """
    A wrapper that implements the k-nearest
    neighbors(KNN) classification algorithm.

    This model finds the k closest observations in the training data to a given
    input, based on distance, and predicts the outcome as the most common label
    among those neighbors.

    """
    _knn: knn = PrivateAttr(default=None)

    def __init__(self, k: int = 3) -> None:
        """
        Initializes the KNNWrapper model
        with the specified number of neighbors.

        Arg:
            k:  The number of nearest neighbors to consider when making
            predictions (default is 3).
        """
        super().__init__()
        self._knn = knn(n_neighbors=k)
        self._type = "classification"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """
        Trains the KNN model on the provided dataset.

        Arg:
            observations (ndarray): The input data, a matrix where each
            row (n) is an observation. Each coloumn (p) is a feature.
            ground_truths (ndarray): An array containing the true
            labels for each observation.
        """
        self._knn.fit(observations, ground_truths)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
       Predicts the label for a single observation using k nearest neighbors.

        Arg:
            observations (ndarray): The input data, a matrix where each
            row (n) is an observation. Each coloumn (p) is a feature.

        Returns:
            A list of predicted labels for each observation the input data.

        """
        return self._knn.predict(observations)
