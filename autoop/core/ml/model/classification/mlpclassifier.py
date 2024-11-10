import numpy as np
from pydantic import PrivateAttr
from sklearn.neural_network import MLPClassifier as mlp

from autoop.core.ml.model import Model


class Neural_network_classifier(Model):
    """
    A wrapper that implements a neural
    network classifier using multi-layer perceptron (MLP).

    This model uses a multi-layer perceptron (MLP) to classify observations
    into specified categories trained through backpropagation on labeled data.
    """

    _mlp: mlp = PrivateAttr(default=None)

    def __init__(self) -> None:
        """
        Initializes the neural network classifier.
        """

        super().__init__()
        self._logr = mlp()
        self._type = "classification"

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """
        Trains the neural network classifier on the provided dataset.

        Arg:
            observations (ndarray): The input data, a matrix where each
            row (n) is an observation and each coloumn (p) is a feature.
            ground_truths (ndarray): An array containing the true
            labels for each observation.
        """

        self._logr.fit(observations, ground_truths)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for the given observations using the trained
        neural network model.

        Arg:
            observations (ndarray): The input data, a matrix where each
            row (n) is an observation. Each column (p) is a feature.

        Return:
            A list of predicted labels for each observation in the input data.

        """
        return self._logr.predict(observations)
