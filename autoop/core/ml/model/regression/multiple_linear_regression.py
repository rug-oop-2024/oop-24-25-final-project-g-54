import numpy as np

from autoop.core.ml.model import Model


class MultipleLinearRegression(Model):

    def __init__(self) -> None:
        super().__init__()
        self._type = "regression"

    def _x_bar(self, observations: np.ndarray) -> np.ndarray:
        x_bar = np.hstack((observations, np.ones((observations.shape[0, 1]))))
        return x_bar

    def fit(self, observations: np.ndarray,
            ground_truths: np.ndarray) -> None:

        x_bar_T = np.transpose(self._x_bar(observations))
        inverse = np.linalg.inv(x_bar_T @ self._x_bar(observations))

        w_parameters = inverse @ (x_bar_T @ ground_truths)
        self._param = {"optimal_paramters": w_parameters}

    def predict(self, observations: np.ndarray) -> np.ndarray:
        return self._x_bar(observations) @ self._param["optimal_parameters"]
