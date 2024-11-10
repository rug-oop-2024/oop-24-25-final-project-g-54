import numpy as np

from autoop.core.ml.model import Model


class MultipleLinearRegression(Model):
    """
    A model that predicts outcomes using multiple features.

    This model fits a linear equation to provided observed data and makes
    predictions based on learned parameters
    """

    def __init__(self) -> None:
        """
        Initializes the multiple linear regression model.
        """
        super().__init__()
        self._type = "regression"

    def _x_bar(self, observations: np.ndarray) -> np.ndarray:
        """
        Adds a bias term (intercept) to the observation matrix.

        Args:
            observations (ndarray): The input data, where each row (n) is
            an observation and each column (p) is a feature.

        Returns:
            An augmented matrix with an added column for the intercept term.
        """
        x_bar = np.hstack((observations, np.ones((observations.shape[0], 1))))
        return x_bar

    def fit(self, observations: np.ndarray,
            ground_truths: np.ndarray) -> None:
        """
        Calculates the optimal parameters for
        the linear model based on provided data.

        Args:
            observations (ndarray): A matrix of observations where
            each row (n) represents a sample and each column (p)
            represents a feature.
            ground_truths (ndarray): An array of true target values
            for each observation.

        """

        x_bar_T = np.transpose(self._x_bar(observations))
        inverse = np.linalg.inv(x_bar_T @ self._x_bar(observations))

        w_parameters = inverse @ (x_bar_T @ ground_truths)
        self._param = {"optimal_parameters": w_parameters}

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts outcomes based on the learned
        linear relationship from the fit method.

        Args:
            observations (ndarray): A matrix where each row (n)
            represents an observation and each column (p) represents a feature.

        Returns:
            A list of predicted values corresponding to each observation.
        """
        return self._x_bar(observations) @ self._param["optimal_parameters"]
