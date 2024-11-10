from abc import ABC, abstractmethod
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "mean_absolute_error",
    "r-squared"
]


def get_metric(name: str) -> "Metric":
    """Gets the metric instance by name

    Args:
        name (str): The name of the metric to get

    Raises:
        ValueError: Error occurs if the provided name does not exist

    Returns:
        Metric: An instance of the metric
    """

    metrics_map = {
       "mean_squared_error": MeanSquaredError(),
       "accuracy": Accuracy(),
       "macro_precision": MacroPrecision(),
       "macro_recall": MacroRecall(),
       "mean_absolute_error": MeanAbsoluteError(),
       "r-squared": Rsquared()
    }

    if name not in metrics_map:
        raise ValueError(f"Unknown metric name: {name}")

    return metrics_map[name]


class Metric(ABC):
    """Base class for all metrics."""

    def __call__(self, y_ground: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the metric value.

        Args:
            y_ground (np.ndarray): True Values.
            y_pred (np.ndarray): Predicted Values.

        Returns:
            float: Calculated metric value
        """
        return self.evaluate(y_ground, y_pred)

    @abstractmethod
    def evaluate(self, y_ground: np.ndarray, y_pred: np.ndarray,) -> float:
        pass

# add here concrete implementations of the Metric class


class MeanSquaredError(Metric):
    """Class for MeanSquaredError. Inherits from Metric
    """

    def evaluate(self, y_ground: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the Mean Squared Error

        Args:
            y_ground (np.ndarray): True Values
            y_pred (np.ndarray): Predicted Values

        Returns:
            float: Calculated Mean Squared Error
        """
        return np.mean((y_ground - y_pred)**2)


class Accuracy(Metric):
    """Class for Accuracy. Inherits from Metric
    """

    def evaluate(self, y_ground: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the Accuracy

        Args:
            y_ground (np.ndarray): True Values
            y_pred (np.ndarray): Predicted Values

        Returns:
            float: Calculated Accuracy
        """
        return np.mean(y_ground == y_pred)


class MacroPrecision(Metric):
    """Class for MacroPrecision. Inherits from Metric
    """

    def evaluate(self, y_ground: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the Precision

        Args:
            y_ground (np.ndarray): True Values
            y_pred (np.ndarray): Predicted Values

        Returns:
            float: Calculated Precision
        """
        unique_classes = np.unique(y_ground)
        precision_scores = []

        for cls in unique_classes:
            true_positive = np.sum((y_pred == cls) & (y_ground == cls))
            predicted_p = np.sum(y_pred == cls)

            precision = true_positive / predicted_p if predicted_p > 0 else 0.0
            precision_scores.append(precision)

        return np.mean(precision_scores)


class MacroRecall(Metric):
    """Class for MacroRecall. Inherits from Metric
    """

    def evaluate(self, y_ground: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the Recall

        Args:
            y_ground (np.ndarray): True Values
            y_pred (np.ndarray): Predicted Values

        Returns:
            float: Calculated Recall
        """
        unique_classes = np.unique(y_ground)
        recall_scores = []

        for cls in unique_classes:
            true_positive = np.sum((y_pred == cls) & (y_ground == cls))
            actual_positive = np.sum(y_ground == cls)

            recall = (
                true_positive / actual_positive if actual_positive > 0 else 0.0
            )
            recall_scores.append(recall)

        return np.mean(recall_scores)


class MeanAbsoluteError(Metric):
    """Class for MeanAbsoluteError. Inherits from Metric
    """

    def evaluate(self, y_ground: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the Mean Absolute Error

        Args:
            y_ground (np.ndarray): True Values
            y_pred (np.ndarray): Predicted Values

        Returns:
            float: Calculated Mean Absolute Error
        """
        return np.mean(np.abs(y_ground - y_pred))


class Rsquared(Metric):
    """Class for Rsquared. Inherits from Metric
    """

    def evaluate(self, y_ground: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the R^2

        Args:
            y_ground (np.ndarray): True Values
            y_pred (np.ndarray): Predicted Values

        Returns:
            float: Calculated R^2
        """

        ss_tot = np.sum((y_ground - np.mean(y_ground)) ** 2)
        ss_res = np.sum((y_ground - y_pred) ** 2)

        return 1 - (ss_res / ss_tot)
