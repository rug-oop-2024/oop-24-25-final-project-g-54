
from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
import pickle
from copy import deepcopy
from typing import Literal
from pydantic import PrivateAttr


class Model(Artifact, ABC):
    """
    An abstract base class (ABC) for a generic machine learning model.

    This model provides a "blueprint" for building a machine learning
    algorithm.
    It uses "fit" and "predict" methods, which derived classes must implement.
    """

    _param: dict = PrivateAttr(default=dict)
    _type: Literal["classification", "regression"]

    def __init__(self, **kwargs):
        """
        Initializes the model with specified parameters.

        Args:
            **kwargs: Additional keyword arguments for configuring the model.
        """
        super(Artifact, self).__init__(**kwargs)

    @property
    def type(self) -> str:
        """
        Getter method for the model's type.

        Returns:
            str: A string indicating the model type
            which is either 'classification' or 'regression'.
        """
        return self._type

    @property
    def parameters(self) -> dict:
        """
        Getter method for the model's parameters.

        Deepcopy ensures that the original parameters are not modified when/if
        accessed. It prevents leakage.

        Returns:
            dict: A deep copy of the model's parameters as a sictionary.
        """
        return deepcopy(self._param)

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """
        Abstract method to fit and train the model with provided data.

        Args:
            observations (ndarray): The input data, a matrix where
            each row is an observation.
            ground_truths (ndarray): The ground truths or targets
            for input data.

            The method calculates parameters based on the provided data.
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Abstract method to make predictions based on input observations.

        Args:
            observations (ndarray): The input data, which predictions are
            to be made from.

        Returns:
            np.ndarray: The predicted values, a matrix or array.

        The method should use learned parameters from the "fit" method to
        make predictions for the input observations.
        """
        pass

    def to_artifact(self, name: str, asset_path: str) -> Artifact:
        """
        Converts the model to an artifact for
        saving including model type and parameters.

        Args:
            name (str): The name of the artifact.
            asset_path (str): The file path where the artifact will be stored.

        Returns:
            Artifact: An artifact object containing serialized model data.
        """
        model_data = {
            "type": self._type,
            "parameters": self._param
        }

        serialized_data = pickle.dumps(model_data)

        artifact = Artifact(
            name=name,
            type=self._type,
            asset_path=asset_path,
            data=serialized_data
        )
        return artifact
