
from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
import pickle
from copy import deepcopy
from typing import Literal
from pydantic import PrivateAttr


class Model(ABC):

    _param: dict = PrivateAttr(default=dict)
    _type: Literal["classification", "regression"] = PrivateAttr(default=None)

    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, value: Literal["classification", "regression"]) -> None:
        if value not in ["classification", "regression"]:
            raise ValueError(
                f"Type not 'classification', or 'regression'. Got {value}."
                )
        self._type = value

    @property
    def parameters(self) -> dict:
        return deepcopy(self._param)

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        pass

    def to_artifact(self, name: str, asset_path: str) -> Artifact:
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
