
from pydantic import BaseModel, Field
from typing import Literal
# import numpy as np

# from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    # attributes here
    name: str = Field()
    type: Literal["categorical", "numerical"] = Field()

    def __str__(self):
        return f"The feature '{self.name}' is '{self.type}'."
