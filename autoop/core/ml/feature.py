
from pydantic import BaseModel, Field
from typing import Literal


class Feature(BaseModel):
    """
    A class representing a feature with a specified name and type.

    This class is used to define individual features with a name and type
    which can be either 'categorical' or 'numerical'.
    """

    name: str = Field()
    type: Literal["categorical", "numerical"] = Field()

    def __str__(self) -> str:
        """
        Returns a string representation of the feature's name and type.

        Returns:
            str: A description of the feature in the format
        """
        return f"The feature '{self.name}' is '{self.type}'."
