# In artifact.py
from pydantic import BaseModel, Field


class Artifact(BaseModel):
    name: str = Field()
    type: str = Field()
    asset_path: str = Field()
    version: str = Field("1.0.0")
    data: bytes = None

    def read(self) -> bytes:
        if self.data is None:
            raise ValueError("No data available to read.")
        return self.data

    def save(self, data) -> None:
        if not isinstance(data, bytes):
            raise TypeError("Data should be in bytes.")
        self.data = data
