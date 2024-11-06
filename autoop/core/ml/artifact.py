# In artifact.py
from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    name: str = Field(default=None)
    type: str = Field(default=None)
    asset_path: str = Field(default=None)
    data: bytes = Field(default=None)
    tags: list = Field(default_factory=list)
    version: str = Field(default=None)
    id: str = Field(default=None)
    metadata: dict = Field(default_factory=dict)

    def generate_id(self) -> str:
        base64_id = base64.b64encode(self.name.encode()).decode()
        return f"{base64_id}_{self.version}"

    def __init__(self, **data):
        super().__init__(**data)
        self.id = self.generate_id() 

    def read(self) -> bytes:
        if self.data is None:
            raise ValueError("No data available to read.")
        return self.data

    def save(self, data) -> None:
        if not isinstance(data, bytes):
            raise TypeError("Data should be in bytes.")
        self.data = data
        
