# In artifact.py
from typing import Optional
import base64
from copy import deepcopy


class Artifact:
    """
    A class representing an artifact that
    can store model data, metadata, and unique identifiers.

    This class provides functionality for managing artifacts
    including generating unique IDs,
    storing data, and saving or reading data as bytes.
    """

    def generate_id(self) -> str:
        """
        Generates a unique ID for the artifact
        by encoding the artifact's name and version.

        Returns:
            str: A unique identifier for the artifact
            based on base64 encoding of its name and version.
        """
        base64_id = base64.b64encode(self.name.encode()).decode()
        return f"{base64_id}_{self.version}"

    def __init__(self, name: str = None, type: str = None,
                 asset_path: str = None, data: Optional[bytes] = None,
                 tags: list[str] = None, version: str = None,
                 metadata: dict = None) -> None:
        """
        Initializes the artifact with specified attributes.

        Args:
            name: The name of the artifact.
            type: The type of the artifact.
            asset_path: The file path where the artifact is stored.
            data: The binary data associated with the artifact.
            tags: A list of tags for categorizing or identifying the artifact.
            version: The version of the artifact.
            id: The unique identifier for the artifact.
            metadata: Additional metadata as key-value pairs.
        """
        self._name = name
        self._type = type
        self._asset_path = asset_path
        self._data = data
        self._tags = tags if tags is not None else []
        self._version = version
        self._id = self.generate_id()
        self._metadata = metadata if metadata is not None else {}

    @property
    def name(self) -> str:
        """
        Getter method for self._name
        """
        return self._name

    @property
    def type(self) -> str:
        """
        Getter method for self._type
        """
        return self._type

    @property
    def asset_path(self) -> str:
        """
        Getter method for self._asset_path
        """
        return self._asset_path

    @property
    def data(self) -> str:
        """
        Getter method for self._data
        """
        return self._data

    @property
    def tags(self) -> list:
        """
        Getter method for self._data

        Returns:
            Returns a deepcopy of the tags.
        """
        return deepcopy(self._tags)

    @property
    def version(self) -> str:
        """
        Getter method for self._version
        """
        return self._version

    @property
    def id(self) -> str:
        """
        Getter method for self._id
        """
        return self._id

    @property
    def metadata(self) -> dict:
        """
        Getter method for self._metadata

        Returns:
            Returns a deepcopy of the metadata.
        """

        return deepcopy(self._metadata)

    def read(self) -> bytes:
        """
        Reads and returns the artifact's data in bytes.

        Raises:
            ValueError: If the artifact has no data available.

        Returns:
            bytes: The binary data of the artifact.
        """
        if self.data is None:
            raise ValueError("No data available to read.")
        return self.data

    def save(self, data: Optional[bytes]) -> None:
        """
        Saves data to the artifact after validating it as bytes.

        Args:
            data: The binary data to store in the artifact.

        Raises:
            TypeError: If the data is not of type bytes.
        """
        if not isinstance(data, bytes):
            raise TypeError("Data should be in bytes.")
        self.data = data
