# In artifact.py
import base64


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

    def __init__(self,
                 name=None,
                 type=None,
                 asset_path=None,
                 data=None, tags=None,
                 version=None, id=None,
                 metadata=None):
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
        self.name = name
        self.type = type
        self.asset_path = asset_path
        self.data = data
        self.tags = tags if tags is not None else []
        self.version = version
        self.id = self.generate_id()
        self.metadata = metadata if metadata is not None else {}

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

    def save(self, data) -> None:
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
