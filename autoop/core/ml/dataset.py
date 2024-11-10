from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """
    A class representing a dataset artifact enabling
    storage, retrieval and conversion of tabular data
    between a pandas DataFrame and byte-encoded CSV format.

    This class extends the base Artifact class
    to handle dataset-specific functionality,
    including reading from and saving to a DataFrame format.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the dataset artifact with a type set to 'dataset'.

        Args:
            *args: Variable length argument
            list for superclass initialization.
            **kwargs: Arbitrary keyword arguments
            for superclass initialization.
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str,
                       version: str = "1_0_0") -> "Dataset":
        """
        Creates a Dataset instance from a pandas DataFrame.

        Args:
            data (DataFrame): The DataFrame containing the dataset.
            name (str): The name of the dataset artifact.
            asset_path (str): The path where the dataset will be stored.
            version: The version of the dataset artifact. Defaults to "1_0_0".

        Returns:
            Dataset: A Dataset object initialized
            with the encoded DataFrame data.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Reads the dataset's byte-encoded CSV data
        and converts it to a pandas DataFrame.

        Returns:
            DataFrame: The decoded DataFrame of the dataset.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Saves a pandas DataFrame to the
        dataset artifact in byte-encoded CSV format.

        Args:
            data(DataFrame): The DataFrame to encode and save.

        Returns:
            bytes: The encoded byte representation of the DataFrame.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)

    @staticmethod
    def from_artifact(artifact: Artifact) -> "Dataset":
        """
    Creates a Dataset instance from an existing Artifact.

    Args:
        artifact (Artifact): The artifact to convert into a Dataset.

    Returns:
        Dataset: A Dataset object initialized with
        the data and metadata from the provided artifact.
    """
        return Dataset(
            name=artifact.name,
            version=artifact.version,
            asset_path=artifact.asset_path,
            tags=artifact.tags,
            metadata=artifact.metadata,
            data=artifact.data
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset's stored
        byte-encoded CSV data into a pandas DataFrame.

        Returns:
            DataFrame: The decoded DataFrame representation of the dataset.
        """
        return self.read()
