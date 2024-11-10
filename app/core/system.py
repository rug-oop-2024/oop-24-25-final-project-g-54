from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry():
    """
    A registry for managing the storage and retrieval of artifacts
    including their metadata within a storage and database system.

    This class provides methods to register, list, retrieve
    and delete artifacts integrating storage and database
    functionalities to handle artifacts efficiently.

    """
    def __init__(self,
                 database: Database,
                 storage: Storage):
        """
        Initializes the ArtifactRegistry.

        Args:
            database (Database): The database for storing artifact metadata.
            storage (Storage): The storage system for managing artifact data.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact):
        """
        Registers an artifact by saving its data in
        storage and its metadata in the database.

        Args:
            artifact (Artifact): The artifact to register
            including its data and metadata.
        """
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        Lists all artifacts with an option to filter by type.

        Args:
            type (str): The type of artifacts to list.

        Returns:
            List[Artifact]: A list of artifacts matching the specified type.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieves an artifact by its ID loading its data from storage.

        Args:
            artifact_id (str): The ID of the artifact to retrieve.

        Returns:
            Artifact: The artifact corresponding to the specified ID.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str):
        """
        Deletes an artifact by its ID removing its data
        from storage and metadata from the database.

        Args:
            artifact_id (str): The ID of the artifact to delete.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """
    A singleton class representing an automated machine learning system.

    The system manages artifact storage, retrieval through a registry providing
    a centralized instance for coordinating storage and database operations.
    """
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database):
        """
        Initializes the AutoML system with storage and database components.

        Args:
            storage (LocalStorage): The storage system for
            managing artifact files.
            database (Database): The database for storing artifact metadata.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance():
        """
        Creates and retrieves the singleton instance of the AutoMLSystem.

        This method ensures that only one instance of the AutoMLSystem
        exists across the application managing the storage and database
        components through a single centralized system.

        Returns:
            AutoMLSystem: The singleton instance of the AutoML system.
    """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self):
        """
        Getter method for self._registry

        Returns:
            ArtifactRegistry: The registry used
            for managing artifacts in the system.
        """
        return self._registry
