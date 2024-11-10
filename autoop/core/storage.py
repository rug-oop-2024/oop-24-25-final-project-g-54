from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """
    Exception raised when a specified path is not found.
    """
    def __init__(self, path) -> None:
        """
        Initializes the error with the missing path.

        Args:
            path (str): The path that was not found.
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    An abstract storage interface for saving,loading,deleting and listing data
    """

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """
    A storage class for managing local
    file operations including saving,loading,
    deleting and listing files under a specified base path.
    """

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initializes the local storage with
        a base directory path.
        If the base path does not exist
        it creates the necessary directory structure.

        Args:
            base_path (str): The root directory for storage operations.
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Saves data to the specified key within the base path.
        Creates any necessary parent directories.

        Args:
            data (bytes): The binary data to save.
            key (str): The relative path key within the base path where data
            will be stored.
        """
        path = self._join_path(key)
        print(path, key)
        # Ensure parent directories are created
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Loads and returns binary data from the specified key.

        Args:
            key (str): The relative path key within the base path from which
            to load data.

        Returns:
            bytes: The binary data loaded from the specified path.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Deletes the file at the specified key path.

        Args:
            key (str): The relative path key within the base path
            of the file to delete. Defaults to the root ("/").

        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        Lists all file paths under a
        specified prefix within the base path.

        Args:
            prefix (str, optional): The relative path prefix
            within the base path to list files under.
            Defaults to the root ("/").

        Returns:
            List[str]: A list of file paths relative to the base path.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        # Use os.path.join for compatibility across platforms
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path) for p in keys
                if os.path.isfile(p)]

    def _assert_path_exists(self, path: str) -> None:
        """
        Checks if a path exists and raises a NotFoundError if it does not.

        Args:
            path (str): The path to check.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Joins the specified path with the base path ensuring compatibility
        across operating systems.

        Args:
            path (str): The relative path to join with the base path.

        Returns:
            str: The full normalized path.
        """
        # Ensure paths are OS-agnostic
        return os.path.normpath(os.path.join(self._base_path, path))
