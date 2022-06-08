from pathlib import Path
import shutil
from typing import Union


cdef class FolderManagement:

    @staticmethod
    def create_directory(path: Union[str, Path], remove_if_exists=True) -> Path:
        try:

            if isinstance(path, str):
                path: Path = Path(path)

            if path.exists() and remove_if_exists:
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
            return path

        except BaseException as ex:
            raise

    @staticmethod
    def delete_directory(path: Union[Path, str]):
        if isinstance(path, str):
            path: Path = Path(path)

        if path.exists():
            shutil.rmtree(path)
