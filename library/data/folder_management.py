from pathlib import Path
import shutil


class FolderManagement:

    @staticmethod
    def create_directory(path: Path, remove_if_exists=True) -> Path:
        try:
            if path.exists() and remove_if_exists:
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
            return path

        except BaseException as ex:
            raise

    @staticmethod
    def delete_directory(path: Path):
        if path.exists():
            shutil.rmtree(path)
