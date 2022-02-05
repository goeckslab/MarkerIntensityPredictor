from pathlib import Path
import shutil


class FolderManagement:

    @staticmethod
    def create_directory(path: Path, remove_if_exists=True) -> Path:
        if remove_if_exists and path.exists():
            shutil.rmtree(path)

        path.mkdir(parents=True, exist_ok=True)
        return path
