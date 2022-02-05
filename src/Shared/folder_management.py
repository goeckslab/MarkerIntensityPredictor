from pathlib import Path
import logging
import sys
import shutil


class FolderManagement:

    @staticmethod
    def create_folders(vae_base_path: Path, ae_base_path: Path):
        try:
            FolderManagement.__create_vae_folders(vae_base_path)
            FolderManagement.__create_ae_folders(ae_base_path)


        except BaseException as ex:
            logging.info("Could not create path. Aborting!")
            print(ex)
            sys.exit(20)

    @staticmethod
    def __create_vae_folders(base_path: Path):
        try:
            base = Path(base_path)
            if base.exists():
                shutil.rmtree(base)
            base.mkdir(parents=True, exist_ok=True)

            base = Path(base_path, "generated")
            if base.exists():
                shutil.rmtree(base)
            base.mkdir(parents=True, exist_ok=True)


        except BaseException as ex:
            logging.info("Could not create path. Aborting!")
            print(ex)
            sys.exit(20)

    @staticmethod
    def __create_ae_folders(base_path: Path):
        try:
            base = Path(base_path)
            if base.exists():
                shutil.rmtree(base)
            base.mkdir(parents=True, exist_ok=True)



        except BaseException as ex:
            logging.info("Could not create path. Aborting!")
            print(ex)
            sys.exit(20)

    @staticmethod
    def create_directory(path: Path, remove_if_exists=True) -> Path:
        if remove_if_exists and path.exists():
            shutil.rmtree(path)

        path.mkdir(parents=True, exist_ok=True)
        return path
