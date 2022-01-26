from pathlib import Path
import logging
import sys
import shutil


class FolderManagement:

    @staticmethod
    def create_folders(args):
        try:
            base = Path("VAE", "results")
            if base.exists():
                shutil.rmtree(base)
            base.mkdir(parents=True, exist_ok=True)

        except BaseException as ex:
            logging.info("Could not create path. Aborting!")
            print(ex)
            sys.exit(20)
