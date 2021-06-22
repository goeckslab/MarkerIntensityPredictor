from pathlib import Path
import logging
import sys


class Prepare:

    @staticmethod
    def create_folders():
        try:
            path = Path("results/ae")
            path.mkdir(parents=True, exist_ok=True)

            path = Path("results/dae")
            path.mkdir(parents=True, exist_ok=True)

            path = Path("results/lr")
            path.mkdir(parents=True, exist_ok=True)

            path = Path("results/vae")
            path.mkdir(parents=True, exist_ok=True)
        except BaseException as ex:
            logging.info("Could not create path. Aborting!")
            print(ex)
            sys.exit(20)
