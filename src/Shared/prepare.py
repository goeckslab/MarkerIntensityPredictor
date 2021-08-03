from pathlib import Path
import logging
import sys
import shutil


class Prepare:

    @staticmethod
    def create_folders(args):
        try:
            base = Path("results")
            base.mkdir(parents=True, exist_ok=True)

            lr = Path("results/lr")
            if lr.exists() and args.remove:
                shutil.rmtree(lr)

            lr.mkdir(parents=True, exist_ok=True)

            ae = Path("results/ae")
            if ae.exists() and args.remove:
                shutil.rmtree(ae)
            ae.mkdir(parents=True, exist_ok=True)

            dae = Path("results/dae")
            if dae.exists() and args.remove:
                shutil.rmtree(dae)
            dae.mkdir(parents=True, exist_ok=True)

            vae = Path("results/vae")
            if vae.exists() and args.remove:
                shutil.rmtree(vae)
            vae.mkdir(parents=True, exist_ok=True)

            cluster = Path("results/cluster")
            if cluster.exists() and args.remove:
                shutil.rmtree(cluster)
            cluster.mkdir(parents=True, exist_ok=True)

            pca = Path("results/pca")
            if pca.exists() and args.remove:
                shutil.rmtree(pca)
            pca.mkdir(parents=True, exist_ok=True)

        except BaseException as ex:
            logging.info("Could not create path. Aborting!")
            print(ex)
            sys.exit(20)
