import pandas as pd
from library.image_vae.image_vae_model import ImageVAE
from library.plotting.plots import Plotting
from pathlib import Path
from library.data.folder_management import FolderManagement
import argparse

base_path = Path("ImageVAE")


def get_args():
    """
       Load all provided cli args
       """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", action="store", required=True, help="The file used for testing")
    parser.add_argument("--folder", action="store", required=True,
                        help="The folder to use for training")
    parser.add_argument("--seed", "-s", action="store", help="The see to use", type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    FolderManagement.create_directory(path=base_path)

    try:
        plotter: Plotting = Plotting(base_path=base_path, args=None, use_mlflow=False)

        image_vae: ImageVAE = ImageVAE(embedding_dimensions=5, n_channels=60, filters=10, kernel_size=(640, 480),
                                       learning_rate=0.0001)

        image_vae.build_model()

        plotter.plot_model_architecture(image_vae.feature_encoder, "Feature Encoder")
        plotter.plot_model_architecture(image_vae.feature_decoder, "Feature Decoder")
        plotter.plot_model_architecture(image_vae.image_encoder, "Image Encoder")
        plotter.plot_model_architecture(image_vae.image_decoder, "Image Decoder")
        plotter.plot_model_architecture(image_vae.vae, "VAE")
        print("Plotting done")
    except BaseException as ex:
        raise
    finally:
        input()
        FolderManagement.delete_directory(path=base_path)

    # image_vae.train()
