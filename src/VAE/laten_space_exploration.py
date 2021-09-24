import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorflow import keras
import pandas as pd
import argparse
import numpy as np


# Weights
# https://stackoverflow.com/questions/58364974/how-to-load-trained-autoencoder-weights-for-decoder

# Latent space exploration
# https://towardsdatascience.com/variational-autoencoders-as-generative-models-with-keras-e0c79415a7eb
# https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf

def explore_latent_space(args):
    """
    Explore the latent space
    """

    test_data = pd.read_csv(args.file)
    model = keras.models.load_model(Path("results", "vae", "Run_0", "model"))

    x_values = np.linspace(test_data.min(), test_data.max(), 30)

    count: int = 0
    for ix, x in enumerate(x_values):
        latent_point = x.reshape(1, x.shape[0])
        generated_cell = model.decoder.predict(latent_point)
        print(generated_cell)
        count += 1

    print(f"Generated {count} new cells.")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True, action="store",
                        help="The file to load and use")

    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()
    explore_latent_space(args)
